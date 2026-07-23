"""
ml_models.py

Supervised models predicting oxidation_state (classification),
coordination_number (classification), and mean_bond_length_angstrom
(regression) from engineered XANES spectral-shape features -- "can we
infer structure/electronic state from the spectrum alone," the reverse
direction of feature_engineering.py.

Design decisions, stated explicitly:
  - Training data: only XANES records that have BOTH computed
    spectral_features AND a label (task 2) for their mp_id. That's 539
    materials -- 475 with oxidation_state, 539 with coordination_number
    and bond_length (every material with a label has both of those two,
    since label_fetch.py only skips a material entirely when NEITHER is
    available).
  - Feature vector deliberately EXCLUDES raw edge_energy_eV. Absolute
    edge energy is essentially a lookup table for which element it is
    (Cu K-edge is always ~8985 eV, Ce K-edge is always ~40500 eV
    regardless of oxidation state) -- including it would let the model
    "cheat" by learning elemental identity instead of learning whether
    spectral SHAPE encodes chemistry, which is the actual question here.
    White-line and pre-edge positions are used as OFFSETS from the edge
    (eV relative to edge_energy_ev), not absolute energies, for the same
    reason -- and because offsets are what's physically comparable across
    different elements/edges in the first place (see
    clustering_similarity.py's docstring for the same reasoning).
    Feature vector: [edge_jump, max_derivative, white_line_offset_ev,
    white_line_intensity, has_pre_edge, pre_edge_offset_ev,
    pre_edge_intensity]. has_pre_edge is an explicit 0/1 flag because
    pre_edge is only populated for ~40% of XANES records (see
    feature_engineering.py) -- imputing 0 for the offset/intensity
    without a flag would silently tell the model "no pre-edge" and
    "pre-edge exactly at the edge" are the same thing, which they aren't.
  - oxidation_state is ROUNDED to the nearest integer for classification.
    100/475 (21%) of label_fetch.py's oxidation_state values are
    non-integer averages (from the Composition.oxi_state_guesses
    fallback when Bond Valence Analysis fails -- see label_fetch.py).
    Treating oxidation_state as a small number of integer classes matches
    how it's conventionally used and discussed; a continuous regression
    target would be defensible too, this is a stated choice, not the only
    valid one.
  - Model: hand-rolled k-NN (classification by majority vote,
    regression by mean of neighbors), NOT scikit-learn. This project has
    repeatedly hit pip install failures/timeouts for optional packages in
    the dev sandbox this was built in (scipy for feature_engineering.py,
    scikit-learn attempted here too) -- k-NN on a ~500-row, 7-feature
    dataset is simple enough to hand-roll correctly and verify directly,
    same reasoning as clustering_similarity.py's hand-rolled k-means. A
    tree ensemble (Random Forest / Gradient Boosting) would likely
    outperform k-NN on tabular data like this -- if scikit-learn installs
    cleanly in your environment, swapping it in is a reasonable upgrade,
    not a requirement.
  - Evaluation: Leave-One-Out Cross-Validation (LOOCV), not a single
    train/test split -- at ~500-540 samples a single split leaves either
    training or evaluation data uncomfortably small, and LOOCV uses every
    sample for both without the variance of an arbitrary split. Every
    individual LOOCV prediction is stored in `model_predictions` (not
    just an aggregate accuracy number) so results are auditable --  you
    can look up exactly which materials the model got wrong.
  - Features are standardized (z-score) using GLOBAL mean/std computed
    once, not recomputed per LOOCV fold. With ~500+ samples, leaving one
    out changes the global mean/std negligibly -- a stated simplification
    for code clarity, not a rigor claim.
  - k is chosen by sweeping k in {1,3,5,7,9,15,21} and picking the best
    LOOCV score. This is a mild form of information leakage (k selected
    using the same metric being reported) -- standard practice for
    lightweight model selection at this data scale, but stated plainly
    rather than presented as an untouched holdout result.
  - Every reported metric is compared against a naive baseline (majority-
    class accuracy for classification, mean-prediction MAE for
    regression) so "65% accuracy" can be judged against what a model
    that learned nothing would already get.

Usage:
    python src/ml_models.py
"""

import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from db import get_session, Spectrum, SpectralFeatures, MaterialLabel, ModelPrediction

K_SWEEP = [1, 3, 5, 7, 9, 15, 21]


def assemble_dataset(session):
    """One row per XANES material with computed spectral_features and a
    label row for its mp_id (oxidation_state may still be None on some
    rows -- coordination_number/bond_length are not, per label_fetch.py's
    own "nothing usable" filter)."""
    rows = (
        session.query(Spectrum, SpectralFeatures, MaterialLabel)
        .join(SpectralFeatures, Spectrum.record_id == SpectralFeatures.record_id)
        .join(MaterialLabel, Spectrum.mp_id == MaterialLabel.mp_id)
        .filter(Spectrum.modality == "XANES")
        .all()
    )
    dataset = []
    for spec, feat, label in rows:
        has_pre_edge = feat.pre_edge_energy_ev is not None
        features = np.array([
            feat.edge_jump,
            feat.max_derivative,
            feat.white_line_energy_ev - feat.edge_energy_ev,
            feat.white_line_intensity,
            1.0 if has_pre_edge else 0.0,
            (feat.pre_edge_energy_ev - feat.edge_energy_ev) if has_pre_edge else 0.0,
            feat.pre_edge_intensity if has_pre_edge else 0.0,
        ], dtype=float)
        dataset.append({
            "record_id": spec.record_id,
            "formula": spec.material_formula,
            "features": features,
            "oxidation_state": round(label.oxidation_state) if label.oxidation_state is not None else None,
            "coordination_number": label.coordination_number,
            "bond_length": label.mean_bond_length_angstrom,
        })
    return dataset


def _standardize(X):
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0] = 1.0
    return (X - mean) / std


def knn_classify_loocv(X, y, k):
    n = len(X)
    preds = np.empty(n, dtype=y.dtype)
    for i in range(n):
        dists = np.linalg.norm(X - X[i], axis=1)
        dists[i] = np.inf
        nearest = np.argsort(dists)[:k]
        values, counts = np.unique(y[nearest], return_counts=True)
        preds[i] = values[np.argmax(counts)]
    return preds


def knn_regress_loocv(X, y, k):
    n = len(X)
    preds = np.empty(n)
    for i in range(n):
        dists = np.linalg.norm(X - X[i], axis=1)
        dists[i] = np.inf
        nearest = np.argsort(dists)[:k]
        preds[i] = y[nearest].mean()
    return preds


def run_classification_task(session, dataset, target_key, task_name):
    filtered = [d for d in dataset if d[target_key] is not None]
    X = _standardize(np.array([d["features"] for d in filtered]))
    y = np.array([d[target_key] for d in filtered])

    values, counts = np.unique(y, return_counts=True)
    majority = values[np.argmax(counts)]
    baseline_acc = float((y == majority).mean())

    best_k, best_acc, best_preds = None, -1.0, None
    for k in K_SWEEP:
        if k >= len(y):
            continue
        preds = knn_classify_loocv(X, y, k)
        acc = float((preds == y).mean())
        if acc > best_acc:
            best_k, best_acc, best_preds = k, acc, preds

    print(f"{task_name}: n={len(y)}  majority-class baseline={baseline_acc:.3f}  "
          f"k-NN (k={best_k}) LOOCV accuracy={best_acc:.3f}  "
          f"(lift over baseline: {best_acc - baseline_acc:+.3f})")

    _store_predictions(session, task_name, f"knn k={best_k} (LOOCV)", filtered, best_preds, y)
    return best_k, best_acc, baseline_acc


def run_regression_task(session, dataset, target_key, task_name):
    filtered = [d for d in dataset if d[target_key] is not None]
    X = _standardize(np.array([d["features"] for d in filtered]))
    y = np.array([d[target_key] for d in filtered])

    baseline_pred = y.mean()
    baseline_mae = float(np.mean(np.abs(y - baseline_pred)))

    best_k, best_mae, best_preds = None, np.inf, None
    for k in K_SWEEP:
        if k >= len(y):
            continue
        preds = knn_regress_loocv(X, y, k)
        mae = float(np.mean(np.abs(preds - y)))
        if mae < best_mae:
            best_k, best_mae, best_preds = k, mae, preds

    print(f"{task_name}: n={len(y)}  mean-prediction baseline MAE={baseline_mae:.4f}  "
          f"k-NN (k={best_k}) LOOCV MAE={best_mae:.4f}  "
          f"(improvement over baseline: {baseline_mae - best_mae:+.4f})")

    _store_predictions(session, task_name, f"knn k={best_k} (LOOCV)", filtered, best_preds, y)
    return best_k, best_mae, baseline_mae


def _store_predictions(session, task_name, method, filtered, preds, y_true):
    session.query(ModelPrediction).filter_by(task=task_name).delete()
    now = datetime.now(timezone.utc).isoformat()
    for d, pred, true in zip(filtered, preds, y_true):
        session.add(ModelPrediction(
            record_id=d["record_id"], task=task_name, method=method,
            predicted_value=float(pred), true_value=float(true), computed_at_utc=now,
        ))
    session.commit()


def main():
    parser = argparse.ArgumentParser(description="Train/evaluate supervised models on XANES spectral features.")
    parser.add_argument("--db", default=None)
    args = parser.parse_args()

    db_path = Path(args.db) if args.db else None
    session = get_session(db_path)

    dataset = assemble_dataset(session)
    print(f"{len(dataset)} XANES materials with computed features and a label.")
    if not dataset:
        return

    print()
    run_classification_task(session, dataset, "oxidation_state", "oxidation_state")
    run_classification_task(session, dataset, "coordination_number", "coordination_number")
    run_regression_task(session, dataset, "bond_length", "bond_length")


if __name__ == "__main__":
    main()
