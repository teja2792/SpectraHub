"""
clustering_similarity.py

Unsupervised clustering and nearest-neighbor similarity search over XANES
spectral shape -- "find materials with a similar XANES fingerprint to X."
No labels required (that's the point of doing this before the supervised
models), but where labels exist (task 2's oxidation_state/coordination
data) this script also uses them purely for VERIFYING that shape-based
clusters correspond to real chemistry, not for the clustering itself.

Design decisions, stated explicitly:
  - XANES ONLY, not XAFS/EXAFS. XANES records are all fetched with dense,
    consistent sampling right at the edge (100 points over a narrow
    window); XAFS/EXAFS cover a much wider energy range at coarser
    resolution, tuned for EXAFS oscillation analysis, not near-edge shape
    comparison. Mixing modalities into one similarity space would compare
    apples to oranges.
  - Spectra are EDGE-ALIGNED before comparison: each spectrum's energy
    axis is re-centered on its own edge_energy_ev (from
    feature_engineering.py) and interpolated onto a common relative-energy
    grid, REL_ENERGY_GRID eV relative to the edge. This is required to
    compare spectra across different elements/edges at all -- Cu K-edge
    (~9000 eV) and Ce K-edge (~40500 eV) are meaningless to compare on
    absolute energy, but their near-edge SHAPE (driven by local
    coordination/oxidation state, not absolute edge position) is
    comparable once aligned. This is standard practice in XAS fingerprint
    matching (e.g. linear combination fitting against reference spectra).
  - Grid range [-10, +30] eV relative to edge: chosen from actual data
    coverage (checked against all 634 XANES records -- 611/634 have
    >=30 eV of real data above the edge; pre-edge coverage is much
    thinner, median 10.8 eV). Points outside a spectrum's actual data
    range are clamped to the nearest real boundary value (np.interp's
    left=/right=), not extrapolated by shape -- an honest "assume flat"
    approximation, not a fabricated curve.
  - Each aligned vector is MEAN-CENTERED (per-row, subtract that row's
    own mean) before clustering or similarity comparison. Discovered by
    testing, not assumed: raw cosine similarity on un-centered vectors
    is dominated by the generic "edge step from ~0 to ~1" that nearly
    every XANES spectrum shares regardless of element, so almost
    everything scored >0.98 similar to almost everything else -- e.g.
    Cu2O's raw-vector top matches included Ti- and Cr-based oxides
    ranked nearly as high as chemically related Cu compounds. Mean-
    centering removes that shared trend and leaves the actual
    distinguishing shape (white-line sharpness, pre-edge shoulders,
    oscillation pattern), which is what "fingerprint similarity" should
    mean.
  - K-means implemented by hand (k-means++ init + Lloyd's algorithm, best
    of n_init random restarts by inertia) instead of scikit-learn. This
    project has hit repeated pip install failures/timeouts for optional
    packages in the dev sandbox this was built in; a ~40-line k-means on
    an 81-dimensional, 634-row dataset is well within what's safe to
    hand-roll and verify directly, so that risk isn't worth taking on
    just for this. Similarity search is exact brute-force cosine
    similarity -- at this data scale (634 XANES records) there's no
    reason to reach for approximate nearest-neighbor libraries.

Usage:
    python src/clustering_similarity.py --cluster --k 12
    python src/clustering_similarity.py --similar-to Cu2O_Cu_Kedge_XANES_aaaaaanx --top 10
"""

import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from db import get_session, Spectrum, SpectralFeatures, MaterialLabel, SpectralCluster

REL_ENERGY_GRID = np.linspace(-10, 30, 81)  # eV relative to edge_energy_ev


def build_aligned_matrix(session):
    """Returns (record_ids, X) for every XANES record with a computed
    edge_energy_ev. X.shape == (n_records, len(REL_ENERGY_GRID))."""
    rows = (
        session.query(Spectrum, SpectralFeatures)
        .join(SpectralFeatures, Spectrum.record_id == SpectralFeatures.record_id)
        .filter(Spectrum.modality == "XANES")
        .all()
    )
    record_ids, vectors = [], []
    for spec, feat in rows:
        if feat is None or feat.edge_energy_ev is None:
            continue
        x = np.array(spec.x_values(), dtype=float)
        y = np.array(spec.y_values(), dtype=float)
        order = np.argsort(x)
        x, y = x[order], y[order]
        rel_x = x - feat.edge_energy_ev
        aligned = np.interp(REL_ENERGY_GRID, rel_x, y, left=y[0], right=y[-1])
        record_ids.append(spec.record_id)
        vectors.append(aligned)
    X = np.array(vectors)
    # Mean-center each row -- see module docstring for why this is
    # required, not optional (raw un-centered cosine similarity was
    # dominated by the shared generic edge-step trend).
    X_centered = X - X.mean(axis=1, keepdims=True)
    return record_ids, X_centered


def cosine_similarities(X, idx):
    """Cosine similarity of row idx against every row of X, including
    itself (will be 1.0 at position idx)."""
    v = X[idx]
    norms = np.linalg.norm(X, axis=1) * np.linalg.norm(v)
    norms[norms == 0] = 1e-12
    return (X @ v) / norms


def kmeans(X, k, n_init=10, max_iter=300, seed=0):
    """k-means++ init + Lloyd's algorithm, best of n_init restarts by
    inertia. See module docstring for why this is hand-rolled rather than
    scikit-learn. Returns (labels, centroids, inertia)."""
    rng = np.random.default_rng(seed)
    best_labels = best_centroids = None
    best_inertia = np.inf

    for _ in range(n_init):
        centroids = np.empty((k, X.shape[1]))
        centroids[0] = X[rng.integers(0, len(X))]
        closest_sq_dist = np.sum((X - centroids[0]) ** 2, axis=1)
        for i in range(1, k):
            total = closest_sq_dist.sum()
            probs = closest_sq_dist / total if total > 0 else np.full(len(X), 1 / len(X))
            next_idx = rng.choice(len(X), p=probs)
            centroids[i] = X[next_idx]
            new_sq_dist = np.sum((X - centroids[i]) ** 2, axis=1)
            closest_sq_dist = np.minimum(closest_sq_dist, new_sq_dist)

        labels = np.full(len(X), -1)
        for iteration in range(max_iter):
            dists = np.linalg.norm(X[:, None, :] - centroids[None, :, :], axis=2)
            new_labels = np.argmin(dists, axis=1)
            if np.array_equal(new_labels, labels):
                break
            labels = new_labels
            for c in range(k):
                mask = labels == c
                if mask.any():
                    centroids[c] = X[mask].mean(axis=0)

        inertia = sum(float(np.sum((X[labels == c] - centroids[c]) ** 2)) for c in range(k))
        if inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels.copy()
            best_centroids = centroids.copy()

    return best_labels, best_centroids, best_inertia


def _material_info(session, record_id):
    spec = session.query(Spectrum).filter_by(record_id=record_id).one_or_none()
    if spec is None:
        return None
    label = session.query(MaterialLabel).filter_by(mp_id=spec.mp_id).one_or_none()
    return {
        "record_id": record_id,
        "formula": spec.material_formula,
        "element": spec.absorbing_element,
        "oxidation_state": label.oxidation_state if label else None,
        "coordination_number": label.coordination_number if label else None,
    }


def run_similarity_search(session, record_ids, X, query_record_id, top_n):
    if query_record_id not in record_ids:
        print(f"'{query_record_id}' not found among XANES records with computed features.")
        return
    idx = record_ids.index(query_record_id)
    sims = cosine_similarities(X, idx)
    order = np.argsort(-sims)

    query_info = _material_info(session, query_record_id)
    print(f"Query: {query_info['formula']} ({query_info['element']}, "
          f"oxidation_state={query_info['oxidation_state']}, "
          f"coordination_number={query_info['coordination_number']})")
    print(f"Top {top_n} most similar XANES fingerprints (excluding self):")
    shown = 0
    for i in order:
        if record_ids[i] == query_record_id:
            continue
        info = _material_info(session, record_ids[i])
        print(f"  {sims[i]:.4f}  {info['formula']:12s} {info['element']:3s} "
              f"oxi={info['oxidation_state']}  CN={info['coordination_number']}  ({record_ids[i]})")
        shown += 1
        if shown >= top_n:
            break


def run_clustering(session, record_ids, X, k, seed=0):
    labels, centroids, inertia = kmeans(X, k, seed=seed)
    print(f"K-means with k={k}: inertia={inertia:.4f}")

    now = datetime.now(timezone.utc).isoformat()
    for record_id, cluster_id in zip(record_ids, labels):
        existing = session.get(SpectralCluster, record_id)
        if existing is None:
            session.add(SpectralCluster(
                record_id=record_id, cluster_id=int(cluster_id),
                algorithm=f"kmeans k={k}", computed_at_utc=now,
            ))
        else:
            existing.cluster_id = int(cluster_id)
            existing.algorithm = f"kmeans k={k}"
            existing.computed_at_utc = now
    session.commit()

    # Verification: do shape-based clusters correspond to real chemistry?
    # For each cluster, report its dominant absorbing elements and the
    # spread of known oxidation states -- tight, low-spread clusters
    # (compared to the dataset-wide spread) are evidence the shape
    # grouping tracks real chemical similarity, not noise.
    print()
    print("Cluster summary (chemistry check, not part of the algorithm itself):")
    all_oxi = []
    for record_id in record_ids:
        info = _material_info(session, record_id)
        if info["oxidation_state"] is not None:
            all_oxi.append(info["oxidation_state"])
    dataset_oxi_std = float(np.std(all_oxi)) if all_oxi else None
    print(f"  dataset-wide oxidation_state std dev: {dataset_oxi_std}")

    for c in range(k):
        members = [rid for rid, lab in zip(record_ids, labels) if lab == c]
        infos = [_material_info(session, rid) for rid in members]
        elements = [i["element"] for i in infos]
        elem_counts = {}
        for e in elements:
            elem_counts[e] = elem_counts.get(e, 0) + 1
        top_elems = sorted(elem_counts.items(), key=lambda kv: -kv[1])[:3]
        oxi_vals = [i["oxidation_state"] for i in infos if i["oxidation_state"] is not None]
        oxi_std = float(np.std(oxi_vals)) if len(oxi_vals) >= 2 else None
        print(f"  cluster {c}: n={len(members)}  top elements={top_elems}  "
              f"oxidation_state std={oxi_std} (n_labeled={len(oxi_vals)})")


def main():
    parser = argparse.ArgumentParser(description="Cluster and search XANES spectra by shape.")
    parser.add_argument("--db", default=None)
    parser.add_argument("--cluster", action="store_true", help="Run k-means and store cluster assignments.")
    parser.add_argument("--k", type=int, default=12)
    parser.add_argument("--similar-to", default=None, help="record_id to find similar XANES fingerprints for.")
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    db_path = Path(args.db) if args.db else None
    session = get_session(db_path)

    record_ids, X = build_aligned_matrix(session)
    print(f"{len(record_ids)} XANES records with computed features, aligned to a "
          f"{X.shape[1]}-point relative-energy grid.")
    if not record_ids:
        return

    if args.similar_to:
        run_similarity_search(session, record_ids, X, args.similar_to, args.top)
    if args.cluster:
        run_clustering(session, record_ids, X, args.k, seed=args.seed)
    if not args.similar_to and not args.cluster:
        print("Nothing to do -- pass --cluster and/or --similar-to <record_id>.")


if __name__ == "__main__":
    main()
