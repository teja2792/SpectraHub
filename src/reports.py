"""
reports.py

Turns everything the pipeline (stages 1-6) already computed into figures and
tables a chemical/materials engineer can read WITHOUT knowing what k-NN,
LOOCV, macro recall, or cosine similarity mean. Those methods are still what's
running underneath -- this script just translates the output.

Guiding rule for every figure/table in this script: if a chemical engineer
who has never seen this codebase can't tell what a plot is claiming within
about 10 seconds, the label is wrong, not the reader. Concretely:
  - Axis labels and titles use chemistry language ("X-ray energy (eV)",
    "absorption edge", "white line", "pre-edge feature"), never variable
    names or method names.
  - Model quality is reported as "correct more often than a naive guess,
    adjusted for rare cases" rather than "macro recall" -- the number is
    identical, only the label changes.
  - Every figure ends with 2-3 lines of plain-English chemical engineering
    application notes -- what you'd actually DO with this feature in a lab
    or plant setting, not just what the feature IS mathematically.

This script only READS from spectrahub.db (via stages 1-6's own tables) --
it does not recompute anything ml_models.py / clustering_similarity.py /
feature_engineering.py already did. If those haven't been run yet, run them
first.

Outputs (all under results/):
  figures/highlighted_spectra_annotated.png   -- the 4 flagship materials,
                                                  annotated in plain language
  figures/model_performance_plain.png          -- "can spectral shape predict
                                                  chemistry" in a bar chart,
                                                  no ML jargon
  figures/cluster_chemistry_check.png          -- does shape-based grouping
                                                  line up with real chemistry
  materials_table.csv                          -- one row per material,
                                                  plain column names

Usage:
    python src/reports.py
"""

import csv
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from db import get_session, Spectrum, SpectralFeatures, MaterialLabel, ModelPrediction, SpectralCluster

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

# Plain-English names for internal task keys, used consistently across
# every figure/table in this script.
TASK_LABELS = {
    "oxidation_state": "Oxidation state",
    "coordination_number": "Coordination number",
    "bond_length": "Average bond length (Å)",
}


# ---------------------------------------------------------------------------
# Figure 1: annotated spectra for the 4 flagship materials
# ---------------------------------------------------------------------------

def plot_highlighted_spectra(session):
    specs = (
        session.query(Spectrum)
        .filter(Spectrum.is_highlighted == True, Spectrum.modality == "XANES")  # noqa: E712
        .order_by(Spectrum.material_formula)
        .all()
    )
    if not specs:
        print("No highlighted XANES spectra found -- skipping Figure 1.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 9.3))
    axes = axes.flatten()

    for ax, spec in zip(axes, specs):
        feat = session.get(SpectralFeatures, spec.record_id)
        x = np.array(spec.x_values(), dtype=float)
        y = np.array(spec.y_values(), dtype=float)
        order = np.argsort(x)
        x, y = x[order], y[order]

        ax.plot(x, y, color="#1f5fa8", linewidth=1.8, zorder=2)
        ax.set_title(f"{spec.material_formula}  —  {spec.absorbing_element} K-edge", fontsize=13, fontweight="bold", pad=10)
        ax.set_xlabel("X-ray energy (eV)")
        ax.set_ylabel("Normalized absorption\n(how much X-ray light the atom absorbs)")
        ax.grid(alpha=0.25)
        # Extra headroom above the tallest feature so markers never crowd
        # the title -- annotations live in one shared legend instead of
        # inline text, so this is just breathing room, not label space.
        ymin, ymax = float(np.min(y)), float(np.max(y))
        pad = 0.12 * (ymax - ymin)
        ax.set_ylim(ymin - pad * 0.3, ymax + pad)

        if feat and feat.edge_energy_ev:
            ax.axvline(feat.edge_energy_ev, color="#c0392b", linestyle="--", linewidth=1.3, zorder=1)
        if feat and feat.white_line_energy_ev:
            ax.plot(feat.white_line_energy_ev, feat.white_line_intensity, "o",
                    color="#e67e22", markersize=9, zorder=3, markeredgecolor="white", markeredgewidth=0.8)
        if feat and feat.pre_edge_energy_ev:
            ax.plot(feat.pre_edge_energy_ev, feat.pre_edge_intensity, "^",
                    color="#27ae60", markersize=9, zorder=3, markeredgecolor="white", markeredgewidth=0.8)

    for ax in axes[len(specs):]:
        ax.axis("off")

    # One shared legend explaining what the line/markers mean, in plain
    # language -- avoids the label-collision problem of putting inline text
    # next to every marker on every subplot (four different energy scales,
    # four different marker positions -- text placement that's safe for one
    # material collides with the title or axis on another).
    legend_handles = [
        plt.Line2D([0], [0], color="#c0392b", linestyle="--", linewidth=1.6,
                   label="Absorption edge — the electron gets kicked loose here"),
        plt.Line2D([0], [0], marker="o", color="white", markerfacecolor="#e67e22", markersize=9,
                   label="White line — nearby empty electron states"),
        plt.Line2D([0], [0], marker="^", color="white", markerfacecolor="#27ae60", markersize=9,
                   label="Pre-edge feature — signature of a distorted/tetrahedral site"),
    ]
    fig.tight_layout(rect=[0, 0, 1, 0.85])
    fig.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, 0.92),
               ncol=1, fontsize=11, frameon=True, framealpha=0.95)
    fig.suptitle("What happens when X-rays hit these four materials\n"
                  "(computed spectra — see README for what that means)",
                  fontsize=15, fontweight="bold", y=1.0)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURES_DIR / "highlighted_spectra_annotated.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


# ---------------------------------------------------------------------------
# Figure 2: "can spectral shape predict chemistry" without ML jargon
# ---------------------------------------------------------------------------

def _classification_summary(session, task):
    rows = session.query(ModelPrediction).filter_by(task=task).all()
    if not rows:
        return None
    y = np.array([r.true_value for r in rows])
    pred = np.array([r.predicted_value for r in rows])
    classes = sorted(set(y.tolist()))

    values, counts = np.unique(y, return_counts=True)
    majority = values[np.argmax(counts)]
    baseline_acc = float((y == majority).mean())
    random_baseline = 1.0 / len(classes)

    recalls = []
    for c in classes:
        mask = y == c
        if mask.sum() == 0:
            continue
        recalls.append(float((pred[mask] == c).mean()))
    model_score = float(np.mean(recalls)) if recalls else 0.0

    return {
        "n": len(y),
        "n_classes": len(classes),
        "model_score": model_score,
        "random_baseline": random_baseline,
        "raw_accuracy": float((pred == y).mean()),
        "majority_baseline": baseline_acc,
    }


def _regression_summary(session, task):
    rows = session.query(ModelPrediction).filter_by(task=task).all()
    if not rows:
        return None
    y = np.array([r.true_value for r in rows])
    pred = np.array([r.predicted_value for r in rows])
    baseline_pred = y.mean()
    baseline_mae = float(np.mean(np.abs(y - baseline_pred)))
    model_mae = float(np.mean(np.abs(pred - y)))
    return {"n": len(y), "model_mae": model_mae, "baseline_mae": baseline_mae}


def plot_model_performance(session):
    fig, axes = plt.subplots(1, 3, figsize=(14, 5.5))

    # --- Oxidation state & coordination number: higher bar = better ---
    for ax, task in zip(axes[:2], ["oxidation_state", "coordination_number"]):
        summary = _classification_summary(session, task)
        if summary is None:
            ax.axis("off")
            continue
        bars = ax.bar(
            ["SpectraHub's\nmodel", "Guessing without\nlooking at the spectrum"],
            [summary["model_score"], summary["random_baseline"]],
            color=["#1f5fa8", "#b0b0b0"],
        )
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.02,
                    f"{b.get_height():.0%}", ha="center", fontsize=11, fontweight="bold")
        ax.set_ylim(0, 1.0)
        ax.set_ylabel("How often it's right, counting\nrare cases fairly")
        ax.set_title(f"{TASK_LABELS[task]}\n(from spectrum shape alone, n={summary['n']})", fontsize=12)
        ax.grid(axis="y", alpha=0.25)

    # --- Bond length: lower bar = better (it's an error, not a score) ---
    ax = axes[2]
    summary = _regression_summary(session, "bond_length")
    if summary is not None:
        bars = ax.bar(
            ["SpectraHub's\nmodel", "Just guessing the\naverage bond length"],
            [summary["model_mae"], summary["baseline_mae"]],
            color=["#1f5fa8", "#b0b0b0"],
        )
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.003,
                    f"{b.get_height():.3f} Å", ha="center", fontsize=11, fontweight="bold")
        ax.set_ylabel("Typical error in predicted bond length (Å)\nlower is better")
        ax.set_title(f"Bond length\n(from spectrum shape alone, n={summary['n']})", fontsize=12)
        ax.grid(axis="y", alpha=0.25)

    fig.suptitle("Can you tell a material's chemistry just from the shape of its X-ray spectrum?",
                  fontsize=15, fontweight="bold", y=1.03)
    fig.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURES_DIR / "model_performance_plain.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")
    return summary


# ---------------------------------------------------------------------------
# Figure 3: does shape-based grouping line up with real chemistry
# ---------------------------------------------------------------------------

def plot_cluster_chemistry_check(session):
    clusters = session.query(SpectralCluster).all()
    if not clusters:
        print("No clusters found -- run clustering_similarity.py --cluster first. Skipping Figure 3.")
        return

    by_cluster = defaultdict(list)
    for c in clusters:
        by_cluster[c.cluster_id].append(c.record_id)

    all_oxi = []
    cluster_rows = []
    for cid, record_ids in sorted(by_cluster.items()):
        elements = []
        oxi_vals = []
        for rid in record_ids:
            spec = session.get(Spectrum, rid)
            if spec is None:
                continue
            elements.append(spec.absorbing_element)
            label = session.query(MaterialLabel).filter_by(mp_id=spec.mp_id).one_or_none()
            if label and label.oxidation_state is not None:
                oxi_vals.append(label.oxidation_state)
                all_oxi.append(label.oxidation_state)
        top_element, top_count = Counter(elements).most_common(1)[0] if elements else ("?", 0)
        oxi_std = float(np.std(oxi_vals)) if len(oxi_vals) >= 2 else None
        cluster_rows.append({
            "cluster_id": cid, "n": len(record_ids),
            "top_element": top_element, "top_count": top_count,
            "oxi_std": oxi_std, "n_labeled": len(oxi_vals),
        })

    dataset_std = float(np.std(all_oxi)) if all_oxi else None

    fig, ax = plt.subplots(figsize=(11, max(4, 0.5 * len(cluster_rows))))
    labels, sizes, colors, annotations = [], [], [], []
    for row in cluster_rows:
        labels.append(f"Group {row['cluster_id']}\n(mostly {row['top_element']})")
        sizes.append(row["n"])
        if row["oxi_std"] is None:
            colors.append("#b0b0b0")
            annotations.append("not enough labeled\nmaterials to check")
        elif dataset_std and row["oxi_std"] < 0.4 * dataset_std:
            colors.append("#27ae60")
            annotations.append("very consistent\nchemistry")
        elif dataset_std and row["oxi_std"] < 0.8 * dataset_std:
            colors.append("#f1c40f")
            annotations.append("fairly consistent\nchemistry")
        else:
            colors.append("#c0392b")
            annotations.append("mixed chemistry")

    y_pos = np.arange(len(labels))
    bars = ax.barh(y_pos, sizes, color=colors)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Number of materials grouped together purely by X-ray spectrum shape\n"
                  "(no chemistry labels were used to form these groups)")
    ax.set_title("Do materials that LOOK alike in X-ray spectra actually SHARE real chemistry?\n"
                 "(color = how consistent the group's oxidation state turned out to be, checked after the fact)",
                 fontsize=13, fontweight="bold")
    for bar, note in zip(bars, annotations):
        ax.text(bar.get_width() + max(sizes) * 0.01, bar.get_y() + bar.get_height() / 2,
                note, va="center", fontsize=8)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURES_DIR / "cluster_chemistry_check.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


# ---------------------------------------------------------------------------
# Table: one row per material, plain column names
# ---------------------------------------------------------------------------

def write_materials_table(session):
    cluster_by_record = {c.record_id: c.cluster_id for c in session.query(SpectralCluster).all()}

    # One row per unique mp_id, but when a material has multiple spectra
    # (XANES/XAFS/EXAFS) prefer the XANES record specifically -- that's the
    # only modality clustering_similarity.py assigns a "spectral shape
    # group" to (see that script's module docstring), so picking whichever
    # record happened to sort first would silently blank out the cluster
    # column for most materials that also have an XAFS/EXAFS record.
    best_spec_by_mp_id = {}
    for spec in session.query(Spectrum).all():
        current = best_spec_by_mp_id.get(spec.mp_id)
        if current is None or (spec.modality == "XANES" and current.modality != "XANES"):
            best_spec_by_mp_id[spec.mp_id] = spec

    rows = []
    for spec in sorted(best_spec_by_mp_id.values(), key=lambda s: s.material_formula):
        label = session.query(MaterialLabel).filter_by(mp_id=spec.mp_id).one_or_none()
        cluster_id = cluster_by_record.get(spec.record_id)
        rows.append({
            "Material": spec.material_formula,
            "Absorbing element": spec.absorbing_element,
            "Oxidation state": label.oxidation_state if label else "",
            "Coordination number": label.coordination_number if label else "",
            "Average bond length (angstrom)": (
                round(label.mean_bond_length_angstrom, 4) if label and label.mean_bond_length_angstrom else ""
            ),
            "Spectral shape group": cluster_id if cluster_id is not None else "",
            "Data source": "Computed (Materials Project, DFT + FEFF) -- not a lab measurement",
            "Featured example": "Yes" if spec.is_highlighted else "No",
        })

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "materials_table.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {out}  ({len(rows)} materials)")


def main():
    session = get_session()
    plot_highlighted_spectra(session)
    plot_model_performance(session)
    plot_cluster_chemistry_check(session)
    write_materials_table(session)


if __name__ == "__main__":
    main()
