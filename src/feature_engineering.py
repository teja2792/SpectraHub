"""
feature_engineering.py

Derives numerical features from each raw XANES/XAFS/EXAFS spectrum
(energy_eV vs normalized_absorption arrays already in the `spectra`
table) and writes them to `spectral_features`, keyed on record_id. These
feed both the unsupervised clustering/similarity search (task 4) and the
supervised models (task 5) -- comparing raw, unaligned energy/absorption
arrays directly is not a sound basis for either.

IMPORTANT -- what this is and isn't:
  This is a lightweight, dependency-free (pure numpy, no scipy) heuristic
  implementation, NOT a rigorous XAS analysis package like Athena/Larch.
  Real XAS analysis uses careful background subtraction (linear/quadratic
  pre-edge and post-edge fits extrapolated across the whole spectrum) for
  edge-jump normalization, and dedicated peak-fitting (pseudo-Voigt, etc.)
  for pre-edge/white-line characterization. This script instead uses
  simple percentile-window averages and local-maxima detection on the
  already-normalized absorption values MP provides. That's a reasonable
  and honest basis for ML input features (relative comparisons across
  the dataset), but the absolute numbers should not be quoted as
  publication-grade XAS analysis results.

Algorithm, stated explicitly:
  - edge_energy_eV: energy at the point of maximum dy/dx (standard E0
    convention -- steepest point of the absorption rise).
  - max_derivative: the dy/dx value at that point (characterizes edge
    sharpness -- steeper edges are more "sudden" electronic transitions).
  - edge_jump: mean(y) in the top 20% of the record's energy range minus
    mean(y) in the bottom 10% -- a coarse proxy for the absorption jump
    magnitude, NOT a properly background-subtracted edge jump.
  - white_line_energy_eV / white_line_intensity: position and value of
    the maximum y within [edge_energy, edge_energy + 30 eV] (or up to
    the record's max energy if the window is narrower than that). This
    is reported even when there's no sharp resolved peak -- for a K-edge
    without a strong white line it's just "brightest point right after
    the edge", which is still a meaningful, well-defined ML feature.
  - pre_edge_energy_eV / pre_edge_intensity: position and value of a
    genuine LOCAL MAXIMUM (y[i] > both neighbors) within
    [edge_energy - 25 eV, edge_energy - 2 eV], intersected with whatever
    data actually exists below the edge. Unlike white_line, this is only
    populated when a real local peak is found -- reporting the highest
    point of a monotonically-rising pre-edge baseline as a "peak" would
    be actively misleading (there's no real pre-edge transition there).
    Requires >= 5 eV of data below the edge to attempt at all (checked:
    across all 634 XANES records, coverage is real but partial -- median
    ~10.8 eV of pre-edge room, 424/634 have >=5 eV, 339/634 have >=10 eV;
    the rest simply don't have pre-edge data because MP's fetched window
    starts at or very near the edge itself for many materials). This
    script reports real coverage numbers at the end, same convention as
    label_fetch.py.

Usage:
    python src/feature_engineering.py
    python src/feature_engineering.py --limit 20
"""

import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from db import get_session, Spectrum, SpectralFeatures

PRE_EDGE_MIN_MARGIN_EV = 5.0
PRE_EDGE_MAX_WINDOW_EV = 25.0
PRE_EDGE_EXCLUSION_EV = 2.0   # don't search right up against the edge itself
WHITE_LINE_WINDOW_EV = 30.0


def _local_maxima(y):
    """Indices i where y[i] > y[i-1] and y[i] > y[i+1] -- dependency-free
    stand-in for scipy.signal.find_peaks (not available in this
    environment; see module docstring)."""
    if len(y) < 3:
        return []
    return [i for i in range(1, len(y) - 1) if y[i] > y[i - 1] and y[i] > y[i + 1]]


def compute_features(x, y):
    """x, y: 1D numpy arrays (energy_eV, normalized_absorption), same
    length, x assumed sortable. Returns a dict matching SpectralFeatures
    columns (minus record_id/computed_at_utc)."""
    order = np.argsort(x)
    x, y = x[order], y[order]

    dy = np.gradient(y, x)
    edge_idx = int(np.argmax(dy))
    edge_energy = float(x[edge_idx])
    max_derivative = float(dy[edge_idx])

    n = len(x)
    bottom10 = max(1, int(0.10 * n))
    top20 = max(1, int(0.20 * n))
    baseline_pre = float(np.mean(y[:bottom10]))
    plateau_post = float(np.mean(y[-top20:]))
    edge_jump = plateau_post - baseline_pre

    # White line: brightest point in [edge, edge + WHITE_LINE_WINDOW_EV],
    # clipped to available data. Always computed if there's any post-edge
    # data at all (there always is, since edge_energy is itself in-range).
    wl_mask = (x >= edge_energy) & (x <= edge_energy + WHITE_LINE_WINDOW_EV)
    if wl_mask.any():
        wl_idx_local = int(np.argmax(y[wl_mask]))
        wl_x = x[wl_mask][wl_idx_local]
        wl_y = y[wl_mask][wl_idx_local]
        white_line_energy = float(wl_x)
        white_line_intensity = float(wl_y)
    else:
        white_line_energy = white_line_intensity = None

    # Pre-edge: requires a genuine local peak, and enough room below the
    # edge to look for one at all.
    pre_edge_margin = edge_energy - float(x.min())
    pre_edge_energy = pre_edge_intensity = None
    if pre_edge_margin >= PRE_EDGE_MIN_MARGIN_EV:
        lo = edge_energy - min(PRE_EDGE_MAX_WINDOW_EV, pre_edge_margin)
        hi = edge_energy - PRE_EDGE_EXCLUSION_EV
        pe_mask = (x >= lo) & (x <= hi)
        if pe_mask.sum() >= 3:
            x_pe, y_pe = x[pe_mask], y[pe_mask]
            peaks = _local_maxima(y_pe)
            if peaks:
                best = max(peaks, key=lambda i: y_pe[i])
                pre_edge_energy = float(x_pe[best])
                pre_edge_intensity = float(y_pe[best])

    return {
        "edge_energy_ev": edge_energy,
        "edge_jump": edge_jump,
        "max_derivative": max_derivative,
        "white_line_energy_ev": white_line_energy,
        "white_line_intensity": white_line_intensity,
        "pre_edge_energy_ev": pre_edge_energy,
        "pre_edge_intensity": pre_edge_intensity,
        "pre_edge_margin_ev": pre_edge_margin,
    }


def main():
    parser = argparse.ArgumentParser(description="Engineer spectral features for SpectraHub records.")
    parser.add_argument("--db", default=None, help="Path to SQLite file (default: data/spectrahub.db)")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    db_path = Path(args.db) if args.db else None
    session = get_session(db_path)

    query = session.query(Spectrum)
    if args.limit:
        query = query.limit(args.limit)
    records = query.all()
    print(f"{len(records)} spectra to process.")
    if not records:
        return

    now = datetime.now(timezone.utc).isoformat()
    got_white_line = got_pre_edge = 0
    by_modality = {}

    for r in records:
        x = np.array(r.x_values(), dtype=float)
        y = np.array(r.y_values(), dtype=float)
        if len(x) < 3:
            continue
        feats = compute_features(x, y)

        if feats["white_line_intensity"] is not None:
            got_white_line += 1
        if feats["pre_edge_intensity"] is not None:
            got_pre_edge += 1
        by_modality.setdefault(r.modality, [0, 0])
        by_modality[r.modality][0] += 1
        if feats["pre_edge_intensity"] is not None:
            by_modality[r.modality][1] += 1

        existing = session.get(SpectralFeatures, r.record_id)
        if existing is None:
            session.add(SpectralFeatures(record_id=r.record_id, computed_at_utc=now, **feats))
        else:
            for key, value in feats.items():
                setattr(existing, key, value)
            existing.computed_at_utc = now

    session.commit()

    total = len(records)
    print()
    print(f"Done. {total} spectra processed.")
    print(f"  white_line populated: {got_white_line}/{total} ({100*got_white_line/total:.1f}%)")
    print(f"  pre_edge populated:   {got_pre_edge}/{total} ({100*got_pre_edge/total:.1f}%)")
    print("  pre_edge coverage by modality (populated/total):")
    for modality, (tot, pre) in sorted(by_modality.items()):
        print(f"    {modality}: {pre}/{tot} ({100*pre/tot:.1f}%)")


if __name__ == "__main__":
    main()
