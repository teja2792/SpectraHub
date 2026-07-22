"""
mp_xas_fetch.py  (renamed from mp_xanes_fetch.py -- now fetches all three
X-ray absorption spectrum types Materials Project exposes: XANES, XAFS,
and EXAFS, not just XANES.)

Pulls FEFF-computed K-edge X-ray absorption spectra from the Materials
Project.

SCOPE: ALL metal oxides (any metal element + oxygen) that Materials Project
has FEFF K-edge XAS data for, across all three spectrum_type values
(XANES / XAFS / EXAFS). The four materials most relevant to this
portfolio (Cu2O, Fe2O3, TiO2, CeO2) are always fetched first and flagged
is_highlighted=True.

IMPORTANT -- what this data is and isn't:
    Ab initio COMPUTED (FEFF real-space Green's-function multiple-
    scattering theory), NOT experimentally measured. Every record is
    tagged source_type = "computed-database".

VERIFIED against live MP_API_KEY runs (2026-07-22):
    - Identifier field is `task_id`, not `material_id`.
    - `r.spectrum` exposes `.x` / `.y` list attributes directly.
    - One formula/element/edge query returns MULTIPLE spectrum_type
      entries (XANES/XAFS/EXAFS) for the SAME material (same task_id).
      This version fetches all three deliberately, instead of filtering
      down to just XANES.
    - CeO2's XAS onset is ~50 eV above the tabulated experimental Ce
      K-edge, larger than FEFF's documented normal Fermi-level
      referencing error ("only a few eV"). Cause unconfirmed -- see
      KNOWN_CAVEATS below. No correction shift applied.

Requires:
    pip install mp-api pymatgen
    $env:MP_API_KEY="your_key_here"

Usage:
    python src/mp_xas_fetch.py                  # highlighted 4 only (fast, default)
    python src/mp_xas_fetch.py --mode all        # every metal oxide MP has XAS for (slow)
"""

import argparse
import csv
import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List

HIGHLIGHTED_MATERIALS = {
    "Cu2O": "Cu",
    "Fe2O3": "Fe",
    "TiO2": "Ti",
    "CeO2": "Ce",
}

PREFERRED_MP_ID_HINTS = {
    "Cu2O": "mp-361",
    "Fe2O3": "mp-19770",
    "TiO2": "mp-2657",     # unverified tie-break hint
    "CeO2": "mp-20194",
}

DESIRED_SPECTRUM_TYPES = ["XANES", "XAFS", "EXAFS"]

KNOWN_CAVEATS = {
    "CeO2": (
        "XAS onset sits ~50 eV above the tabulated experimental Ce K-edge "
        "(40443 eV) -- larger than FEFF's own documented typical "
        "Fermi-level referencing error ('only a few eV', FEFF9 user's "
        "guide, feffproject.org). Cause unconfirmed as of 2026-07-22. "
        "Consistent across independent XANES/XAFS/EXAFS computations for "
        "the same mp-id -- treated as a genuine database-entry feature, "
        "not a fetch error, but no verified mechanism established. No "
        "energy-shift correction applied."
    ),
}


@dataclass
class XASRecord:
    material_formula: str
    mp_id: str
    absorbing_element: str
    edge: str
    spectrum_type: str  # "XANES" | "XAFS" | "EXAFS"
    energy_ev: list
    normalized_absorption: list
    is_highlighted: bool = False
    retrieved_at_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    notes: str = "Ab initio computed spectrum, not an experimental measurement."

    def to_schema_record(self) -> dict:
        notes = self.notes
        caveat = KNOWN_CAVEATS.get(self.material_formula)
        if caveat:
            notes = f"{notes} CAVEAT: {caveat}"
        return {
            "record_id": f"{self.material_formula}_{self.absorbing_element}_{self.edge}edge_{self.spectrum_type}_{self.mp_id}",
            "material_formula": self.material_formula,
            "modality": self.spectrum_type,  # "XANES" / "XAFS" / "EXAFS" -- see schema.py Modality enum
            "edge": self.edge,
            "absorbing_element": self.absorbing_element,
            "mp_id": self.mp_id,
            "x_axis": "energy_eV",
            "y_axis": "normalized_absorption",
            "x_values": self.energy_ev,
            "y_values": self.normalized_absorption,
            "source_type": "computed-database",
            "citation": (
                "Materials Project FEFF XAS database; Zheng, C. et al. "
                "Sci. Data 5, 180151 (2018)."
            ),
            "license": "CC-BY-4.0",
            "digitization_error_estimate": None,
            "linked_properties": {"is_highlighted": self.is_highlighted},
            "retrieved_at_utc": self.retrieved_at_utc,
            "notes": notes,
        }


def _get_client(api_key: Optional[str] = None):
    from mp_api.client import MPRester
    key = api_key or os.environ.get("MP_API_KEY")
    if not key:
        raise RuntimeError(
            "No Materials Project API key found. Set MP_API_KEY, or pass "
            "--api-key. Get a free key at https://next-gen.materialsproject.org/api"
        )
    return MPRester(key)


def _spectrum_arrays(spectrum):
    if isinstance(spectrum, dict):
        return list(spectrum.get("x", [])), list(spectrum.get("y", []))
    energy = list(getattr(spectrum, "x", spectrum[0] if isinstance(spectrum, (list, tuple)) else []))
    absorption = list(getattr(spectrum, "y", spectrum[1] if isinstance(spectrum, (list, tuple)) else []))
    return energy, absorption


def fetch_highlighted(mpr) -> List[XASRecord]:
    from emmet.core.xas import Edge
    records = []
    for formula, elem in HIGHLIGHTED_MATERIALS.items():
        print(f"Fetching highlighted material {formula} ({elem} K-edge)...")
        try:
            results = mpr.materials.xas.search(formula=formula, absorbing_element=elem, edge=Edge.K)
        except Exception as exc:
            print(f"  [ERROR] {exc}")
            continue
        if not results:
            print(f"  [WARN] No XAS data found for {formula}.")
            continue
        hint = PREFERRED_MP_ID_HINTS.get(formula)
        for spectrum_type in DESIRED_SPECTRUM_TYPES:
            candidates = [r for r in results if getattr(r, "spectrum_type", None) == spectrum_type]
            if not candidates:
                print(f"  [WARN] {formula}: no {spectrum_type} entry found.")
                continue
            chosen = next((r for r in candidates if str(r.task_id) == hint), None) if hint else None
            if chosen is None:
                chosen = candidates[0]
                if len(candidates) > 1:
                    print(f"  [INFO] {len(candidates)} {spectrum_type} matches for {formula}; "
                          f"using mp-id={chosen.task_id}")
            energy, absorption = _spectrum_arrays(chosen.spectrum)
            if not energy or not absorption:
                print(f"  [WARN] {formula} {spectrum_type}: mp-id={chosen.task_id} had empty spectrum arrays.")
                continue
            records.append(XASRecord(
                material_formula=formula, mp_id=str(chosen.task_id),
                absorbing_element=elem, edge="K", spectrum_type=spectrum_type,
                energy_ev=energy, normalized_absorption=absorption,
                is_highlighted=True,
            ))
            print(f"  Got {spectrum_type} mp-id={chosen.task_id}, {len(energy)} points, "
                  f"energy range [{min(energy):.1f}, {max(energy):.1f}] eV")
    return records


def discover_all_metals() -> List[str]:
    from pymatgen.core.periodic_table import Element
    return sorted({el.symbol for el in Element if el.is_metal})


def fetch_all_metal_oxides(mpr, already_fetched: set) -> List[XASRecord]:
    """already_fetched holds (task_id, spectrum_type) tuples -- a bare
    task_id isn't enough now, since one material legitimately has up to 3
    kept records (one per spectrum type)."""
    from emmet.core.xas import Edge
    metals = discover_all_metals()
    print(f"\n--mode all: scanning {len(metals)} metal elements for M-O XAS data "
          f"(one API call per metal, up to 3 records each -- slow, may hit rate limits).")
    records = []
    for i, metal in enumerate(metals, 1):
        print(f"[{i}/{len(metals)}] {metal}-O ...")
        try:
            results = mpr.materials.xas.search(elements=[metal, "O"], absorbing_element=metal, edge=Edge.K)
        except Exception as exc:
            print(f"  [ERROR] {exc}")
            continue
        if not results:
            continue
        found_here = 0
        for r in results:
            spectrum_type = getattr(r, "spectrum_type", None)
            if spectrum_type not in DESIRED_SPECTRUM_TYPES:
                continue
            # mp-api/emmet hands back an enum object here, not a plain str
            # (unlike fetch_highlighted(), which reuses a local string loop
            # variable). Coerce to plain str now -- storing the raw enum
            # caused validate_record() to crash later (enum.__eq__ raises
            # ValueError when compared against "" instead of returning False).
            spectrum_type = spectrum_type.value if hasattr(spectrum_type, "value") else str(spectrum_type)
            key = (str(r.task_id), spectrum_type)
            if key in already_fetched:
                continue
            els = getattr(r, "elements", None)
            if els is not None:
                symbols = {str(e) for e in els}
                if symbols != {metal, "O"}:
                    continue
            energy, absorption = _spectrum_arrays(r.spectrum)
            if not energy or not absorption:
                continue
            records.append(XASRecord(
                material_formula=str(getattr(r, "formula_pretty", f"{metal}xOy")),
                mp_id=str(r.task_id),
                absorbing_element=metal, edge="K", spectrum_type=spectrum_type,
                energy_ev=energy, normalized_absorption=absorption,
                is_highlighted=False,
            ))
            already_fetched.add(key)
            found_here += 1
        if found_here:
            print(f"  +{found_here} binary {metal}-O XAS record(s)")
    return records


def write_summary_csv(schema_records: List[dict], out_path: Path):
    fields = [
        "record_id", "material_formula", "modality", "edge",
        "absorbing_element", "mp_id", "n_points", "energy_min_eV",
        "energy_max_eV", "is_highlighted", "source_type", "license",
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in schema_records:
            x = r.get("x_values", [])
            writer.writerow({
                "record_id": r["record_id"],
                "material_formula": r["material_formula"],
                "modality": r["modality"],
                "edge": r.get("edge"),
                "absorbing_element": r.get("absorbing_element"),
                "mp_id": r.get("mp_id"),
                "n_points": len(x),
                "energy_min_eV": min(x) if x else None,
                "energy_max_eV": max(x) if x else None,
                "is_highlighted": r.get("linked_properties", {}).get("is_highlighted", False),
                "source_type": r.get("source_type"),
                "license": r.get("license"),
            })
    print(f"Wrote {out_path} ({len(schema_records)} rows)")


def plot_highlighted(records: List[XASRecord], out_path: Path):
    import matplotlib
    matplotlib.use("Agg")  # headless-safe; no GUI backend needed
    import matplotlib.pyplot as plt

    highlighted = [r for r in records if r.is_highlighted and r.spectrum_type == "XANES"]
    if not highlighted:
        print("No highlighted XANES records found -- skipping summary plot.")
        return
    highlighted = sorted(highlighted, key=lambda r: r.material_formula)
    n = len(highlighted)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 3.5))
    if n == 1:
        axes = [axes]
    for ax, r in zip(axes, highlighted):
        ax.plot(r.energy_ev, r.normalized_absorption)
        ax.set_title(f"{r.material_formula} ({r.absorbing_element} K-edge)")
        ax.set_xlabel("Energy (eV)")
        ax.set_ylabel("Normalized absorption")
    fig.suptitle("SpectraHub -- Highlighted materials, XANES (FEFF-computed, Materials Project)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Fetch FEFF-computed XAS (XANES/XAFS/EXAFS) for metal oxides.")
    parser.add_argument("--mode", choices=["highlighted", "all"], default="highlighted")
    parser.add_argument("--out", default="data/xanes")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--api-key", default=None)
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    mpr = _get_client(args.api_key)

    with mpr:
        all_records = fetch_highlighted(mpr)
        if args.mode == "all":
            already = {(r.mp_id, r.spectrum_type) for r in all_records}
            all_records += fetch_all_metal_oxides(mpr, already)

    from schema import validate_record
    manifest = []
    schema_records = []
    for rec in all_records:
        schema_rec = rec.to_schema_record()
        issues = validate_record(schema_rec)
        if issues:
            print(f"  [SCHEMA WARNING] {schema_rec['record_id']}: {issues}")
        out_path = out_dir / f"{schema_rec['record_id']}.json"
        with open(out_path, "w") as f:
            json.dump(schema_rec, f, indent=2)
        schema_records.append(schema_rec)
        manifest.append({
            "record_id": schema_rec["record_id"],
            "material_formula": rec.material_formula,
            "spectrum_type": rec.spectrum_type,
            "mp_id": rec.mp_id,
            "is_highlighted": rec.is_highlighted,
        })

    with open(out_dir / "MANIFEST.json", "w") as f:
        json.dump(manifest, f, indent=2)

    write_summary_csv(schema_records, out_dir / "summary.csv")
    plot_highlighted(all_records, Path(args.results_dir) / "xas_summary.png")

    n_highlighted = sum(1 for r in all_records if r.is_highlighted)
    print(f"\nDone. {len(all_records)} XAS records saved to {out_dir}/ "
          f"({n_highlighted} highlighted, {len(all_records) - n_highlighted} other metal oxides).")


if __name__ == "__main__":
    main()