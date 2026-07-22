"""
mp_xanes_fetch.py

Pulls FEFF-computed K-edge XANES spectra from the Materials Project.

SCOPE: ALL metal oxides (any metal element + oxygen) that Materials Project
has FEFF K-edge XANES data for -- not restricted to a fixed list of
materials. The four materials most relevant to this portfolio (Cu2O,
Fe2O3, TiO2, CeO2) are always fetched first via a precise formula-based
query and flagged is_highlighted=True in the output. Everything else
(--mode all) comes from a broad discovery scan over every metal element
pymatgen knows about, restricted to binary M-O compounds.

IMPORTANT -- what this data is and isn't:
    These spectra are AB INITIO COMPUTED (FEFF real-space Green's-function
    multiple-scattering theory), NOT experimentally measured. Every record
    is tagged source_type = "computed-database".

VERIFIED so far against a live MP_API_KEY run (2026-07-22):
    - Identifier field on returned mp-api objects is `task_id`, not
      `material_id`.
    - `r.spectrum` exposes `.x` / `.y` list attributes directly (confirmed
      live -- not a dict).
    - A single formula/element/edge query can return MULTIPLE spectrum_type
      values (XANES / XAFS / EXAFS) for the SAME material (same task_id).
      This script now explicitly filters to spectrum_type == "XANES" --
      earlier versions relied on results[0] happening to be the narrow
      entry, which worked by luck for all four V1 materials but was not a
      real filter.
    - CeO2's computed K-edge onset is ~50 eV above the tabulated
      experimental Ce K-edge (see KNOWN_CAVEATS below) -- this is
      reproducible across 3 independent FEFF computations for that
      material, but the cause has NOT been confirmed against FEFF's own
      documentation (which describes normal Fermi-level referencing error
      as "only a few eV," not ~50). Treat as an open question, not a
      solved one -- do not apply an unverified correction shift.

Requires:
    pip install mp-api pymatgen
    $env:MP_API_KEY="your_key_here"   (free key: https://next-gen.materialsproject.org/api)

Usage:
    python src/mp_xanes_fetch.py                  # highlighted 4 only (fast, default)
    python src/mp_xanes_fetch.py --mode all        # every metal oxide MP has XANES for (slow)
"""

import argparse
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

# Data-quality caveats that get folded directly into a record's `notes`
# field -- travels with the data itself, not just documented in README.
KNOWN_CAVEATS = {
    "CeO2": (
        "Computed K-edge onset sits ~50 eV above the tabulated experimental "
        "Ce K-edge (40443 eV) -- larger than FEFF's own documented typical "
        "Fermi-level referencing error ('only a few eV', per the FEFF9 "
        "user's guide, feffproject.org). Cause unconfirmed as of 2026-07-22. "
        "The offset is consistent across 3 independent FEFF computations "
        "(XANES/XAFS/EXAFS spectrum_type) for the same mp-id, so it is "
        "treated as a genuine feature of this database entry, not a "
        "fetch/parsing error -- but no verified mechanistic explanation has "
        "been established. Do not apply an ad hoc energy-shift correction "
        "without independent justification."
    ),
}


@dataclass
class XANESRecord:
    material_formula: str
    mp_id: str
    absorbing_element: str
    edge: str
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
            "record_id": f"{self.material_formula}_{self.absorbing_element}_{self.edge}edge_{self.mp_id}",
            "material_formula": self.material_formula,
            "modality": "XANES",
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


def _filter_xanes(results, formula):
    """Explicit spectrum_type filter -- a single query can return XANES,
    XAFS, and EXAFS entries for the same material. Earlier versions of this
    script took results[0] and got the right one by luck; this is the real
    fix."""
    xanes = [r for r in results if getattr(r, "spectrum_type", None) == "XANES"]
    if not xanes:
        found_types = sorted({str(getattr(r, "spectrum_type", None)) for r in results})
        print(f"  [WARN] {formula}: {len(results)} XAS entries found but none are "
              f"spectrum_type=='XANES' (found: {found_types}). Skipping.")
    return xanes


def fetch_highlighted(mpr) -> List[XANESRecord]:
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
        results = _filter_xanes(results, formula)
        if not results:
            continue
        hint = PREFERRED_MP_ID_HINTS.get(formula)
        chosen = next((r for r in results if str(r.task_id) == hint), None) if hint else None
        if chosen is None:
            chosen = results[0]
            if len(results) > 1:
                print(f"  [INFO] {len(results)} XANES matches for {formula}; using mp-id={chosen.task_id} "
                      f"(no exact hint match).")
        energy, absorption = _spectrum_arrays(chosen.spectrum)
        if not energy or not absorption:
            print(f"  [WARN] {formula}: mp-id={chosen.task_id} had empty spectrum arrays.")
            continue
        records.append(XANESRecord(
            material_formula=formula, mp_id=str(chosen.task_id),
            absorbing_element=elem, edge="K",
            energy_ev=energy, normalized_absorption=absorption,
            is_highlighted=True,
        ))
        print(f"  Got mp-id={chosen.task_id}, {len(energy)} points, "
              f"energy range [{min(energy):.1f}, {max(energy):.1f}] eV")
    return records


def discover_all_metals() -> List[str]:
    from pymatgen.core.periodic_table import Element
    return sorted({el.symbol for el in Element if el.is_metal})


def fetch_all_metal_oxides(mpr, already_fetched_ids: set) -> List[XANESRecord]:
    from emmet.core.xas import Edge
    metals = discover_all_metals()
    print(f"\n--mode all: scanning {len(metals)} metal elements for M-O XANES data "
          f"(one API call per metal -- this is slow and may hit rate limits).")
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
        results = _filter_xanes(results, f"{metal}-O")
        found_here = 0
        for r in results:
            if str(r.task_id) in already_fetched_ids:
                continue
            els = getattr(r, "elements", None)
            if els is not None:
                symbols = {str(e) for e in els}
                if symbols != {metal, "O"}:
                    continue
            energy, absorption = _spectrum_arrays(r.spectrum)
            if not energy or not absorption:
                continue
            records.append(XANESRecord(
                material_formula=str(getattr(r, "formula_pretty", f"{metal}xOy")),
                mp_id=str(r.task_id),
                absorbing_element=metal, edge="K",
                energy_ev=energy, normalized_absorption=absorption,
                is_highlighted=False,
            ))
            already_fetched_ids.add(str(r.task_id))
            found_here += 1
        if found_here:
            print(f"  +{found_here} binary {metal}-O XANES record(s)")
    return records


def main():
    parser = argparse.ArgumentParser(description="Fetch FEFF-computed XANES for metal oxides from Materials Project.")
    parser.add_argument("--mode", choices=["highlighted", "all"], default="highlighted")
    parser.add_argument("--out", default="data/xanes")
    parser.add_argument("--api-key", default=None)
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    mpr = _get_client(args.api_key)

    with mpr:
        all_records = fetch_highlighted(mpr)
        if args.mode == "all":
            already_ids = {r.mp_id for r in all_records}
            all_records += fetch_all_metal_oxides(mpr, already_ids)

    from schema import validate_record
    manifest = []
    for rec in all_records:
        schema_rec = rec.to_schema_record()
        issues = validate_record(schema_rec)
        if issues:
            print(f"  [SCHEMA WARNING] {schema_rec['record_id']}: {issues}")
        out_path = out_dir / f"{schema_rec['record_id']}.json"
        with open(out_path, "w") as f:
            json.dump(schema_rec, f, indent=2)
        manifest.append({
            "record_id": schema_rec["record_id"],
            "material_formula": rec.material_formula,
            "mp_id": rec.mp_id,
            "is_highlighted": rec.is_highlighted,
        })

    with open(out_dir / "MANIFEST.json", "w") as f:
        json.dump(manifest, f, indent=2)

    n_highlighted = sum(1 for r in all_records if r.is_highlighted)
    print(f"\nDone. {len(all_records)} XANES records saved to {out_dir}/ "
          f"({n_highlighted} highlighted, {len(all_records) - n_highlighted} other metal oxides).")


if __name__ == "__main__":
    main()