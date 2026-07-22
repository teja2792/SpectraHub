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
    multiple-scattering theory), NOT experimentally measured. See README.md
    for the underlying theory. Every record is tagged source_type =
    "computed-database" so nothing downstream can mistake it for
    experimental data.

Requires:
    pip install mp-api pymatgen --break-system-packages
    export MP_API_KEY=your_key_here   (free key: https://next-gen.materialsproject.org/api)

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

# ---------------------------------------------------------------------------
# The four materials this portfolio already models elsewhere (MPExplorer,
# CatalystML, MieCatalystML). Always fetched first, always flagged
# is_highlighted=True. This is curation, not the scope limit -- the scope
# limit is controlled by --mode.
# ---------------------------------------------------------------------------
HIGHLIGHTED_MATERIALS = {
    "Cu2O": "Cu",
    "Fe2O3": "Fe",
    "TiO2": "Ti",
    "CeO2": "Ce",
}

# Tie-break hints only -- used to pick a specific polymorph when a formula
# query returns more than one match. NOT the sole source of truth (the
# script always queries live). mp-2657 (rutile TiO2) was not independently
# search-verified the way the other three were -- flagged here on purpose.
PREFERRED_MP_ID_HINTS = {
    "Cu2O": "mp-361",      # cuprite, cubic Pn-3m -- verified
    "Fe2O3": "mp-19770",   # hematite, trigonal R-3c -- verified
    "TiO2": "mp-2657",     # rutile -- UNVERIFIED, sanity-check before trusting
    "CeO2": "mp-20194",    # fluorite, cubic Fm-3m -- verified
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
                "'Automated generation and ensemble-learned matching of "
                "X-ray absorption spectra.' Sci. Data 5, 180151 (2018)."
            ),
            "license": "CC-BY-4.0",
            "digitization_error_estimate": None,
            "linked_properties": {"is_highlighted": self.is_highlighted},
            "retrieved_at_utc": self.retrieved_at_utc,
            "notes": self.notes,
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
    energy = list(getattr(spectrum, "x", spectrum[0] if isinstance(spectrum, (list, tuple)) else []))
    absorption = list(getattr(spectrum, "y", spectrum[1] if isinstance(spectrum, (list, tuple)) else []))
    return energy, absorption


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
            print(f"  [WARN] No XANES K-edge data found for {formula}.")
            continue
        hint = PREFERRED_MP_ID_HINTS.get(formula)
        chosen = next((r for r in results if str(r.material_id) == hint), None) if hint else None
        if chosen is None:
            chosen = results[0]
            if len(results) > 1:
                print(f"  [INFO] {len(results)} matches for {formula}; using mp-id={chosen.material_id} "
                      f"(no exact hint match).")
        energy, absorption = _spectrum_arrays(chosen.spectrum)
        if not energy or not absorption:
            print(f"  [WARN] {formula}: mp-id={chosen.material_id} had empty spectrum arrays.")
            continue
        records.append(XANESRecord(
            material_formula=formula, mp_id=str(chosen.material_id),
            absorbing_element=elem, edge="K",
            energy_ev=energy, normalized_absorption=absorption,
            is_highlighted=True,
        ))
        print(f"  Got mp-id={chosen.material_id}, {len(energy)} points")
    return records


def discover_all_metals() -> List[str]:
    """Every metal element pymatgen's periodic table knows about -- not a
    hand-typed list, so 'all metal oxides' actually means all of them, not
    whatever set of metals I happened to remember."""
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
        found_here = 0
        for r in results:
            if str(r.material_id) in already_fetched_ids:
                continue
            # Restrict to binary M-O compounds where that attribute is
            # available; if the API doesn't expose it on this object, don't
            # silently over-filter -- keep the result and let it through.
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
                mp_id=str(r.material_id),
                absorbing_element=metal, edge="K",
                energy_ev=energy, normalized_absorption=absorption,
                is_highlighted=False,
            ))
            already_fetched_ids.add(str(r.material_id))
            found_here += 1
        if found_here:
            print(f"  +{found_here} binary {metal}-O XANES record(s)")
    return records


def main():
    parser = argparse.ArgumentParser(description="Fetch FEFF-computed XANES for metal oxides from Materials Project.")
    parser.add_argument("--mode", choices=["highlighted", "all"], default="highlighted",
                         help="'highlighted' = just Cu2O/Fe2O3/TiO2/CeO2 (fast, default). "
                              "'all' = every metal oxide MP has XANES for (slow, many API calls).")
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