"""
label_fetch.py

Fetches structural/electronic labels (oxidation state, coordination
number, mean bond length) for every unique mp_id already in the spectra
DB, and writes them to the `labels` table (see db.py: MaterialLabel).
These are the ML training targets for the supervised tasks -- clustering
and similarity search don't need this.

Verified before writing this script (see diagnose_mp_api.py output):
  - mp_id values stored in the spectra table are XAS-route `task_id`
    AlphaIDs (no "mp-" prefix, e.g. "aaaaaanx"). Confirmed live that these
    resolve correctly on both mpr.materials.summary.search() and
    mpr.materials.oxidation_states.search() -- no "mp-" prefix needed.
  - Every mp_id maps to exactly one absorbing_element across all its
    spectra rows (checked: 0/888 counterexamples), so a label row keyed
    only on mp_id is well-defined -- we don't need to disambiguate by
    element.

Data source and method, stated explicitly:
  - oxidation_state: mpr.materials.oxidation_states route's
    average_oxidation_states[absorbing_element]. This is MP's own
    precomputed value (Bond Valence Analysis, falling back to
    Composition.oxi_state_guesses when BVA fails -- see emmet-core's
    OxidationStateDoc.from_structure). It is an AVERAGE across all sites
    of that element, not a per-site value -- structures with multiple
    crystallographically distinct sites of the absorbing element (e.g.
    both tetrahedral and octahedral) get one blended number. That's a
    real methodological simplification, not a hidden bug.
  - coordination_number / mean_bond_length_angstrom: NOT provided by any
    MP route directly. Computed locally from the oxidation_states route's
    own `structure` field (already fetched, no extra API call) using
    pymatgen's CrystalNN nearest-neighbor algorithm, averaged across all
    sites of the absorbing element in that structure -- same averaging
    convention as oxidation_state, for consistency.

Coverage is expected to be well under 100%: BVA/oxi-state-guessing fails
for some structures, and CrystalNN fails or gives degenerate results for
some geometries. This script reports real coverage numbers at the end
rather than silently dropping failures.

One call per material -- search() with a single-item material_ids list
each time, not a multi-ID batched search() and not get_data_by_id
(deprecated by MP). See fetch_label_for_one()'s docstring for why
batching by ID list doesn't work here (query by task_id AlphaID, API can
return material_id in a completely different ID namespace for legacy
materials, so there's no reliable way to match a batched result back to
its input).

Usage:
    python src/label_fetch.py
    python src/label_fetch.py --limit 10          # sanity-check first
    python src/label_fetch.py --db data/spectrahub.db --progress-every 50
"""

import argparse
from datetime import datetime, timezone
from pathlib import Path

from db import get_session, Spectrum, MaterialLabel


def _get_client(api_key=None):
    from mp_xas_fetch import _get_client as _shared_get_client
    return _shared_get_client(api_key)


def unique_mp_ids_with_element(session):
    """Returns list of (mp_id, absorbing_element) for every unique mp_id
    in the spectra table. Already verified 1:1 mp_id -> absorbing_element,
    so picking any one row per mp_id is safe."""
    rows = session.query(Spectrum.mp_id, Spectrum.absorbing_element).distinct().all()
    seen = {}
    for mp_id, elem in rows:
        if mp_id and mp_id not in seen:
            seen[mp_id] = elem
    return list(seen.items())


def compute_coordination(structure, element_symbol):
    """Average coordination number and mean bond length (Angstrom) across
    every site of element_symbol in structure, using CrystalNN. Returns
    (cn, bond_length) or (None, None) if no sites match or CrystalNN fails
    on all of them."""
    from pymatgen.analysis.local_env import CrystalNN

    cnn = CrystalNN()
    cns, bond_lengths = [], []
    for idx, site in enumerate(structure):
        site_symbol = getattr(site.specie, "symbol", str(site.specie))
        if site_symbol != element_symbol:
            continue
        try:
            nn_info = cnn.get_nn_info(structure, idx)
        except Exception:
            continue
        if not nn_info:
            continue
        cns.append(len(nn_info))
        for neighbor in nn_info:
            bond_lengths.append(structure[idx].distance(neighbor["site"]))

    if not cns:
        return None, None
    avg_cn = sum(cns) / len(cns)
    avg_bond_length = sum(bond_lengths) / len(bond_lengths) if bond_lengths else None
    return avg_cn, avg_bond_length


def fetch_label_for_one(mpr, mp_id, element):
    """Fetch oxidation-state doc for a single mp_id via search() with a
    single-item material_ids list -- NOT get_data_by_id (deprecated by MP,
    "will be removed soon"), and NOT a multi-ID batched search() either.
    One ID in, one doc out, so there's no ambiguity matching a result back
    to its input regardless of what ID namespace the doc's own
    material_id happens to report (see module docstring: task_id AlphaID
    queried in, material_id can come back as a totally different legacy
    numeric MPID for older materials).

    Returns a label dict, or None if nothing usable came back."""
    try:
        results = mpr.materials.oxidation_states.search(
            material_ids=[mp_id],
            fields=["material_id", "average_oxidation_states", "possible_valences", "structure", "method"],
        )
    except Exception:
        return None
    if not results:
        return None
    doc = results[0]

    oxi_states = getattr(doc, "average_oxidation_states", {}) or {}
    oxidation_state = oxi_states.get(element)

    cn = bond_length = None
    structure = getattr(doc, "structure", None)
    if structure is not None:
        # The doc's structure does NOT reliably come back oxidation-decorated
        # over the API, even though emmet's OxidationStateDoc.from_structure
        # decorates it server-side (confirmed live: CrystalNN warns "No
        # oxidation states specified on sites!" without this). CrystalNN is
        # documented to be more reliable with oxidation-decorated sites (uses
        # ionic radii instead of generic covalent/atomic radii), so
        # re-decorate manually from the doc's own per-site possible_valences
        # before running it.
        valences = getattr(doc, "possible_valences", None)
        if valences and len(valences) == len(structure):
            try:
                structure = structure.copy()
                structure.add_oxidation_state_by_site(valences)
            except Exception:
                pass  # fall back to undecorated structure below

        try:
            cn, bond_length = compute_coordination(structure, element)
        except Exception as exc:
            print(f"    [WARN] {mp_id}: CrystalNN failed ({exc})")

    if oxidation_state is None and cn is None:
        return None  # nothing usable for this material

    return {
        "oxidation_state": oxidation_state,
        "coordination_number": round(cn) if cn is not None else None,
        "mean_bond_length_angstrom": bond_length,
        "label_source": f"MP oxidation_states route ({getattr(doc, 'method', 'unknown')}) + pymatgen CrystalNN",
    }


def main():
    parser = argparse.ArgumentParser(description="Fetch structural labels for SpectraHub materials.")
    parser.add_argument("--db", default=None, help="Path to SQLite file (default: data/spectrahub.db)")
    parser.add_argument("--progress-every", type=int, default=50,
                         help="Print a progress line every N materials (this is a single-ID-per-call "
                              "fetch, not a batched query -- see fetch_label_for_one docstring for why).")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--limit", type=int, default=None,
                         help="Only process the first N mp_ids -- use this to sanity-check "
                              "before running against the full set.")
    args = parser.parse_args()

    db_path = Path(args.db) if args.db else None
    session = get_session(db_path)

    targets = unique_mp_ids_with_element(session)
    if args.limit:
        targets = targets[:args.limit]
    print(f"{len(targets)} unique mp_ids to label.")
    if not targets:
        return

    mpr = _get_client(args.api_key)
    got_oxidation = got_coordination = got_nothing = 0
    now = datetime.now(timezone.utc).isoformat()

    with mpr:
        for i, (mp_id, element) in enumerate(targets, 1):
            data = fetch_label_for_one(mpr, mp_id, element)

            if data is None:
                got_nothing += 1
            else:
                if data["oxidation_state"] is not None:
                    got_oxidation += 1
                if data["coordination_number"] is not None:
                    got_coordination += 1

                existing = session.get(MaterialLabel, mp_id)
                if existing is None:
                    session.add(MaterialLabel(mp_id=mp_id, fetched_at_utc=now, **data))
                else:
                    for key, value in data.items():
                        setattr(existing, key, value)
                    existing.fetched_at_utc = now

            if i % args.progress_every == 0 or i == len(targets):
                session.commit()  # commit incrementally so a late failure doesn't lose earlier work
                print(f"[{i}/{len(targets)}] oxidation_state={got_oxidation} "
                      f"coordination_number={got_coordination} nothing_usable={got_nothing}")

    total = len(targets)
    print()
    print(f"Done. {total} mp_ids attempted.")
    print(f"  oxidation_state populated:      {got_oxidation}/{total} ({100*got_oxidation/total:.1f}%)")
    print(f"  coordination_number populated:  {got_coordination}/{total} ({100*got_coordination/total:.1f}%)")
    print(f"  nothing usable at all:          {got_nothing}/{total} ({100*got_nothing/total:.1f}%)")
    print("These are the real coverage numbers -- treat anything claiming higher as suspect.")


if __name__ == "__main__":
    main()
