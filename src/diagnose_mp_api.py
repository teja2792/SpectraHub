"""
diagnose_mp_api.py

One-off diagnostic. Every mp_id currently in data/xanes/*.json is an
8-character all-lowercase string (e.g. "aaaaaanx"). Real Materials Project
IDs are always "mp-<integer>" -- verified live on next-gen.materialsproject.org
that mp-361 is genuinely Cu2O. That means either the fetch that produced
data/xanes/ wasn't really hitting the live MP API, or something between
the API response and the JSON files on disk is mangling task_id.

This script prints exactly what the live API hands back, before any of
mp_xas_fetch.py's own code touches it, so we can see where -- if anywhere
-- the ID format diverges from "mp-<integer>".

Usage:
    python src/diagnose_mp_api.py

Run this from the repo's src/ directory (same as mp_xas_fetch.py), with
MP_API_KEY set the same way you already have it for mp_xas_fetch.py.
"""

import sys
import importlib.util

print("=== 1. Which package is actually being imported ===")
for name in ("mp_api", "emmet", "pymatgen"):
    spec = importlib.util.find_spec(name)
    if spec is None:
        print(f"  {name}: NOT INSTALLED")
    else:
        print(f"  {name}: {spec.origin}")

print()
print("=== 2. sys.path (first 10 entries, in search order) ===")
for p in sys.path[:10]:
    print(f"  {p}")

print()
print("=== 3. Live API check ===")
try:
    from mp_xas_fetch import _get_client
    from emmet.core.xas import Edge

    mpr = _get_client()
    with mpr:
        print("mpr.materials.summary.search(material_ids=['mp-361']) ...")
        summary = mpr.materials.summary.search(
            material_ids=["mp-361"], fields=["material_id", "formula_pretty"]
        )
        if summary:
            mid = summary[0].material_id
            formula = summary[0].formula_pretty
            print(f"  material_id={mid!r}  formula_pretty={formula!r}")
            print(f"  expected:   material_id='mp-361'  formula_pretty='Cu2O'")
            print(f"  MATCH: {str(mid) == 'mp-361' and formula == 'Cu2O'}")
        else:
            print("  [WARN] No result for mp-361 via summary route.")

        print()
        print("mpr.materials.xas.search(formula='Cu2O', absorbing_element='Cu', edge=Edge.K) ...")
        results = mpr.materials.xas.search(formula="Cu2O", absorbing_element="Cu", edge=Edge.K)
        if not results:
            print("  [WARN] No XAS results for Cu2O.")
        else:
            r = results[0]
            print(f"  task_id: {r.task_id!r}  (type: {type(r.task_id).__name__})")
            print(f"  spectrum_type: {r.spectrum_type!r}  (type: {type(r.spectrum_type).__name__})")
            print()
            if str(r.task_id).startswith("mp-"):
                print("  ==> Real Materials Project ID. Problem is DOWNSTREAM of the API call --")
                print("      i.e. data/xanes/*.json was not produced by a clean run of the current script.")
            else:
                print("  ==> NOT a real Materials Project ID format. Problem is UPSTREAM --")
                print("      in the installed package or account, not in mp_xas_fetch.py's logic.")
except Exception as exc:
    print(f"  [ERROR] {type(exc).__name__}: {exc}")
    print("  Paste this full error back too -- it's diagnostic either way.")
