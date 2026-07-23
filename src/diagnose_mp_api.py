"""
diagnose_mp_api.py

One-off diagnostic, resolved: every mp_id in data/xanes/*.json is an
8-character all-lowercase AlphaID (e.g. "aaaaaanx"). This is real MP data
-- as of database version 2026-04-13, Materials Project switched new
materials/tasks from numeric MPIDs ("mp-149") to base-26 AlphaIDs
("mp-hilze") to stop conflating the material_id and task_id namespaces
(docs.materialsproject.org/data-production/identifiers). Confirmed live:
mp-361 resolves to real Cu2O on both the website and the summary API route.

Open question this section 4 answers: the canonical AlphaID format per
MP's docs includes an "mp-" prefix (AlphaID(mp-hilze)), but task_id from
the XAS route prints WITHOUT one (AlphaID(aaaaaanx)) -- and the docs say
MP is now deliberately distinguishing material_id from task_id, which
used to be identical. mp_xas_fetch.py has been storing task_id as mp_id.
Section 4 checks whether that value (with or without "mp-" prepended) or
a separate material_id field actually resolves on the material-level
routes (summary, oxidation_states) that src/label_fetch.py will depend on.

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
            material_id = getattr(r, "material_id", None)
            print(f"  material_id attr on XAS result: {material_id!r}")

            print()
            print("=== 4. Does task_id resolve as a material_id on the summary/oxidation_states routes? ===")
            # Per MP's own docs (docs.materialsproject.org/data-production/identifiers),
            # AlphaIDs and MPIDs should be cross-queryable. task_id from the XAS route
            # printed WITHOUT an "mp-" prefix above -- try it both ways.
            candidates_to_try = []
            raw_task_id = str(r.task_id)
            candidates_to_try.append(raw_task_id)
            if not raw_task_id.startswith("mp-"):
                candidates_to_try.append(f"mp-{raw_task_id}")
            if material_id:
                candidates_to_try.append(str(material_id))

            for candidate in dict.fromkeys(candidates_to_try):  # dedupe, keep order
                print(f"  Trying material_ids=[{candidate!r}] on materials.summary.search ...")
                try:
                    hit = mpr.materials.summary.search(
                        material_ids=[candidate], fields=["material_id", "formula_pretty"]
                    )
                    if hit:
                        print(f"    SUCCESS: material_id={hit[0].material_id!r} formula_pretty={hit[0].formula_pretty!r}")
                    else:
                        print("    no results (query ran, but returned empty)")
                except Exception as exc:
                    print(f"    FAILED: {type(exc).__name__}: {exc}")

                print(f"  Trying material_ids=[{candidate!r}] on materials.oxidation_states.search ...")
                try:
                    hit = mpr.materials.oxidation_states.search(material_ids=[candidate])
                    if hit:
                        print(f"    SUCCESS: average_oxidation_states={hit[0].average_oxidation_states!r}")
                    else:
                        print("    no results (query ran, but returned empty)")
                except Exception as exc:
                    print(f"    FAILED: {type(exc).__name__}: {exc}")
except Exception as exc:
    print(f"  [ERROR] {type(exc).__name__}: {exc}")
    print("  Paste this full error back too -- it's diagnostic either way.")
