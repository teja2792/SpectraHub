# SpectraHub

A small end-to-end pipeline that pulls computed X-ray absorption spectra (XAS) for metal oxides from the [Materials Project](https://materialsproject.org), turns them into a queryable database, engineers features from the raw spectra, and uses those features to cluster materials by spectral "fingerprint" and predict oxidation state, coordination number, and bond length directly from the shape of a spectrum.

Everything below reflects the actual, currently-verified state of the data as of this write-up: **2004 spectra**, **888 unique materials**, **737 with structural labels**.

## What this is (and isn't)

Every spectrum here is **ab initio computed** (FEFF real-space Green's-function multiple-scattering theory), not experimentally measured. Materials Project tags every record `source_type = "computed-database"`, and this project preserves that tag end to end — nothing here should be read as lab-measured data.

Four materials are hand-picked and flagged `is_highlighted = True` because they're the ones this project cares most about: **Cu2O, Fe2O3, TiO2, CeO2**. Everything else (884 more materials) comes from an open-ended scan of every metal-oxide combination Materials Project has FEFF K-edge data for.

**Known data caveat:** CeO2's computed XAS onset sits ~50 eV above the tabulated experimental Ce K-edge — larger than FEFF's documented typical referencing error. This is consistent across independent XANES/XAFS/EXAFS calculations for the same material, so it's treated as a genuine feature of that database entry, not a fetch bug — but no verified mechanism has been established, and no correction is applied.

## The pipeline

Six stages, each a standalone script, each writing to one shared SQLite database (`data/spectrahub.db`):

| # | Script | What it does | Needs MP_API_KEY? |
|---|---|---|---|
| 1 | `src/mp_xas_fetch.py` | Fetches XANES/XAFS/EXAFS spectra from Materials Project, writes JSON + CSV + summary plot | Yes |
| 2 | `src/ingest.py` | Loads the fetched JSON records into `spectrahub.db` | No |
| 3 | `src/label_fetch.py` | Fetches oxidation state, coordination number, and bond length per material from Materials Project's structure data | Yes |
| 4 | `src/feature_engineering.py` | Computes edge energy, edge jump, white-line, and pre-edge features from each raw spectrum | No |
| 5 | `src/clustering_similarity.py` | Clusters XANES spectra by shape and powers "find a similar fingerprint" search | No |
| 6 | `src/ml_models.py` | Predicts oxidation state / coordination number / bond length from spectral features, with honest leave-one-out evaluation | No |
| 7 | `src/api.py` | Read-only FastAPI layer serving all of the above | No |

Only steps 1 and 3 talk to Materials Project's live API — everything else runs offline against the local database.

## Quick start

```bash
pip install -r requirements.txt
$env:MP_API_KEY="your_key_here"          # PowerShell; use export on macOS/Linux

cd src
python mp_xas_fetch.py --mode all         # ~25-30 min: fetches all 888 materials
python ingest.py                          # seconds: loads JSON into spectrahub.db
python label_fetch.py                     # several minutes: one MP API call per material
python feature_engineering.py             # seconds: pure numpy, no network
python clustering_similarity.py --cluster --k 12
python ml_models.py                       # seconds: trains + evaluates
uvicorn api:app --reload                  # starts the API at http://127.0.0.1:8000/docs
```

`python mp_xas_fetch.py` with no `--mode` flag fetches just the 4 highlighted materials in seconds, useful for a quick smoke test before committing to the full ~30-minute run.

## Real results, not projections

**Coverage** (how much of the data actually has usable values — not padded):

- Structural labels: 737/888 materials (83%) got a coordination number, 651/888 (73%) got an oxidation state.
- Spectral features: white-line features computed for 100% of spectra (2004/2004); pre-edge features only for 466/2004 (23%) — because many fetched XANES windows start too close to the edge to have pre-edge data at all, not because of a bug (documented and measured directly in `feature_engineering.py`).

**Does spectral shape encode chemistry?** Tested with leave-one-out cross-validation against a naive baseline, on 539 XANES materials:

| Target | Model accuracy / error | Naive baseline | Verdict |
|---|---|---|---|
| Oxidation state (classification) | 59.2% | 36.2% (majority class) | Real signal — matches the known "chemical shift" effect in XAS |
| Coordination number (classification) | 53.4% | 50.3% (majority class) | Weak — barely beats guessing |
| Bond length (regression) | MAE 0.123 Å | MAE 0.204 Å (mean baseline) | Real signal — ~40% error reduction |

Coordination number being the weak link makes physical sense: it's a geometric property that near-edge XANES shape encodes less directly than electronic oxidation state does.

**Does the unsupervised clustering find real chemistry, with zero labels involved in the clustering itself?** Cluster 2 (20 materials, mostly Al2O3, found purely from spectral shape) has an oxidation-state standard deviation of **0.0** — every single member shares the same oxidation state — against a dataset-wide baseline of 1.16. Not every cluster is this clean, and that's stated honestly in the script's own output, not smoothed over.

**Chemistry sanity checks that passed:** pre-edge intensity across the four highlighted materials follows exactly the ordering XAS theory predicts by d-electron count — none for Cu2O (d¹⁰, no empty d-states), weak for Fe2O3 (d⁵, dipole-forbidden in its octahedral site), strong for TiO2 (d⁰, the textbook case for a resolved pre-edge). Cu2O's computed bond length (1.839 Å) and coordination number (2) match the Materials Project website's own description of mp-361 exactly.

## API

```bash
cd src && uvicorn api:app --reload
```

Then visit `http://127.0.0.1:8000/docs` for interactive documentation, or:

```
GET /stats                                          overview counts
GET /materials?formula_contains=Cu2O                 search materials
GET /materials/{mp_id}                               full detail for one material
GET /spectra/{record_id}                              raw spectrum + computed features
GET /spectra/{record_id}/similar?top=10                nearest XANES fingerprints
GET /clusters/{cluster_id}                            all members of a cluster
GET /predictions?task=oxidation_state&mismatches_only=true   audit model errors directly
```

The API is read-only by design — it serves what the pipeline scripts already computed, and never re-runs a model or triggers a live Materials Project fetch on your behalf.

## Project structure

```
src/
  mp_xas_fetch.py          # stage 1: fetch spectra from Materials Project
  ingest.py                 # stage 2: load JSON into SQLite
  label_fetch.py             # stage 3: fetch structural labels
  feature_engineering.py      # stage 4: derive spectral features
  clustering_similarity.py     # stage 5: cluster + similarity search
  ml_models.py                  # stage 6: supervised prediction models
  api.py                          # stage 7: FastAPI layer
  db.py                             # shared SQLAlchemy schema
  schema.py                          # JSON record schema + validator
  diagnose_mp_api.py                  # one-off MP API diagnostic (kept for reference)
data/
  xanes/                    # per-record JSON + MANIFEST.json + summary.csv
  spectrahub.db              # SQLite database (all 7 tables)
results/
  xas_summary.png            # overlay plot of the 4 highlighted materials
```

## Design choices worth knowing about

- **No scikit-learn or scipy.** Both were unreliable to install in this project's development environment, so clustering (k-means), similarity search (cosine similarity), and prediction (k-nearest-neighbors) are all hand-implemented in pure numpy — verified directly against real data rather than assumed correct. If scikit-learn installs cleanly for you, swapping in a Random Forest for `ml_models.py` would likely improve on the k-NN results above.
- **SQLite, not Postgres.** Fine at this scale (~2000 records); the schema uses plain SQLAlchemy ORM with no SQLite-specific types, so switching the engine URL later doesn't require a rewrite.
- **Materials Project's identifier scheme changed recently** (database version `2026-04-13`): IDs are transitioning from numeric (`mp-149`) to base-26 `AlphaID`s (`mp-hilze`). This project's `mp_id` values are the AlphaID `task_id` from the XAS API route — confirmed live to resolve correctly against Materials Project's other API routes, with or without the `mp-` prefix.

## Requirements

```
mp-api>=0.41
pymatgen>=2024.1.1
matplotlib>=3.7
sqlalchemy>=2.0
numpy>=1.24
fastapi>=0.110
uvicorn[standard]>=0.29
```

A free Materials Project API key is required for stages 1 and 3: [next-gen.materialsproject.org/api](https://next-gen.materialsproject.org/api).
