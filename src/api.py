"""
api.py

Read-only FastAPI layer over the SpectraHub SQLite database (spectra,
labels, spectral_features, spectral_clusters, model_predictions).

Design decisions, stated explicitly:
  - READ-ONLY. This API does not expose endpoints to trigger
    mp_xas_fetch.py / label_fetch.py / feature_engineering.py /
    ml_models.py / clustering_similarity.py. Those make real external
    calls to the Materials Project API and can run for many minutes (the
    full metal-oxide scan took ~25+ min); they stay explicit scripts you
    run yourself, not remote-triggerable jobs.
  - The /spectra/{record_id}/similar endpoint reuses
    clustering_similarity.build_aligned_matrix() and
    cosine_similarities() DIRECTLY (imported, not reimplemented) so
    results here are always identical to what that script reports --
    no risk of a second, subtly different implementation drifting out
    of sync with the first. The aligned XANES matrix is built ONCE at
    startup (see `lifespan` below), not on every request.
  - Prediction results (/predictions) SERVE the already-computed,
    already-verified LOOCV predictions stored by ml_models.py in
    `model_predictions` -- this API does not run k-NN inference live.
    That's a deliberate scope decision: reimplementing single-query k-NN
    here would duplicate ml_models.py's logic for no real benefit, since
    every material in the dataset already has a stored, honestly-labeled
    (correct or wrong) LOOCV prediction to serve instead.
  - DB session per request via FastAPI's Depends(), opened and closed
    per call -- get_session() in db.py creates a fresh SQLite connection
    each time, which is fine for local/single-user use but would want a
    shared connection pool for real concurrent traffic. Stated
    limitation, not hidden.

Run:
    pip install -r requirements.txt
    uvicorn api:app --reload --app-dir src
    # then open http://127.0.0.1:8000/docs for interactive API docs
"""

from contextlib import asynccontextmanager
from typing import List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException, Query, Depends
from pydantic import BaseModel

from db import get_session, Spectrum, SpectralFeatures, MaterialLabel, SpectralCluster, ModelPrediction
from clustering_similarity import build_aligned_matrix, cosine_similarities

_similarity_state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    session = get_session()
    try:
        record_ids, X = build_aligned_matrix(session)
    finally:
        session.close()
    _similarity_state["record_ids"] = record_ids
    _similarity_state["X"] = X
    print(f"[startup] Loaded aligned XANES matrix for similarity search: {len(record_ids)} records.")
    yield
    _similarity_state.clear()


app = FastAPI(
    title="SpectraHub API",
    description="Read-only API over Materials Project FEFF-computed XAS data, "
                 "structural labels, engineered spectral features, clusters, and model predictions.",
    lifespan=lifespan,
)


def get_db():
    session = get_session()
    try:
        yield session
    finally:
        session.close()


# ---------- response models ----------

class MaterialSummary(BaseModel):
    mp_id: str
    formula: str
    element: str
    oxidation_state: Optional[float] = None
    coordination_number: Optional[int] = None
    mean_bond_length_angstrom: Optional[float] = None
    is_highlighted: bool
    n_spectra: int


class SpectrumSummary(BaseModel):
    record_id: str
    modality: str
    n_points: int
    energy_min_ev: Optional[float] = None
    energy_max_ev: Optional[float] = None


class MaterialDetail(BaseModel):
    mp_id: str
    formula: str
    element: str
    oxidation_state: Optional[float] = None
    coordination_number: Optional[int] = None
    mean_bond_length_angstrom: Optional[float] = None
    label_source: Optional[str] = None
    spectra: List[SpectrumSummary]


class SpectrumDetail(BaseModel):
    record_id: str
    material_formula: str
    modality: str
    absorbing_element: Optional[str] = None
    mp_id: Optional[str] = None
    x_values: List[float]
    y_values: List[float]
    edge_energy_ev: Optional[float] = None
    edge_jump: Optional[float] = None
    max_derivative: Optional[float] = None
    white_line_energy_ev: Optional[float] = None
    white_line_intensity: Optional[float] = None
    pre_edge_energy_ev: Optional[float] = None
    pre_edge_intensity: Optional[float] = None
    cluster_id: Optional[int] = None


class SimilarMatch(BaseModel):
    record_id: str
    formula: str
    element: str
    similarity: float
    oxidation_state: Optional[float] = None
    coordination_number: Optional[int] = None


class PredictionRow(BaseModel):
    record_id: str
    formula: str
    task: str
    method: Optional[str] = None
    predicted_value: float
    true_value: float
    correct_or_close: bool


class StatsResponse(BaseModel):
    n_spectra: int
    n_materials: int
    n_labeled_materials: int
    n_xanes_with_features: int
    n_clustered: int
    modality_counts: dict
    prediction_task_counts: dict


# ---------- helpers ----------

def _material_summary(session, mp_id: str) -> Optional[MaterialSummary]:
    specs = session.query(Spectrum).filter_by(mp_id=mp_id).all()
    if not specs:
        return None
    label = session.query(MaterialLabel).filter_by(mp_id=mp_id).one_or_none()
    first = specs[0]
    return MaterialSummary(
        mp_id=mp_id,
        formula=first.material_formula,
        element=first.absorbing_element,
        oxidation_state=label.oxidation_state if label else None,
        coordination_number=label.coordination_number if label else None,
        mean_bond_length_angstrom=label.mean_bond_length_angstrom if label else None,
        is_highlighted=first.is_highlighted,
        n_spectra=len(specs),
    )


# ---------- endpoints ----------

@app.get("/", tags=["meta"])
def root():
    return {"name": "SpectraHub API", "docs": "/docs", "stats": "/stats"}


@app.get("/stats", response_model=StatsResponse, tags=["meta"])
def stats(session=Depends(get_db)):
    n_spectra = session.query(Spectrum).count()
    n_materials = session.query(Spectrum.mp_id).distinct().count()
    n_labeled = session.query(MaterialLabel).count()
    n_xanes_feat = (
        session.query(SpectralFeatures)
        .join(Spectrum, Spectrum.record_id == SpectralFeatures.record_id)
        .filter(Spectrum.modality == "XANES")
        .count()
    )
    n_clustered = session.query(SpectralCluster).count()

    modality_counts = {}
    for modality, count in session.query(Spectrum.modality, Spectrum.record_id).all():
        modality_counts[modality] = modality_counts.get(modality, 0) + 1

    task_counts = {}
    for task, in session.query(ModelPrediction.task).all():
        task_counts[task] = task_counts.get(task, 0) + 1

    return StatsResponse(
        n_spectra=n_spectra, n_materials=n_materials, n_labeled_materials=n_labeled,
        n_xanes_with_features=n_xanes_feat, n_clustered=n_clustered,
        modality_counts=modality_counts, prediction_task_counts=task_counts,
    )


@app.get("/materials", response_model=List[MaterialSummary], tags=["materials"])
def list_materials(
    element: Optional[str] = None,
    formula_contains: Optional[str] = None,
    highlighted_only: bool = False,
    min_oxidation_state: Optional[float] = None,
    max_oxidation_state: Optional[float] = None,
    limit: int = Query(50, le=500),
    offset: int = 0,
    session=Depends(get_db),
):
    query = session.query(Spectrum.mp_id).distinct()
    if element:
        query = query.filter(Spectrum.absorbing_element == element)
    if formula_contains:
        query = query.filter(Spectrum.material_formula.contains(formula_contains))
    if highlighted_only:
        query = query.filter(Spectrum.is_highlighted.is_(True))
    mp_ids = [row[0] for row in query.all() if row[0]]

    results = []
    for mp_id in mp_ids:
        summary = _material_summary(session, mp_id)
        if summary is None:
            continue
        if min_oxidation_state is not None and (
            summary.oxidation_state is None or summary.oxidation_state < min_oxidation_state
        ):
            continue
        if max_oxidation_state is not None and (
            summary.oxidation_state is None or summary.oxidation_state > max_oxidation_state
        ):
            continue
        results.append(summary)

    results.sort(key=lambda m: m.formula)
    return results[offset: offset + limit]


@app.get("/materials/{mp_id}", response_model=MaterialDetail, tags=["materials"])
def get_material(mp_id: str, session=Depends(get_db)):
    specs = session.query(Spectrum).filter_by(mp_id=mp_id).all()
    if not specs:
        raise HTTPException(404, f"No material found with mp_id={mp_id!r}")
    label = session.query(MaterialLabel).filter_by(mp_id=mp_id).one_or_none()
    first = specs[0]
    return MaterialDetail(
        mp_id=mp_id,
        formula=first.material_formula,
        element=first.absorbing_element,
        oxidation_state=label.oxidation_state if label else None,
        coordination_number=label.coordination_number if label else None,
        mean_bond_length_angstrom=label.mean_bond_length_angstrom if label else None,
        label_source=label.label_source if label else None,
        spectra=[
            SpectrumSummary(
                record_id=s.record_id, modality=s.modality, n_points=s.n_points,
                energy_min_ev=s.energy_min_ev, energy_max_ev=s.energy_max_ev,
            ) for s in specs
        ],
    )


@app.get("/spectra/{record_id}", response_model=SpectrumDetail, tags=["spectra"])
def get_spectrum(record_id: str, session=Depends(get_db)):
    spec = session.query(Spectrum).filter_by(record_id=record_id).one_or_none()
    if spec is None:
        raise HTTPException(404, f"No spectrum found with record_id={record_id!r}")
    feat = session.query(SpectralFeatures).filter_by(record_id=record_id).one_or_none()
    cluster = session.query(SpectralCluster).filter_by(record_id=record_id).one_or_none()
    return SpectrumDetail(
        record_id=spec.record_id, material_formula=spec.material_formula, modality=spec.modality,
        absorbing_element=spec.absorbing_element, mp_id=spec.mp_id,
        x_values=spec.x_values(), y_values=spec.y_values(),
        edge_energy_ev=feat.edge_energy_ev if feat else None,
        edge_jump=feat.edge_jump if feat else None,
        max_derivative=feat.max_derivative if feat else None,
        white_line_energy_ev=feat.white_line_energy_ev if feat else None,
        white_line_intensity=feat.white_line_intensity if feat else None,
        pre_edge_energy_ev=feat.pre_edge_energy_ev if feat else None,
        pre_edge_intensity=feat.pre_edge_intensity if feat else None,
        cluster_id=cluster.cluster_id if cluster else None,
    )


@app.get("/spectra/{record_id}/similar", response_model=List[SimilarMatch], tags=["spectra"])
def similar_spectra(record_id: str, top: int = Query(10, le=100), session=Depends(get_db)):
    record_ids = _similarity_state.get("record_ids")
    X = _similarity_state.get("X")
    if not record_ids or record_id not in record_ids:
        raise HTTPException(
            404,
            f"'{record_id}' not found among XANES records with computed features "
            "(similarity search is XANES-only -- see clustering_similarity.py).",
        )
    idx = record_ids.index(record_id)
    sims = cosine_similarities(X, idx)
    order = np.argsort(-sims)

    results = []
    for i in order:
        if record_ids[i] == record_id:
            continue
        spec = session.query(Spectrum).filter_by(record_id=record_ids[i]).one()
        label = session.query(MaterialLabel).filter_by(mp_id=spec.mp_id).one_or_none()
        results.append(SimilarMatch(
            record_id=record_ids[i], formula=spec.material_formula, element=spec.absorbing_element,
            similarity=float(sims[i]),
            oxidation_state=label.oxidation_state if label else None,
            coordination_number=label.coordination_number if label else None,
        ))
        if len(results) >= top:
            break
    return results


@app.get("/clusters/{cluster_id}", response_model=List[SimilarMatch], tags=["clusters"])
def cluster_members(cluster_id: int, session=Depends(get_db)):
    members = session.query(SpectralCluster).filter_by(cluster_id=cluster_id).all()
    if not members:
        raise HTTPException(404, f"No cluster with id={cluster_id}")
    results = []
    for m in members:
        spec = session.query(Spectrum).filter_by(record_id=m.record_id).one()
        label = session.query(MaterialLabel).filter_by(mp_id=spec.mp_id).one_or_none()
        results.append(SimilarMatch(
            record_id=m.record_id, formula=spec.material_formula, element=spec.absorbing_element,
            similarity=1.0,  # not meaningful here, field reused for shape consistency
            oxidation_state=label.oxidation_state if label else None,
            coordination_number=label.coordination_number if label else None,
        ))
    return results


@app.get("/predictions", response_model=List[PredictionRow], tags=["predictions"])
def predictions(
    task: str = Query(..., description="oxidation_state | coordination_number | bond_length"),
    mismatches_only: bool = False,
    tolerance: float = 0.0,
    limit: int = Query(100, le=1000),
    session=Depends(get_db),
):
    rows = session.query(ModelPrediction).filter_by(task=task).all()
    if not rows:
        raise HTTPException(404, f"No stored predictions for task={task!r}. Run src/ml_models.py first.")
    results = []
    for r in rows:
        close = abs(r.predicted_value - r.true_value) <= tolerance
        if mismatches_only and close:
            continue
        spec = session.query(Spectrum).filter_by(record_id=r.record_id).one_or_none()
        results.append(PredictionRow(
            record_id=r.record_id, formula=spec.material_formula if spec else "?",
            task=r.task, method=r.method, predicted_value=r.predicted_value,
            true_value=r.true_value, correct_or_close=close,
        ))
        if len(results) >= limit:
            break
    return results
