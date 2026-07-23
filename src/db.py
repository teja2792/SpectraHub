"""
db.py

SQLAlchemy schema + engine/session helpers for SpectraHub.

Design decisions (stated explicitly, not silent choices):
  - `x_values`/`y_values` are stored as JSON-encoded TEXT columns on the
    `spectra` row itself, NOT normalized into a separate points table.
    Downstream feature engineering and ML need the full array per record
    loaded into numpy anyway, so one-row-per-record is the right shape
    here -- a fully normalized points table (2004 records x ~100-500
    points each = ~700k+ rows) would just mean a join and a groupby to get
    back to the same numpy array, for no query benefit this project needs.
    If cross-record point-level SQL queries become a real requirement
    later, add a normalized table then -- don't build it speculatively now.
  - SQLite for now (single file, zero setup, fine for ~2k-100k records).
    Models are plain SQLAlchemy ORM with no SQLite-specific types, so
    swapping the engine URL to Postgres later does not require a schema
    rewrite.
  - `labels` is a separate table keyed on mp_id, deliberately decoupled
    from `spectra` (which is keyed on record_id). Multiple spectra
    (XANES/XAFS/EXAFS) share one mp_id and therefore one label row --
    do not duplicate label data per spectrum row.
"""

import json
from pathlib import Path
from typing import Optional

from sqlalchemy import (
    create_engine, Column, String, Float, Integer, Boolean, Text, DateTime,
)
from sqlalchemy.orm import declarative_base, sessionmaker

Base = declarative_base()

DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent / "data" / "spectrahub.db"


class Spectrum(Base):
    __tablename__ = "spectra"

    record_id = Column(String, primary_key=True)
    material_formula = Column(String, nullable=False, index=True)
    modality = Column(String, nullable=False, index=True)
    edge = Column(String, nullable=True)
    absorbing_element = Column(String, nullable=True, index=True)
    mp_id = Column(String, nullable=True, index=True)
    x_axis = Column(String, nullable=False)
    y_axis = Column(String, nullable=False)
    x_values_json = Column(Text, nullable=False)
    y_values_json = Column(Text, nullable=False)
    n_points = Column(Integer, nullable=False)
    energy_min_ev = Column(Float, nullable=True)
    energy_max_ev = Column(Float, nullable=True)
    source_type = Column(String, nullable=False)
    citation = Column(Text, nullable=False)
    license = Column(String, nullable=True)
    digitization_error_estimate = Column(Float, nullable=True)
    is_highlighted = Column(Boolean, nullable=False, default=False)
    retrieved_at_utc = Column(String, nullable=False)
    notes = Column(Text, nullable=True)

    def x_values(self):
        return json.loads(self.x_values_json)

    def y_values(self):
        return json.loads(self.y_values_json)


class MaterialLabel(Base):
    """Structural/electronic labels per unique mp_id, used as ML training
    targets. Populated separately by src/label_fetch.py (task 2) --
    empty until that step runs. Kept as its own table (not columns bolted
    onto `spectra`) because it's keyed on mp_id, not record_id, and one
    mp_id maps to up to 3 spectra rows."""
    __tablename__ = "labels"

    mp_id = Column(String, primary_key=True)
    oxidation_state = Column(Float, nullable=True)
    coordination_number = Column(Integer, nullable=True)
    mean_bond_length_angstrom = Column(Float, nullable=True)
    label_source = Column(String, nullable=True)  # e.g. "pymatgen BVAnalyzer"
    fetched_at_utc = Column(String, nullable=True)


class SpectralFeatures(Base):
    """Engineered numerical features per INDIVIDUAL spectrum (keyed on
    record_id, unlike `labels` which is keyed on mp_id) -- a material with
    XANES/XAFS/EXAFS records gets up to 3 separate feature rows, because
    edge shape genuinely differs across those modalities, not just
    resolution. Populated by src/feature_engineering.py (task 3).

    These are inputs to the ML tasks (clustering/similarity/supervised
    models), not raw physics -- see feature_engineering.py's module
    docstring for the exact algorithm and its documented limitations
    (this is a lightweight numpy implementation, not a rigorous XAS
    analysis package like Athena/Larch)."""
    __tablename__ = "spectral_features"

    record_id = Column(String, primary_key=True)
    edge_energy_ev = Column(Float, nullable=True)
    edge_jump = Column(Float, nullable=True)
    max_derivative = Column(Float, nullable=True)
    white_line_energy_ev = Column(Float, nullable=True)
    white_line_intensity = Column(Float, nullable=True)
    pre_edge_energy_ev = Column(Float, nullable=True)
    pre_edge_intensity = Column(Float, nullable=True)
    pre_edge_margin_ev = Column(Float, nullable=True)  # how much data existed below the edge to search
    computed_at_utc = Column(String, nullable=True)


class SpectralCluster(Base):
    """Unsupervised cluster assignment per XANES record, from
    src/clustering_similarity.py (task 4). Only XANES records get
    clustered -- see that script's module docstring for why."""
    __tablename__ = "spectral_clusters"

    record_id = Column(String, primary_key=True)
    cluster_id = Column(Integer, nullable=False)
    algorithm = Column(String, nullable=True)  # e.g. "kmeans k=10"
    computed_at_utc = Column(String, nullable=True)


def get_engine(db_path: Optional[Path] = None):
    path = db_path or DEFAULT_DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    return create_engine(f"sqlite:///{path}")


def get_session(db_path: Optional[Path] = None):
    engine = get_engine(db_path)
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)()
