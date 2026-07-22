"""
schema.py

Canonical spectral-record schema for SpectraHub. V1 only populates the
"XANES" modality, but the schema covers the full set of modalities in the
original project scope so later phases add data, not a new schema.
"""

from enum import Enum


class Modality(str, Enum):
    XANES = "XANES"
    EXAFS = "EXAFS"
    XPS = "XPS"
    UV_VIS = "UV-Vis"
    RAMAN = "Raman"
    FTIR = "FTIR"
    PL = "PL"
    XRD = "XRD"


class SourceType(str, Enum):
    COMPUTED_DATABASE = "computed-database"
    DEPOSITED_RAW = "deposited-raw-file"
    DIGITIZED_FIGURE = "digitized-from-figure"
    OWN_PUBLISHED_PAPER = "own-published-paper"


SPECTRAL_RECORD_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "SpectraHub Spectral Record",
    "type": "object",
    "required": [
        "record_id", "material_formula", "modality", "x_axis", "y_axis",
        "x_values", "y_values", "source_type", "citation", "retrieved_at_utc",
    ],
    "properties": {
        "record_id": {"type": "string"},
        "material_formula": {"type": "string"},
        "modality": {"enum": [m.value for m in Modality]},
        "edge": {"type": ["string", "null"]},
        "absorbing_element": {"type": ["string", "null"]},
        "mp_id": {"type": ["string", "null"]},
        "x_axis": {"type": "string"},
        "y_axis": {"type": "string"},
        "x_values": {"type": "array", "items": {"type": "number"}},
        "y_values": {"type": "array", "items": {"type": "number"}},
        "source_type": {"enum": [s.value for s in SourceType]},
        "citation": {"type": "string"},
        "license": {"type": ["string", "null"]},
        "digitization_error_estimate": {"type": ["number", "null"]},
        "linked_properties": {"type": "object"},
        "retrieved_at_utc": {"type": "string", "format": "date-time"},
        "notes": {"type": "string"},
    },
}


def validate_record(record: dict) -> list:
    """Minimal dependency-free validator. Returns a list of problem
    strings; empty list means the record passes."""
    problems = []
    for key in SPECTRAL_RECORD_SCHEMA["required"]:
        if key not in record or record[key] in (None, ""):
            problems.append(f"missing required field: {key}")

    if record.get("modality") not in [m.value for m in Modality]:
        problems.append(f"invalid modality: {record.get('modality')!r}")

    if record.get("source_type") not in [s.value for s in SourceType]:
        problems.append(f"invalid source_type: {record.get('source_type')!r}")

    if record.get("source_type") == SourceType.DIGITIZED_FIGURE.value:
        if record.get("digitization_error_estimate") is None:
            problems.append("digitized-from-figure records must set digitization_error_estimate")

    if record.get("source_type") == SourceType.COMPUTED_DATABASE.value and not record.get("mp_id"):
        problems.append("computed-database records should carry an mp_id for traceability")

    x_vals, y_vals = record.get("x_values"), record.get("y_values")
    if isinstance(x_vals, list) and isinstance(y_vals, list) and len(x_vals) != len(y_vals):
        problems.append(f"x_values/y_values length mismatch: {len(x_vals)} vs {len(y_vals)}")

    return problems