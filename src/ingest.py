"""
ingest.py

Loads every per-record JSON file in data/xanes/ (written by
mp_xas_fetch.py) into the SQLite DB defined in db.py. Idempotent: reruns
upsert existing rows by record_id instead of duplicating them, so this is
safe to run again after a fresh mp_xas_fetch.py pull.

Usage:
    python src/ingest.py                       # default data/xanes -> data/spectrahub.db
    python src/ingest.py --data-dir data/xanes --db data/spectrahub.db
"""

import argparse
import json
from pathlib import Path

from db import get_session, Spectrum


def load_record(session, record: dict) -> str:
    """Insert or update one record. Returns 'inserted' or 'updated'."""
    record_id = record["record_id"]
    x_values = record.get("x_values", [])
    existing = session.get(Spectrum, record_id)

    fields = dict(
        material_formula=record["material_formula"],
        modality=record["modality"],
        edge=record.get("edge"),
        absorbing_element=record.get("absorbing_element"),
        mp_id=record.get("mp_id"),
        x_axis=record["x_axis"],
        y_axis=record["y_axis"],
        x_values_json=json.dumps(record.get("x_values", [])),
        y_values_json=json.dumps(record.get("y_values", [])),
        n_points=len(x_values),
        energy_min_ev=min(x_values) if x_values else None,
        energy_max_ev=max(x_values) if x_values else None,
        source_type=record["source_type"],
        citation=record["citation"],
        license=record.get("license"),
        digitization_error_estimate=record.get("digitization_error_estimate"),
        is_highlighted=bool(record.get("linked_properties", {}).get("is_highlighted", False)),
        retrieved_at_utc=record["retrieved_at_utc"],
        notes=record.get("notes"),
    )

    if existing is None:
        session.add(Spectrum(record_id=record_id, **fields))
        return "inserted"
    for key, value in fields.items():
        setattr(existing, key, value)
    return "updated"


def main():
    parser = argparse.ArgumentParser(description="Load SpectraHub JSON records into SQLite.")
    parser.add_argument("--data-dir", default="data/xanes")
    parser.add_argument("--db", default=None, help="Path to SQLite file (default: data/spectrahub.db)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    db_path = Path(args.db) if args.db else None
    session = get_session(db_path)

    json_files = sorted(
        p for p in data_dir.glob("*.json")
        if p.name != "MANIFEST.json"
    )
    if not json_files:
        print(f"No record JSON files found in {data_dir}/ -- nothing to ingest.")
        return

    inserted = updated = skipped = 0
    for path in json_files:
        try:
            record = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            print(f"  [SKIP] {path.name}: unreadable ({exc})")
            skipped += 1
            continue

        required = ["record_id", "material_formula", "modality", "x_axis",
                    "y_axis", "source_type", "citation", "retrieved_at_utc"]
        missing = [k for k in required if k not in record]
        if missing:
            print(f"  [SKIP] {path.name}: missing fields {missing}")
            skipped += 1
            continue

        result = load_record(session, record)
        if result == "inserted":
            inserted += 1
        else:
            updated += 1

    session.commit()
    total_in_db = session.query(Spectrum).count()
    print(f"Ingested {len(json_files)} JSON files from {data_dir}/: "
          f"{inserted} inserted, {updated} updated, {skipped} skipped.")
    print(f"Total rows in spectra table: {total_in_db}")


if __name__ == "__main__":
    main()
