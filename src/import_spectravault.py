"""
import_spectravault.py

The SpectraHub-side half of the SpectraHub <-> SpectraVault connection.
Reads SpectraVault's own data/MANIFEST.json (its already-built index of
every record across every source), validates each record against THIS
project's schema.py, tags it with import provenance, and upserts it into
spectrahub.db.

Design, stated explicitly (this is the piece the two projects' READMEs
both point to as "how the connection actually works"):
  - Local sibling-folder path, not a git submodule and not a packaged
    release artifact. Both repos live on the same machine; SpectraVault is
    just a second folder this script reads JSON out of. No network call,
    no build step, no version pinning to get wrong.
  - One-directional. SpectraHub reads from SpectraVault; SpectraVault has
    no code that knows SpectraHub exists. Keeping the dependency one-way
    means SpectraVault stays usable/testable completely on its own.
  - No copy-pasted upsert logic. This script reuses ingest.py's own
    load_record() for the actual DB write -- a SpectraVault record maps
    onto the exact same Spectrum row shape ingest.py already knows how to
    upsert, so duplicating that logic here would be the "copy-paste
    several times" failure mode this connection was explicitly asked to
    avoid, not a defensible second implementation.
  - Idempotent by record_id, same as ingest.py: rerunning this script after
    SpectraVault gets new records only inserts/updates what changed.
  - Every imported record is schema-validated against SpectraHub's OWN
    schema.py before being written -- not just trusted because it already
    passed SpectraVault's validation. The two schemas are kept
    field-compatible on purpose (see both projects' schema.py docstrings),
    so this should always pass for a well-formed SpectraVault record; if it
    doesn't, that's a real incompatibility worth seeing, not something to
    silently coerce past.
  - Traceability: every imported record's `notes` field gets an appended
    tag recording it came from SpectraVault, which source folder within
    SpectraVault, SpectraVault's confidence score for it, and the exact
    git commit of the SpectraVault repo it was imported from -- so "why is
    this row in my database" is always answerable later, including after
    SpectraVault's data has moved on.
  - Confidence filtering is optional and explicit (--min-confidence), not
    a silent default -- importing everything and letting downstream
    analysis filter by confidence_score (once that column exists on the
    SpectraHub side -- see the note in main()) is a defensible choice too.
    Rejected-by-threshold records are reported, not silently dropped
    without a count.

Usage:
    python src/import_spectravault.py --vault-path ../SpectraVault
    python src/import_spectravault.py --vault-path ../SpectraVault --min-confidence 0.5
    python src/import_spectravault.py --vault-path ../SpectraVault --offset 0 --limit 500

--offset/--limit scan only a slice of the vault's manifest -- added once
SpectraVault's vault grew past ~2,000 records (RRUFF XRD + infrared),
where reading every record file individually over a network-mounted
folder plus the DB upsert can outrun a single shell command's timeout.
Each invocation still commits what it processed, so running this a few
times with increasing --offset covers the whole vault; rerunning any
slice is safe since load_record() upserts by record_id.
"""

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from db import get_session
from schema import validate_record
from ingest import load_record


def _vault_git_commit(vault_path: Path) -> str:
    """Short git commit hash of the SpectraVault checkout being imported
    from, for traceability. Returns 'unknown' rather than raising if the
    folder isn't a git repo (e.g. someone points this at a plain data
    export) -- traceability degrading gracefully beats the whole import
    failing over a missing .git folder."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=vault_path, capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return "unknown"


def load_vault_manifest(vault_path: Path) -> list:
    manifest_path = vault_path / "data" / "MANIFEST.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"No MANIFEST.json at {manifest_path} -- run SpectraVault's "
            f"src/build_manifest.py first."
        )
    with open(manifest_path, encoding="utf-8") as f:
        return json.load(f)


def tag_provenance(record: dict, entry: dict, vault_commit: str) -> dict:
    """Returns a copy of record with an import-provenance note appended --
    never overwrites the original notes, just extends them."""
    record = dict(record)
    tag = (
        f"Imported from SpectraVault (source={entry['source']}, "
        f"vault_confidence_score={entry.get('confidence_score')}, "
        f"vault_git_commit={vault_commit}, "
        f"imported_at_utc={datetime.now(timezone.utc).isoformat()})"
    )
    existing_notes = record.get("notes") or ""
    record["notes"] = f"{existing_notes} | {tag}" if existing_notes else tag
    return record


def main():
    parser = argparse.ArgumentParser(description="Import SpectraVault records into spectrahub.db.")
    parser.add_argument("--vault-path", required=True, help="Path to the SpectraVault repo checkout.")
    parser.add_argument("--db", default=None, help="Path to SQLite file (default: data/spectrahub.db)")
    parser.add_argument("--min-confidence", type=float, default=0.0,
                         help="Skip vault records below this confidence_score (default: 0.0, import everything).")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    vault_path = Path(args.vault_path).resolve()
    manifest = load_vault_manifest(vault_path)
    vault_commit = _vault_git_commit(vault_path)

    total = len(manifest)
    manifest = manifest[args.offset:(args.offset + args.limit) if args.limit else None]
    if args.offset or args.limit:
        print(f"{total} manifest entries total; processing slice "
              f"[{args.offset}:{args.offset + len(manifest)}]")

    session = get_session(Path(args.db) if args.db else None)

    inserted = updated = skipped_confidence = skipped_invalid = 0
    for entry in manifest:
        confidence = entry.get("confidence_score") or 0.0
        if confidence < args.min_confidence:
            skipped_confidence += 1
            continue

        record_path = vault_path / "data" / entry["path"]
        with open(record_path, encoding="utf-8") as f:
            record = json.load(f)

        problems = validate_record(record)
        if problems:
            print(f"  [SKIP] {entry['record_id']}: fails SpectraHub schema validation: {problems}")
            skipped_invalid += 1
            continue

        record = tag_provenance(record, entry, vault_commit)
        result = load_record(session, record)
        if result == "inserted":
            inserted += 1
        else:
            updated += 1

    session.commit()
    print(f"SpectraVault import from {vault_path} (commit {vault_commit}): "
          f"{inserted} inserted, {updated} updated, "
          f"{skipped_confidence} skipped (below --min-confidence {args.min_confidence}), "
          f"{skipped_invalid} skipped (failed validation).")


if __name__ == "__main__":
    main()
