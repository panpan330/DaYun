from __future__ import annotations

from pathlib import Path
from threading import Lock

from app.evaluation.bad_case_registry import BadCaseRecord, BadCaseRegistry


_registry_lock = Lock()


def append_production_bad_case(path: Path, record: BadCaseRecord) -> BadCaseRecord:
    """Atomically append one reviewed production record to the formal local registry."""
    with _registry_lock:
        registry = BadCaseRegistry.model_validate_json(path.read_text(encoding="utf-8"))
        existing = next((item for item in registry.records if item.id == record.id), None)
        if existing is not None:
            return existing
        updated = registry.model_copy(update={"records": [*registry.records, record]})
        temporary = path.with_suffix(f"{path.suffix}.tmp")
        temporary.write_text(updated.model_dump_json(indent=2) + "\n", encoding="utf-8")
        temporary.replace(path)
        return record


def update_production_bad_case(path: Path, record: BadCaseRecord) -> BadCaseRecord:
    """Atomically replace one existing production record (matched by id) in the registry."""
    with _registry_lock:
        registry = BadCaseRegistry.model_validate_json(path.read_text(encoding="utf-8"))
        updated_records = [
            record if item.id == record.id else item for item in registry.records
        ]
        updated = registry.model_copy(update={"records": updated_records})
        temporary = path.with_suffix(f"{path.suffix}.tmp")
        temporary.write_text(updated.model_dump_json(indent=2) + "\n", encoding="utf-8")
        temporary.replace(path)
        return record
