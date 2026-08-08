from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .types import Passage


def _frontmatter(text: str) -> dict[str, str]:
    if not text.startswith("---"):
        return {}
    parts = text.split("---", 2)
    if len(parts) != 3:
        return {}
    result: dict[str, str] = {}
    for line in parts[1].splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            result[key.strip()] = value.strip()
    return result


def _clean(text: str) -> str:
    text = re.sub(r"^---.*?---\s*", "", text, count=1, flags=re.DOTALL)
    text = re.sub(r"^Mac OS X.*$", "", text, flags=re.MULTILINE)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def load_corpus(data_dir: Path) -> list[Passage]:
    """Load current KB documents plus only non-superseded resolved cases."""
    passages: list[Passage] = []
    for path in sorted((data_dir / "knowledge_base").glob("*.md")):
        raw = path.read_text(encoding="utf-8")
        meta = _frontmatter(raw)
        passages.append(
            Passage(
                source_id=meta.get("document_id", path.stem),
                title=meta.get("title", path.stem),
                text=_clean(raw),
                source_type="knowledge_base",
                status="current",
                priority=100,
            )
        )

    cases_path = data_dir / "resolved_cases.json"
    cases: dict[str, Any] = json.loads(cases_path.read_text(encoding="utf-8"))
    for case in cases.get("cases", []):
        status = case.get("status", "resolved")
        if status == "superseded":
            # Retain no superseded resolution in the searchable corpus. This
            # makes it impossible for historical guidance to leak into answers.
            continue
        text_parts = [
            case.get("title", ""),
            "Symptoms: " + "; ".join(case.get("symptoms", [])),
            "Resolution: " + "; ".join(case.get("resolution", [])),
            "Important limit: " + case.get("important_limit", ""),
        ]
        passages.append(
            Passage(
                source_id=case["case_id"],
                title=case.get("title", case["case_id"]),
                text="\n".join(part for part in text_parts if part.strip()),
                source_type="resolved_case",
                status="resolved",
                priority=50,
            )
        )
    return passages