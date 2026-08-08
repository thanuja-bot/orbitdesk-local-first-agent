from __future__ import annotations

import re

from jsonschema import Draft202012Validator

from .types import Passage, SupportResponse


FORBIDDEN_INSTRUCTIONS = (
    "profile > personal token",
    "issue a refund",
    "paste your password",
    "paste your token",
    "send your oauth",
)


class ResponseVerifier:
    def __init__(self, schema: dict):
        self.validator = Draft202012Validator(schema)

    def verify(
        self,
        response: SupportResponse,
        passages: list[Passage],
    ) -> list[str]:
        errors: list[str] = []
        payload = response.model_dump(exclude_none=False)
        errors.extend(error.message for error in self.validator.iter_errors(payload))
        if not response.sources and response.classification != "safe_failure":
            errors.append("The response has no source references.")

        allowed_ids = {passage.source_id for passage in passages} | {"KB-010"}
        for reference in response.sources:
            if reference.source_id not in allowed_ids:
                errors.append(f"Source {reference.source_id} was not retrieved.")
            if not reference.passage.strip():
                errors.append(f"Source {reference.source_id} has an empty passage.")

        lower_answer = response.answer.lower()
        for phrase in FORBIDDEN_INSTRUCTIONS:
            if phrase in lower_answer:
                errors.append(f"Unsupported or unsafe instruction detected: {phrase}.")

        # A source-aware answer must have at least one meaningful evidence
        # anchor. This is intentionally simple and auditable, not a claim of
        # semantic entailment.
        evidence = " ".join(p.text.lower() for p in passages)
        answer_terms = {
            token for token in re.findall(r"[a-z0-9_]+", lower_answer)
            if len(token) > 4
        }
        evidence_terms = set(re.findall(r"[a-z0-9_]+", evidence))
        if response.classification == "answerable" and answer_terms:
            if len(answer_terms & evidence_terms) < 2:
                errors.append("The answer has insufficient lexical support in retrieved evidence.")

        superseded_ids = {p.source_id for p in passages if p.status == "superseded"}
        if superseded_ids & {source.source_id for source in response.sources}:
            errors.append("Superseded case guidance must not be cited as current.")
        return errors