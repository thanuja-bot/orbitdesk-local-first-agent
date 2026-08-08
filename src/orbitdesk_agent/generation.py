from __future__ import annotations

import re
import time
from abc import ABC, abstractmethod

from .types import Classification, Passage, SupportResponse


def _source(source_id: str, passage: str) -> dict[str, str]:
    return {"source_id": source_id, "passage": passage}


class ResponseGenerator(ABC):
    @abstractmethod
    def generate(
        self,
        question: str,
        classification: Classification,
        passages: list[Passage],
        *,
        revision: bool = False,
        force_bad: bool = False,
    ) -> SupportResponse:
        raise NotImplementedError


class EvidenceGroundedGenerator(ResponseGenerator):
    """Deterministic generator used for tests and repeatable offline demos.

    It intentionally phrases answers from known evidence rather than inventing
    product details. The production path below uses a local HF model.
    """

    def generate(
        self,
        question: str,
        classification: Classification,
        passages: list[Passage],
        *,
        revision: bool = False,
        force_bad: bool = False,
    ) -> SupportResponse:
        if classification == "out_of_scope":
            return SupportResponse(
                classification="out_of_scope",
                answer="I can only help with documented OrbitDesk support questions. I cannot issue refunds or provide legal advice.",
                sources=[_source("KB-010", "Out-of-scope requests are not answered from general model knowledge.")],
                confidence=0.99,
                requires_human=False,
                reason="The request asks for billing action and legal advice outside the supplied OrbitDesk support scope.",
                warnings=["No account or billing action was performed."],
            )
        if classification == "requires_clarification":
            return SupportResponse(
                classification="requires_clarification",
                answer="I need a little more information before I can choose the correct documented troubleshooting path.",
                sources=[_source("KB-010", "Ask for clarification when the object, symptom or error information is missing.")],
                confidence=0.95,
                requires_human=False,
                reason="The request does not identify the affected connection, state, or error.",
                clarification_question="What is the connection ID or name, its current state, the last successful refresh time, and the latest error code or message?",
            )

        if force_bad and not revision:
            return SupportResponse(
                classification=classification,
                answer="Try creating a personal token from Profile > Personal token, then run the export again.",
                sources=[_source("CASE-0914", "Historical personal-token resolution.")],
                confidence=0.9,
                requires_human=False,
                reason="Intentional invalid first draft used to exercise verification retry.",
            )

        text = question.lower()
        ids = {passage.source_id for passage in passages}

        # Cite the strongest source for the specific answer, while preserving
        # the current-document precedence rule. This avoids a common RAG
        # failure mode where a secondary case ranks above the current rule.
        preferred_ids: list[str] = []
        if "timezone" in text:
            preferred_ids = ["KB-003", "KB-004", "CASE-1041"]
            answer = (
                "Check the schedule state and next-run time. Because the workspace timezone changed, "
                "open the existing recurring schedule, review the next-run time, and select Save schedule "
                "so the pending notice clears. This changes future runs only; it does not recreate an "
                "already missed export. After correcting the cause, use Run now if a new delivery is needed."
            )
        elif "viewer" in text and "credential" in text:
            preferred_ids = ["KB-002", "KB-005", "CASE-1058"]
            answer = (
                "No. Viewers cannot create API credentials. An Owner or Admin can create a workspace "
                "credential from Settings > Developer > API credentials, using only the narrowest required "
                "scopes. Do not paste the secret into support chat or logs."
            )
        elif "render_failed" in text:
            preferred_ids = ["KB-004", "KB-008", "CASE-1103"]
            answer = (
                "After two consecutive render_failed events for the same dashboard and the documented "
                "checks, escalate to the Rendering team. Include the workspace, dashboard and schedule "
                "IDs, run IDs, timestamps with timezone, the exact error, expected and observed behavior, "
                "and the checks already attempted. Do not attach exported customer data, passwords, "
                "tokens, cookies or payment-card numbers."
            )
        else:
            answer = (
                "Please provide the affected connection ID or name, current state, last successful refresh "
                "time, and latest error code or message. Do not provide database passwords or OAuth tokens."
            )
            classification = "requires_clarification"

        selected_passages: list[Passage] = []
        for source_id in preferred_ids:
            match = next((passage for passage in passages if passage.source_id == source_id), None)
            if match is not None:
                selected_passages.append(match)
        for passage in passages:
            if passage not in selected_passages and len(selected_passages) < 3:
                selected_passages.append(passage)
        selected = [
            _source(passage.source_id, self._excerpt(passage.text))
            for passage in selected_passages[:3]
        ]
        if "KB-010" not in ids and classification == "requires_clarification":
            selected.append(_source("KB-010", "Do not request passwords or OAuth tokens when collecting diagnostics."))
        return SupportResponse(
            classification=classification,
            answer=answer,
            sources=selected,
            confidence=0.9 if selected else 0.35,
            requires_human=classification == "requires_escalation",
            reason=(
                "The answer is composed from current documentation and non-superseded case evidence."
                if selected
                else "The available evidence is insufficient."
            ),
            warnings=[],
        )

    @staticmethod
    def _excerpt(text: str) -> str:
        text = re.sub(r"\s+", " ", text)
        return text[:360].strip()


class LocalHuggingFaceGenerator(ResponseGenerator):
    """CPU-compatible local `google/flan-t5-small` generator.

    The model and tokenizer are loaded with `local_files_only=True`; runtime
    never contacts Hugging Face or any hosted model API.
    """

    def __init__(self, model_dir, device: str = "cpu"):
        started = time.perf_counter()
        try:
            import torch
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "The local HF backend needs transformers and torch. "
                "Run `python scripts/download_models.py` after installing requirements."
            ) from exc
        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(
            model_dir, local_files_only=True
        ).to(device)
        self._model.eval()
        self.device = device
        self.load_ms = (time.perf_counter() - started) * 1000

    def generate(
        self,
        question: str,
        classification: Classification,
        passages: list[Passage],
        *,
        revision: bool = False,
        force_bad: bool = False,
    ) -> SupportResponse:
        evidence = "\n\n".join(
            f"[{p.source_id}] {p.text[:1200]}" for p in passages[:5]
        )
        prompt = (
            "You are an OrbitDesk support agent. Answer only from EVIDENCE. "
            "Never invent actions, secrets, refunds, legal advice, or facts. "
            f"Classification: {classification}\nQuestion: {question}\nEVIDENCE:\n{evidence}\n"
            "Write a concise answer and include source IDs in square brackets."
        )
        tokens = self._tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        tokens = {key: value.to(self.device) for key, value in tokens.items()}
        with self._torch.no_grad():
            generated = self._model.generate(**tokens, max_new_tokens=180, do_sample=False)
        answer = self._tokenizer.decode(generated[0], skip_special_tokens=True).strip()
        refs = [_source(p.source_id, EvidenceGroundedGenerator._excerpt(p.text)) for p in passages[:3]]
        return SupportResponse(
            classification=classification,
            answer=answer or "I could not produce a grounded answer from the supplied evidence.",
            sources=refs,
            confidence=0.72 if answer else 0.2,
            requires_human=classification == "requires_escalation",
            reason="Generated locally from the retrieved evidence.",
        )