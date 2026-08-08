from __future__ import annotations

import json
from pathlib import Path

from .config import AgentConfig
from .generation import EvidenceGroundedGenerator, LocalHuggingFaceGenerator
from .graph import SupportGraph
from .retrieval import LocalRetriever
from .verification import ResponseVerifier


class OrbitDeskAgent:
    def __init__(self, config: AgentConfig | None = None, *, offline_demo: bool = False):
        self.config = config or AgentConfig()
        self.retriever = LocalRetriever(
            self.config if not offline_demo else self.config.__class__(
                data_dir=self.config.data_dir,
                model_dir=self.config.model_dir,
                max_revisions=self.config.max_revisions,
                top_k=self.config.top_k,
                use_huggingface=False,
                model=self.config.model,
            )
        )
        if offline_demo:
            generator = EvidenceGroundedGenerator()
        else:
            generator = LocalHuggingFaceGenerator(
                self.config.model_dir / "generation",
                device=self.config.model.device,
            )
        schema = json.loads(self.config.schema_path.read_text(encoding="utf-8"))
        self.workflow = SupportGraph(
            self.retriever,
            generator,
            ResponseVerifier(schema),
            max_revisions=self.config.max_revisions,
        )

    def answer(self, question: str, *, force_bad_first_attempt: bool = False) -> dict:
        if not question.strip():
            raise ValueError("question must not be empty")
        state = self.workflow.invoke(
            question,
            force_bad_first_attempt=force_bad_first_attempt,
        )
        response = state["response"]
        return {
            "response": response.model_dump(exclude_none=False),
            "trace": state.get("trace", []),
            "retrieval_backend": self.retriever.backend,
            "revision_count": state.get("revision_count", 0),
            "verification_errors": state.get("verification_errors", []),
        }