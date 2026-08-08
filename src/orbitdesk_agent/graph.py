from __future__ import annotations

from typing import Literal

from langgraph.graph import END, START, StateGraph

from .generation import ResponseGenerator
from .retrieval import LocalRetriever
from .types import AgentState, SupportResponse, TraceEvent
from .verification import ResponseVerifier


def _log(state: AgentState, node: str, event: str, detail: str = "") -> list[dict]:
    return [
        *state.get("trace", []),
        TraceEvent(node=node, event=event, detail=detail).model_dump(),
    ]


class SupportGraph:
    def __init__(self, retriever: LocalRetriever, generator: ResponseGenerator, verifier: ResponseVerifier, max_revisions: int = 1):
        self.retriever = retriever
        self.generator = generator
        self.verifier = verifier
        self.max_revisions = max_revisions
        self.graph = self._build()

    def _build(self):
        workflow = StateGraph(AgentState)
        workflow.add_node("triage", self.triage)
        workflow.add_node("retrieve", self.retrieve)
        workflow.add_node("generate", self.generate)
        workflow.add_node("verify", self.verify)
        workflow.add_node("revise", self.revise)
        workflow.add_node("safe_failure", self.safe_failure)
        workflow.add_edge(START, "triage")
        workflow.add_conditional_edges("triage", self.route_after_triage)
        workflow.add_edge("retrieve", "generate")
        workflow.add_edge("generate", "verify")
        workflow.add_conditional_edges("verify", self.route_after_verify)
        workflow.add_edge("revise", "verify")
        workflow.add_edge("safe_failure", END)
        return workflow.compile()

    def triage(self, state: AgentState) -> AgentState:
        question = state["question"].lower()
        if any(term in question for term in ("refund", "legal advice", "cancel my subscription")):
            classification = "out_of_scope"
            reason = "The request asks for billing action or legal advice outside the support scope."
        elif "render_failed" in question or (
            "two" in question and "failed" in question and "export" in question
        ):
            classification = "requires_escalation"
            reason = "The request describes a documented escalation condition."
        elif "not working" in question and not any(
            term in question for term in ("connection", "error", "refresh", "sync id")
        ):
            classification = "requires_clarification"
            reason = "The request lacks the connection identity, state, and error details needed to troubleshoot."
        elif any(term in question for term in ("timezone", "credential", "api", "export", "dashboard")):
            classification = "answerable"
            reason = "The request matches documented OrbitDesk support topics."
        else:
            classification = "requires_clarification"
            reason = "The request does not identify a documented OrbitDesk object or symptom."
        return {
            **state,
            "classification": classification,
            "triage_reason": reason,
            "revision_count": state.get("revision_count", 0),
            "max_revisions": state.get("max_revisions", self.max_revisions),
            "trace": _log(state, "triage", "exit", classification),
        }

    @staticmethod
    def route_after_triage(state: AgentState) -> Literal["retrieve", "generate"]:
        if state["classification"] in {"answerable", "requires_escalation"}:
            return "retrieve"
        return "generate"

    def retrieve(self, state: AgentState) -> AgentState:
        result = self.retriever.search(state["question"])
        detail = f"{len(result.passages)} passages via {result.backend}; {result.latency_ms:.1f} ms"
        return {
            **state,
            "passages": result.passages,
            "trace": _log(state, "retrieve", "exit", detail),
        }

    def generate(self, state: AgentState) -> AgentState:
        response = self.generator.generate(
            state["question"],
            state["classification"],
            state.get("passages", []),
            revision=False,
            force_bad=state.get("force_bad_first_attempt", False),
        )
        return {
            **state,
            "draft": response,
            "trace": _log(state, "generate", "exit", f"attempt={state.get('revision_count', 0) + 1}"),
        }

    def verify(self, state: AgentState) -> AgentState:
        draft = state["draft"]
        errors = self.verifier.verify(draft, state.get("passages", []))
        detail = "passed" if not errors else "; ".join(errors[:2])
        return {
            **state,
            "response": draft if not errors else state.get("response"),
            "verification_errors": errors,
            "trace": _log(state, "verify", "exit", detail),
        }

    @staticmethod
    def route_after_verify(state: AgentState) -> Literal["revise", "safe_failure", "__end__"]:
        if not state.get("verification_errors"):
            return END
        if state.get("revision_count", 0) < state.get("max_revisions", 1):
            return "revise"
        return "safe_failure"

    def revise(self, state: AgentState) -> AgentState:
        next_count = state.get("revision_count", 0) + 1
        response = self.generator.generate(
            state["question"],
            state["classification"],
            state.get("passages", []),
            revision=True,
            force_bad=False,
        )
        return {
            **state,
            "revision_count": next_count,
            "draft": response,
            "trace": _log(state, "revise", "exit", f"revision={next_count}"),
        }

    def safe_failure(self, state: AgentState) -> AgentState:
        fallback = SupportResponse(
            classification="safe_failure",
            answer="I could not verify a safe, evidence-supported answer from the supplied OrbitDesk material. Please provide the exact object, error code, and relevant timestamps, without sharing secrets.",
            sources=[],
            confidence=0.0,
            requires_human=True,
            reason="The generated response failed verification after the maximum allowed revision.",
            warnings=state.get("verification_errors", []),
        )
        return {
            **state,
            "response": fallback,
            "trace": _log(state, "safe_failure", "exit", "verification budget exhausted"),
        }

    def invoke(self, question: str, *, force_bad_first_attempt: bool = False) -> AgentState:
        return self.graph.invoke(
            {
                "question": question,
                "trace": [],
                "revision_count": 0,
                "max_revisions": self.max_revisions,
                "force_bad_first_attempt": force_bad_first_attempt,
            }
        )