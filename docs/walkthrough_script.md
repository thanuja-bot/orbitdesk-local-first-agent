# 4–7 minute walkthrough script

## 0:00–0:45 — What the project does

“This is a local-first OrbitDesk support agent. It answers only from the supplied
knowledge base and eligible resolved cases. There are no hosted language-model
calls at runtime.”

Show the repository tree and the supplied `knowledge_base/`, `resolved_cases.json`,
and `output_schema.json`.

## 0:45–1:45 — Explain the graph

Open `docs/graph_diagram.png` and explain:

1. Triage chooses the route.
2. Answerable and escalation questions retrieve evidence.
3. Generation produces a structured response.
4. Verification checks schema, citations, evidence overlap, unsafe instructions,
   and superseded-case usage.
5. Verification can route to one revision only, then to safe failure.

Point out that `AgentState` is a typed `TypedDict` and the response is a strict
Pydantic model.

## 1:45–2:20 — Show local model loading

Run:

```bash
python scripts/download_models.py
export HF_HUB_OFFLINE=1
python -m orbitdesk_agent.cli --pretty \
  "I am a read-only Viewer. Can I create an API credential for a reporting script?"
```

Show `generation.py` and `retrieval.py`, highlighting the pinned model names and
`local_files_only=True`. State the CPU-only device and the approximate timings
documented in the README.

## 2:20–3:30 — Live answerable and clarification runs

Run the timezone question from Q-001. Show `classification: answerable`,
references to `KB-003` and `KB-004`, and the trace containing triage, retrieval,
generation, and verification.

Run Q-003:

```text
Our data sync is not working. Can you tell me how to fix it?
```

Show `requires_clarification` and the diagnostic question. Emphasize that the
agent does not guess an error code or ask for credentials.

## 3:30–4:20 — Escalation run

Run Q-004. Show `requires_escalation`, `requires_human: true`, references to
`KB-004` and `KB-008`, and the safe list of IDs/timestamps/error details.
Point out that exported customer data is not requested.

## 4:20–5:10 — Verification and revision

Run:

```bash
python -m orbitdesk_agent.cli --offline-demo --pretty \
  --force-bad-first-attempt \
  "I am a read-only Viewer. Can I create an API credential for a reporting script?"
```

Show the trace: `verify` rejects the removed personal-token instruction, then
`revise` runs once, then the corrected answer passes. Explain that a second
failure would route to `safe_failure`, preventing an infinite loop.

## 5:10–5:45 — Tests and design trade-off

Run `pytest -q` and show all tests passing. Mention that the test suite checks
graph routing independently of exact model wording, superseded-case filtering,
schema/safety behavior, and the five required sample paths.

Close with the trade-off: a small local model is slower/weaker than a hosted
model, but the design preserves offline operation and auditable evidence.