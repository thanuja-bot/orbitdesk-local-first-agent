# OrbitDesk Local-First Support Agent Network

An evidence-grounded support agent for the Tantrabodh AI internship assignment.
The workflow answers fictional OrbitDesk support questions using only the supplied
knowledge base and non-superseded resolved cases.

The system is designed to run locally after the Hugging Face model download:
there are no OpenAI, Anthropic, Gemini, hosted LLM, managed vector database, or
other runtime API calls.

## What is implemented

The application is a typed LangGraph state machine:

```text
Question
   |
 Triage -----------------------> Generate
   |                               |
   +--> Local retrieval -----------+
                                   |
                              Verification
                              /    |     \
                         pass   revise   safe failure
```

- **Triage** classifies a question as `answerable`, `requires_clarification`,
  `requires_escalation`, `out_of_scope`, or `safe_failure`.
- **Retrieval** searches the Markdown knowledge base and eligible resolved cases.
  Current KB documents have priority over cases. `CASE-0914`, marked
  `superseded`, is deliberately excluded from the searchable corpus.
- **Generation** uses `google/flan-t5-small` locally and receives only retrieved
  evidence. A deterministic evidence-grounded generator is available for
  repeatable test/demo runs without downloading models.
- **Verification** validates the JSON Schema, checks source IDs, checks evidence
  anchors, blocks unsafe/unsupported instructions, and rejects superseded
  guidance.
- **Revision** runs at most once. If the revised response still fails,
  `safe_failure` returns a bounded, human-reviewable response. The graph has no
  unbounded loop.
- **Trace logging** records the nodes executed, route-relevant details, retrieval
  backend, and verification/revision events.

## Requirements

- Python 3.10 or newer
- CPU is supported; no GPU is required
- Network access is needed only once to download the pinned models
- The model files must be available locally for the normal agent mode

The supplied assignment data is included in:

- `knowledge_base/`
- `resolved_cases.json`
- `sample_questions.json`
- `output_schema.json`

## Setup

From this directory:

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e .
python -m pip install pytest
```

Download the models once:

```bash
python scripts/download_models.py
```

The download script pins these exact Hugging Face repositories and revisions:

| Purpose | Repository | Revision | Approx. size |
| --- | --- | --- | ---: |
| Retrieval embeddings | `sentence-transformers/all-MiniLM-L6-v2` | `c9745ed` | ~90 MB |
| Local response generation | `google/flan-t5-small` | `f6d0c2c` | ~310 MB |

The application loads from `models/` with `local_files_only=True`. To make an
offline run explicit:

```bash
export HF_HUB_OFFLINE=1
```

On Windows PowerShell:

```powershell
$env:HF_HUB_OFFLINE = "1"
```

No model files are committed to GitHub. The `.gitignore` keeps the local model
weights out of the repository.

## Run the agent

Normal local Hugging Face mode:

```bash
PYTHONPATH=src python -m orbitdesk_agent.cli --pretty \
  "I am a read-only Viewer. Can I create an API credential for a reporting script?"
```

Run a deterministic demo without model weights:

```bash
PYTHONPATH=src python -m orbitdesk_agent.cli --offline-demo --pretty \
  "Our data sync is not working. Can you tell me how to fix it?"
```

The command prints both:

1. `response`: the structured response matching `output_schema.json`
2. `trace`: the executed node sequence and verification details

Exercise the revision path explicitly:

```bash
PYTHONPATH=src python -m orbitdesk_agent.cli \
  --offline-demo --pretty --force-bad-first-attempt \
  "I am a read-only Viewer. Can I create an API credential for a reporting script?"
```

The intentionally bad first draft mentions the removed personal-token workflow.
Verification catches it, and the graph revises the answer once using current
evidence.

## Tests and sample outputs

Run the acceptance tests:

```bash
pytest -q
```

The suite covers:

| Case | Expected route |
| --- | --- |
| Q-001 timezone change and missed export | answerable; multi-document evidence |
| Q-002 Viewer API credential | answerable; roles + credentials |
| Q-003 vague data sync failure | requires clarification |
| Q-004 repeated `render_failed` | requires escalation; revision path exercised |
| Q-005 refund/legal request | out of scope |

Generate the checked-in sample output file:

```bash
PYTHONPATH=src python -m orbitdesk_agent.sample_runner
```

This writes `outputs/sample_outputs.json`. The output includes source excerpts,
confidence, human-review flags, warnings, and execution traces for every sample.

## Local-only and safety design

- `AutoTokenizer.from_pretrained` and model loading use `local_files_only=True`
  in runtime code. Runtime does not call a model hub.
- The model receives retrieved passages as its evidence context; it is not
  allowed to use general product knowledge.
- Current knowledge-base documents are assigned higher retrieval priority than
  resolved cases.
- Superseded cases are not indexed and cannot become answer sources.
- The verifier rejects known unsafe guidance such as the removed personal-token
  flow, requests to paste secrets, and unsupported refund/legal actions.
- Diagnostic responses explicitly prohibit passwords, OAuth tokens, session
  cookies, payment-card numbers, credential secrets, and full customer exports.
- The agent does not claim to change settings, execute exports, issue refunds,
  contact recipients, or recover data.

## Performance notes

The reference run environment for this submission was:

- Python 3.13.11
- Linux x86_64
- 4 CPU cores
- 7.8 GiB RAM visible to the process
- CPU-only PyTorch; no GPU or accelerator

Measured deterministic demo performance in this repository was approximately
0.8–1.0 ms for lexical retrieval and under 10 ms for the full graph because it
does not load a neural generator. Expected CPU-only Hugging Face timings on the
reference hardware are approximate:

- `all-MiniLM-L6-v2` load: 2–8 seconds; query embedding/retrieval: 50–250 ms
- `flan-t5-small` load: 2–10 seconds; response generation: 1–8 seconds
- First normal request is slower because model load happens at startup

Actual timings vary with Python, PyTorch, filesystem cache, and CPU. The trace
reports the measured retrieval timing and revision count for each run.

## Project map

```text
src/orbitdesk_agent/
  agent.py          public agent facade
  graph.py          LangGraph nodes and conditional routing
  types.py          typed state and Pydantic response models
  data.py           KB/case loading and precedence filtering
  retrieval.py      local embedding/lexical retrieval
  generation.py     local HF and deterministic demo generators
  verification.py   schema, citation, evidence, and safety checks
  cli.py            minimal command-line interface
  sample_runner.py  five-case output generation
scripts/
  download_models.py
tests/
  test_agent.py
docs/
  graph_diagram.png
  graph_diagram.mmd
```

## Trade-offs and limitations

`flan-t5-small` is intentionally small and CPU-compatible, so wording quality
will be weaker than a hosted model. Correctness is protected by explicit
retrieval, source references, deterministic safety rules, and verification.
The evidence check is an auditable lexical/identifier check rather than a
perfect semantic entailment proof. With more time, I would add a local
cross-encoder reranker, richer paragraph chunking, stronger local entailment
verification, and a persisted local index.

## Walkthrough recording plan

The assignment requests a 4–7 minute recording. `docs/walkthrough_script.md`
contains a concise script covering the graph, local model/device, three routes,
retrieved evidence, the revision path, a trade-off, and a limitation.

## AI coding assistant disclosure

An AI coding assistant was used to accelerate scaffolding, implementation,
test-writing, and documentation. The resulting code was reviewed and executed
against the supplied assignment materials. The design decisions, safety rules,
local-only constraints, evidence precedence, and acceptance tests are included
in this repository so the implementation can be inspected and explained.