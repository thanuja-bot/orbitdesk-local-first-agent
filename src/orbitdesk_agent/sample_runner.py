from __future__ import annotations

import json
from pathlib import Path

from .agent import OrbitDeskAgent
from .config import AgentConfig


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    questions = json.loads((root / "sample_questions.json").read_text(encoding="utf-8"))["questions"]
    output_dir = root / "outputs"
    output_dir.mkdir(exist_ok=True)
    agent = OrbitDeskAgent(AgentConfig(data_dir=root), offline_demo=True)
    results = []
    for item in questions:
        result = agent.answer(
            item["question"],
            force_bad_first_attempt=item["question_id"] == "Q-004",
        )
        results.append({"question_id": item["question_id"], "question": item["question"], **result})
    target = output_dir / "sample_outputs.json"
    target.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {target}")
    for item in results:
        print(f"{item['question_id']}: {item['response']['classification']} ({len(item['trace'])} trace events)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())