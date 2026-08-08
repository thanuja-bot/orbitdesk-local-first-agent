from __future__ import annotations

import argparse
import json
import sys

from .agent import OrbitDeskAgent
from .config import AgentConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the local-only OrbitDesk support agent."
    )
    parser.add_argument("question", nargs="*", help="Support question to answer")
    parser.add_argument(
        "--offline-demo",
        action="store_true",
        help="Use deterministic evidence-grounded generation for demos/tests; no model download.",
    )
    parser.add_argument(
        "--force-bad-first-attempt",
        action="store_true",
        help="Exercise the verification -> revision path.",
    )
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    question = " ".join(args.question).strip()
    if not question:
        question = input("OrbitDesk question: ").strip()
    try:
        agent = OrbitDeskAgent(AgentConfig(), offline_demo=args.offline_demo)
        result = agent.answer(
            question,
            force_bad_first_attempt=args.force_bad_first_attempt,
        )
    except Exception as exc:
        print(f"Agent could not start: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2 if args.pretty else None, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())