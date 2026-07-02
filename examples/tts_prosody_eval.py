#!/usr/bin/env python3
"""Run the TTS prosody optimizer over historical and known-problem samples."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

try:
    from examples.tts_output_validator import validation_records_path
    from examples.tts_prosody_planner import optimize_prosody, prosody_config
    from examples.tts_text_normalizer import normalize_for_tts
except ModuleNotFoundError:
    from tts_output_validator import validation_records_path
    from tts_prosody_planner import optimize_prosody, prosody_config
    from tts_text_normalizer import normalize_for_tts

DEFAULT_OUTPUT_DIR = Path("/media/ivan/55FF-1534/ai-runtimes/faster-qwen3-tts/prosody_eval")
KNOWN_PROBLEM_SAMPLES = [
    (
        "known-user-report-1",
        "好像发了一个 hello，也发了一个 n i h a o，好像报错。感觉还是有不能连贯输出的情况?",
    ),
    (
        "known-user-report-2",
        "这句话的前面部分，一直到好像报错，是一种语气风格，但是后面感觉还是不能连贯输出的情况，它就变成另外一种语气风格。",
    ),
    (
        "known-short-confirmation",
        "好的按照你说的来吧",
    ),
]


def _default_output_path() -> Path:
    ts = time.strftime("%Y%m%d-%H%M%S")
    base = DEFAULT_OUTPUT_DIR if DEFAULT_OUTPUT_DIR.parent.exists() else Path("prosody_eval")
    return base / f"prosody_eval_{ts}.jsonl"


def _iter_history(path: Path, limit: int | None) -> list[tuple[str, str, dict[str, Any]]]:
    rows: list[tuple[str, str, dict[str, Any]]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return rows
    if limit is not None:
        lines = lines[-max(0, limit) :]
    for idx, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        text = str(record.get("expected_text") or "").strip()
        if text:
            sample_id = str(record.get("trace_id") or record.get("validation_id") or f"history-{idx}")
            rows.append((sample_id, text, record))
    return rows


def _dedupe(samples: list[tuple[str, str, dict[str, Any]]]) -> list[tuple[str, str, dict[str, Any]]]:
    seen: set[str] = set()
    result: list[tuple[str, str, dict[str, Any]]] = []
    for sample_id, text, metadata in samples:
        key = text.strip()
        if not key or key in seen:
            continue
        seen.add(key)
        result.append((sample_id, text, metadata))
    return result


def run_eval(history_path: Path, output_path: Path, limit: int | None, include_known: bool) -> int:
    os.environ.setdefault("QWEN_TTS_PROSODY_FAILURE_COOLDOWN_S", "0")
    samples: list[tuple[str, str, dict[str, Any]]] = []
    if include_known:
        samples.extend((sample_id, text, {"source": "known_problem"}) for sample_id, text in KNOWN_PROBLEM_SAMPLES)
    samples.extend(_iter_history(history_path, limit))
    samples = _dedupe(samples)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        for sample_id, text, metadata in samples:
            normalized = normalize_for_tts(text)
            plan = optimize_prosody(normalized.text, trace_id=sample_id)
            row = {
                "sample_id": sample_id,
                "source": metadata.get("source") or "history",
                "input_text": text,
                "normalized_text": normalized.text,
                "optimized_text": plan.text,
                "normalizer": normalized.normalizer,
                "normalization_changed": normalized.changed,
                "prosody": plan.to_trace(),
            }
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "success": True,
                "samples": len(samples),
                "history_path": str(history_path),
                "output_path": str(output_path),
                "prosody_config": prosody_config(),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate local TTS prosody optimization on historical samples.")
    parser.add_argument("--history", default=validation_records_path(), help="Validation JSONL history path.")
    parser.add_argument("--output", default=str(_default_output_path()), help="Output JSONL report path.")
    parser.add_argument("--limit", type=int, default=None, help="Use only the newest N history rows.")
    parser.add_argument("--no-known", action="store_true", help="Do not include fixed known-problem samples.")
    args = parser.parse_args(argv)
    return run_eval(Path(args.history).expanduser(), Path(args.output).expanduser(), args.limit, not args.no_known)


if __name__ == "__main__":
    raise SystemExit(main())
