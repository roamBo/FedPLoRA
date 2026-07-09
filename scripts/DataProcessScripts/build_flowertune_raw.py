"""Build pooled FlowerTune-Mixed raw JSONL for FedPLoRA.

This script downloads four public FlowerTune training-side datasets from
Hugging Face and maps them into the FedPLoRA domain benchmark raw schema:

    {"domain": str, "prompt": str, "response": str, "source_id": str, "metadata": dict}

The resulting JSONL can be passed directly to build_domain_benchmark_v2.py.

Example:
    python scripts/DataProcessScripts/build_flowertune_raw.py \
      --output_path data/raw_flowertune_mixed.jsonl \
      --target_per_domain 4000 \
      --seed 42
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List


DATASETS = {
    "general": "flwrlabs/alpaca-gpt4",
    "finance": "flwrlabs/fingpt-sentiment-train",
    "medical": "flwrlabs/medical-meadow-medical-flashcards",
    "code": "flwrlabs/code-alpaca-20k",
}


def _clean_text(value: Any, *, max_chars: int = 0) -> str:
    text = "" if value is None else str(value)
    text = re.sub(r"\s+", " ", text).strip()
    if max_chars and len(text) > max_chars:
        return text[:max_chars].rstrip()
    return text


def _prompt_from_example(example: Dict[str, Any], *, max_chars: int = 0) -> str:
    instruction = _clean_text(example.get("instruction", ""), max_chars=max_chars)
    input_text = _clean_text(example.get("input", ""), max_chars=max_chars)

    if instruction and input_text:
        return f"{instruction}\n\n{input_text}"
    if instruction:
        return instruction
    if input_text:
        return input_text

    # Defensive fallbacks for dataset revisions or viewer-only text fields.
    for key in ("prompt", "question", "text"):
        value = _clean_text(example.get(key, ""), max_chars=max_chars)
        if value:
            return value
    return ""


def _response_from_example(example: Dict[str, Any], *, max_chars: int = 0) -> str:
    for key in ("output", "response", "answer", "label"):
        value = _clean_text(example.get(key, ""), max_chars=max_chars)
        if value:
            return value
    return ""


def _iter_dataset_rows(dataset_id: str, *, cache_dir: str | None):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: datasets. Install with `pip install datasets` "
            "inside the FedRepo2 environment."
        ) from exc

    ds = load_dataset(dataset_id, split="train", cache_dir=cache_dir)
    for item in ds:
        yield dict(item)


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_rows(args) -> List[Dict[str, Any]]:
    rng = random.Random(int(args.seed))
    rows_by_domain: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for domain, dataset_id in DATASETS.items():
        print(f"[download] domain={domain} dataset={dataset_id}")
        seen_source_ids = set()
        for i, example in enumerate(_iter_dataset_rows(dataset_id, cache_dir=args.cache_dir)):
            prompt = _prompt_from_example(example, max_chars=args.max_chars)
            response = _response_from_example(example, max_chars=args.max_chars)
            if not prompt or not response:
                continue
            source_id = f"{domain}_{dataset_id.split('/')[-1]}_{i:08d}"
            if source_id in seen_source_ids:
                continue
            seen_source_ids.add(source_id)
            rows_by_domain[domain].append(
                {
                    "domain": domain,
                    "prompt": prompt,
                    "response": response,
                    "source_id": source_id,
                    "metadata": {
                        "dataset": dataset_id,
                        "split": "train",
                        "row_index": i,
                    },
                }
            )
        print(f"[loaded] domain={domain} rows={len(rows_by_domain[domain])}")

    target = int(args.target_per_domain or 0)
    if target > 0:
        for domain, rows in list(rows_by_domain.items()):
            if len(rows) > target:
                rows_by_domain[domain] = rng.sample(rows, target)
                print(f"[sample] domain={domain} rows={len(rows)} -> {target}")

    all_rows: List[Dict[str, Any]] = []
    for domain in sorted(rows_by_domain):
        all_rows.extend(rows_by_domain[domain])
    rng.shuffle(all_rows)
    return all_rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Download and pool FlowerTune public training datasets.")
    ap.add_argument("--output_path", default="data/raw_flowertune_mixed.jsonl")
    ap.add_argument("--cache_dir", default=None, help="Optional Hugging Face datasets cache dir.")
    ap.add_argument("--target_per_domain", type=int, default=4000, help="Downsample each large domain to this many rows; 0 disables downsampling.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_chars", type=int, default=20000, help="Trim prompt/response strings to this many chars; 0 disables trimming.")
    args = ap.parse_args()

    rows = build_rows(args)
    out = Path(args.output_path)
    _write_jsonl(out, rows)
    counts = Counter(row["domain"] for row in rows)
    print(f"[ok] wrote {len(rows)} rows -> {out}")
    print(f"[ok] domain_counts={dict(sorted(counts.items()))}")


if __name__ == "__main__":
    main()
