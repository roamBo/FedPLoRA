import argparse
import glob
import json
import os


DEFAULT_DOMAINS = [
    "general",
    "math",
    "code",
    "medical",
    "legal",
    "finance",
    "education",
]


def _iter_json_or_jsonl(path):
    if path.endswith(".jsonl"):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)
        return

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        for row in data:
            yield row
    else:
        raise ValueError(f"unsupported json format in {path}: expected a list")


def _normalize_row(row, domain, source_name, row_idx):
    prompt = (
        row.get("prompt")
        or row.get("instruction")
        or row.get("question")
        or row.get("input")
        or row.get("query")
        or ""
    )
    response = (
        row.get("response")
        or row.get("output")
        or row.get("answer")
        or row.get("target")
        or row.get("completion")
        or ""
    )

    prompt = str(prompt).strip()
    response = str(response).strip()
    if not prompt or not response:
        return None

    out = {
        "domain": domain,
        "prompt": prompt,
        "response": response,
        "source_id": row.get("source_id") or f"{domain}_{source_name}_{row_idx:08d}",
        "metadata": row.get("metadata") or {},
    }
    out["metadata"]["source_file"] = source_name
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Merge per-domain json/jsonl files into one unified FedPLoRA JSONL."
    )
    parser.add_argument(
        "--input_root",
        type=str,
        default="data/domain_sources",
        help="Root dir. Each domain should be a subdir under this path.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/raw/domain_7_all.jsonl",
        help="Merged output JSONL path.",
    )
    parser.add_argument(
        "--domains",
        type=str,
        default=",".join(DEFAULT_DOMAINS),
        help="Comma-separated domain list.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively search json/jsonl files under each domain directory.",
    )
    args = parser.parse_args()

    domains = [x.strip() for x in args.domains.split(",") if x.strip()]
    rows = []
    stats = {}

    for domain in domains:
        domain_dir = os.path.join(args.input_root, domain)
        pattern = "**/*" if args.recursive else "*"
        candidates = sorted(
            p
            for p in glob.glob(os.path.join(domain_dir, pattern), recursive=args.recursive)
            if p.endswith(".json") or p.endswith(".jsonl")
        )
        kept = 0
        skipped = 0
        for path in candidates:
            source_name = os.path.relpath(path, domain_dir)
            for row_idx, row in enumerate(_iter_json_or_jsonl(path)):
                normalized = _normalize_row(row, domain, source_name, row_idx)
                if normalized is None:
                    skipped += 1
                    continue
                rows.append(normalized)
                kept += 1
        stats[domain] = {
            "files": len(candidates),
            "kept_rows": kept,
            "skipped_rows": skipped,
        }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"[ok] merged rows={len(rows)} -> {args.output}")
    for domain, item in stats.items():
        print(
            f"[domain] {domain}: files={item['files']} "
            f"kept={item['kept_rows']} skipped={item['skipped_rows']}"
        )


if __name__ == "__main__":
    main()
