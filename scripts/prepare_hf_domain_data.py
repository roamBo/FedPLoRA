import argparse
import json
import os

from datasets import load_dataset


AUTO_PROMPT_FIELDS = [
    "prompt",
    "instruction",
    "question",
    "problem",
    "query",
    "input",
    "context",
]
AUTO_RESPONSE_FIELDS = [
    "response",
    "output",
    "answer",
    "solution",
    "target",
    "completion",
    "label",
]
AUTO_INPUT_FIELDS = ["input", "context", "details"]
AUTO_MESSAGES_FIELDS = [
    "messages",
    "conversations",
    "conversation",
    "dialog",
    "dialogue",
    "chat",
    "chosen",
]
ROLE_KEYS = ["role", "from", "speaker", "author"]
CONTENT_KEYS = ["content", "value", "text", "utterance", "message"]
ASSISTANT_ROLES = {"assistant", "gpt", "bot", "model", "chatgpt", "assistant_1"}


def _sanitize_text(value):
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float, bool)):
        return str(value).strip()
    return json.dumps(value, ensure_ascii=False).strip()


def _combine_instruction_input(prompt, extra_input):
    prompt = _sanitize_text(prompt)
    extra_input = _sanitize_text(extra_input)
    if prompt and extra_input:
        return f"{prompt}\n\n{extra_input}".strip()
    return prompt or extra_input


def _find_first_present(row, candidates):
    for key in candidates:
        if key in row and row[key] not in (None, "", [], {}):
            return row[key], key
    return None, None


def _extract_messages(row, messages_field=None):
    field_candidates = [messages_field] if messages_field else AUTO_MESSAGES_FIELDS
    for field in field_candidates:
        if field and field in row and isinstance(row[field], list):
            return row[field], field
    return None, None


def _normalize_role(message):
    for key in ROLE_KEYS:
        if key in message and message[key] is not None:
            return _sanitize_text(message[key]).lower()
    return ""


def _normalize_content(message):
    for key in CONTENT_KEYS:
        if key in message and message[key] not in (None, ""):
            return _sanitize_text(message[key])
    return ""


def _format_chat_prompt(messages):
    turns = []
    for role, content in messages:
        if not content:
            continue
        role_tag = role or "user"
        turns.append(f"{role_tag.capitalize()}: {content}")
    return "\n".join(turns).strip()


def _normalize_from_messages(messages):
    parsed = []
    for message in messages:
        if isinstance(message, str):
            content = _sanitize_text(message)
            if content:
                parsed.append(("", content))
            continue
        if not isinstance(message, dict):
            continue
        role = _normalize_role(message)
        content = _normalize_content(message)
        if content:
            parsed.append((role, content))

    if len(parsed) < 2:
        return None, None

    assistant_indices = [
        idx for idx, (role, content) in enumerate(parsed) if role in ASSISTANT_ROLES and content
    ]
    if assistant_indices:
        last_response_idx = assistant_indices[-1]
    else:
        last_response_idx = len(parsed) - 1

    response = parsed[last_response_idx][1]
    prompt = _format_chat_prompt(parsed[:last_response_idx])
    return prompt, response


def _render_template(template, row):
    if not template:
        return ""
    safe_row = {k: _sanitize_text(v) for k, v in row.items()}
    return template.format_map(safe_row).strip()


def normalize_row(row, args, row_idx):
    source_id = row.get("source_id") or f"{args.domain}_{args.source_name}_{row_idx:08d}"

    if args.prompt_template and args.response_template:
        prompt = _render_template(args.prompt_template, row)
        response = _render_template(args.response_template, row)
        if prompt and response:
            return {
                "domain": args.domain,
                "prompt": prompt,
                "response": response,
                "source_id": source_id,
                "metadata": {"dataset": args.dataset, "config": args.config_name or ""},
            }

    if args.messages_field:
        messages = row.get(args.messages_field)
        if isinstance(messages, list):
            prompt, response = _normalize_from_messages(messages)
            if prompt and response:
                return {
                    "domain": args.domain,
                    "prompt": prompt,
                    "response": response,
                    "source_id": source_id,
                    "metadata": {"dataset": args.dataset, "config": args.config_name or ""},
                }

    if args.prompt_field and args.response_field:
        prompt = row.get(args.prompt_field)
        response = row.get(args.response_field)
        if args.input_field:
            prompt = _combine_instruction_input(prompt, row.get(args.input_field))
        else:
            prompt = _sanitize_text(prompt)
        response = _sanitize_text(response)
        if prompt and response:
            return {
                "domain": args.domain,
                "prompt": prompt,
                "response": response,
                "source_id": source_id,
                "metadata": {"dataset": args.dataset, "config": args.config_name or ""},
            }

    prompt_value, _ = _find_first_present(row, AUTO_PROMPT_FIELDS)
    response_value, _ = _find_first_present(row, AUTO_RESPONSE_FIELDS)
    extra_input, _ = _find_first_present(row, AUTO_INPUT_FIELDS)
    prompt = _combine_instruction_input(prompt_value, extra_input)
    response = _sanitize_text(response_value)
    if prompt and response:
        return {
            "domain": args.domain,
            "prompt": prompt,
            "response": response,
            "source_id": source_id,
            "metadata": {"dataset": args.dataset, "config": args.config_name or ""},
        }

    messages, _ = _extract_messages(row, args.messages_field)
    if messages is not None:
        prompt, response = _normalize_from_messages(messages)
        if prompt and response:
            return {
                "domain": args.domain,
                "prompt": prompt,
                "response": response,
                "source_id": source_id,
                "metadata": {"dataset": args.dataset, "config": args.config_name or ""},
            }

    return None


def build_parser(default_domain=None, default_dataset=None, description=None):
    parser = argparse.ArgumentParser(
        description=description or "Prepare a Hugging Face dataset into FedPLoRA domain JSONL."
    )
    parser.add_argument("--dataset", type=str, default=default_dataset or "", help="HF dataset name or local dataset path.")
    parser.add_argument("--config_name", type=str, default="", help="HF dataset config/subset name.")
    parser.add_argument("--splits", type=str, default="train", help="Comma-separated splits, e.g. train,validation.")
    parser.add_argument("--domain", type=str, default=default_domain or "", help="Target domain name.")
    parser.add_argument("--output", type=str, default="", help="Output jsonl path.")
    parser.add_argument("--max_samples", type=int, default=0, help="Optional cap on exported examples.")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle each split before export.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prompt_field", type=str, default="", help="Explicit prompt field name.")
    parser.add_argument("--response_field", type=str, default="", help="Explicit response field name.")
    parser.add_argument("--input_field", type=str, default="", help="Optional extra input/context field appended to prompt.")
    parser.add_argument("--messages_field", type=str, default="", help="Explicit messages/conversations field name.")
    parser.add_argument("--prompt_template", type=str, default="", help="Python format template for prompt, e.g. '{instruction}\\n\\n{input}'.")
    parser.add_argument("--response_template", type=str, default="", help="Python format template for response.")
    parser.add_argument("--source_name", type=str, default="", help="Source tag used in source_id.")
    return parser


def run(args):
    if not args.dataset:
        raise ValueError("--dataset is required")
    if not args.domain:
        raise ValueError("--domain is required")
    if not args.output:
        default_name = args.source_name or args.dataset.split("/")[-1].replace("-", "_")
        args.output = os.path.join("data", "domain_sources", args.domain, f"{default_name}.jsonl")

    dataset_name = args.dataset
    config_name = args.config_name or None
    split_names = [x.strip() for x in args.splits.split(",") if x.strip()]
    args.source_name = args.source_name or dataset_name.split("/")[-1].replace("-", "_")

    rows = []
    skipped = 0
    exported = 0

    for split_name in split_names:
        ds = load_dataset(dataset_name, config_name, split=split_name)
        if args.shuffle:
            ds = ds.shuffle(seed=args.seed)
        for row_idx, row in enumerate(ds):
            normalized = normalize_row(row, args, row_idx)
            if normalized is None:
                skipped += 1
                continue
            normalized["metadata"]["split"] = split_name
            rows.append(normalized)
            exported += 1
            if args.max_samples and exported >= args.max_samples:
                break
        if args.max_samples and exported >= args.max_samples:
            break

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"[ok] dataset={args.dataset} domain={args.domain} output={args.output}")
    print(f"[ok] exported={exported} skipped={skipped} splits={split_names}")


def main(default_domain=None, default_dataset=None, description=None):
    parser = build_parser(default_domain, default_dataset, description)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
