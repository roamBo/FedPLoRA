def _context_text(doc) -> str:
    if "CONTEXTS" in doc:
        ctx = doc["CONTEXTS"]
    elif "context" in doc:
        ctx = doc["context"]
    else:
        if {"_data_files", "_fingerprint", "_split"} <= set(doc.keys()):
            raise KeyError(
                "pubmedqa received dataset metadata instead of an example. "
                "This usually means a stale lm-eval override points dataset_path "
                "to a HuggingFace save_to_disk directory. Regenerate the cache "
                "override with: python scripts/Analysis/prepare_external_lm_eval_hf_cache.py "
                "--tasks pubmedqa --verify_only"
            )
        raise KeyError(f"pubmedqa doc missing context field; keys={sorted(doc.keys())}")
    if isinstance(ctx, str):
        return ctx
    return "\n".join(str(part) for part in ctx)


def _question_text(doc) -> str:
    for key in ("QUESTION", "question"):
        if key in doc:
            return str(doc[key])
    raise KeyError(f"pubmedqa doc missing question field; keys={sorted(doc.keys())}")


def doc_to_text(doc) -> str:
    return "Abstract: {}\nQuestion: {}\nAnswer:".format(
        _context_text(doc),
        _question_text(doc),
    )
