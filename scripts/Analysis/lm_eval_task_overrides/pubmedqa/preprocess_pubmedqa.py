def _context_text(doc) -> str:
    if "CONTEXTS" in doc:
        ctx = doc["CONTEXTS"]
    elif "context" in doc:
        ctx = doc["context"]
    else:
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
