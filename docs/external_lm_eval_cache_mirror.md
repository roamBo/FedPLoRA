# External lm-eval cache: mirror and offline workflow

This project evaluates exported adapters on official `lm-eval` tasks:

- `mmlu` → `cais/mmlu`
- `pubmedqa` → `bigbio/pubmed_qa`
- `mbpp` → `google-research-datasets/mbpp`

For reproducibility, do not replace these with ModelScope datasets unless the
paper explicitly changes the task definition. If `huggingface.co` is slow or
unreachable, build the same HuggingFace cache through a mirror, then run formal
evaluation offline.

## Recommended collaborator command

```bash
cd /path/to/FedPLoRA
git pull
export PY=/path/to/conda/env/bin/python

bash scripts/RunScripts/prepare_external_lm_eval_cache.sh probe
bash scripts/RunScripts/prepare_external_lm_eval_cache.sh prepare mmlu,pubmedqa,mbpp
bash scripts/RunScripts/prepare_external_lm_eval_cache.sh verify mmlu,pubmedqa,mbpp
```

During `verify`, messages like:

```text
Using the latest cached version of the dataset since ... couldn't be found on the Hugging Face Hub (offline mode is enabled).
```

are expected. They mean the script is intentionally not using the network and
has fallen back to the local cache. Treat the check as successful only when the
last line contains:

```text
[hf-cache][ok] offline cache ready
```

The default `prepare` endpoint is:

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

To force the official endpoint instead:

```bash
bash scripts/RunScripts/prepare_external_lm_eval_cache.sh prepare-official mmlu,pubmedqa,mbpp
```

If both endpoints fail, run the same `prepare` command on another online
machine and copy the finished directory to the server:

```bash
rsync -avP data/external_lm_eval_hf_cache/ \
  user@server:/path/to/FedPLoRA/data/external_lm_eval_hf_cache/
```

Formal external evaluation should then keep using:

```bash
--hf_cache_dir "$CODE_DIR/data/external_lm_eval_hf_cache"
```

The runner sets `HF_HUB_OFFLINE=1`, `HF_DATASETS_OFFLINE=1`, and
`TRANSFORMERS_OFFLINE=1` during verification/evaluation so the final numbers do
not depend on live network access.
