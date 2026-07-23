# Eval-Only: In-Domain and Worst In-Domain Evaluation

This order evaluates the finalized checkpoints without retraining. It computes each
client adapter only on the independent test split of its matched (home) domain, then
reports both the client-weighted In-Domain score and the minimum domain-level score
(Worst In-Domain). The generated files are separate `*_matched_domain.json` artifacts;
the original result JSONs are never overwritten.

## 1. Files to synchronize to the original training node

The following files are the complete eval-only patch:

```text
FedPLoRA-main/tasks/fed_train_sft.py
FedPLoRA-main/scripts/RunScripts/run_eval_only_matched_domain.sh
FedPLoRA-main/scripts/Analysis/summarize_matched_domain_eval.py
```

The commands below must be executed on the node that contains both
`/data/yaominghao/gb/models/trained_models_LW` and the finalized result directories.

## 2. Build the exact main-table run lists

```bash
set -euo pipefail

cd /data/yaominghao/gb/FedPLoRA

RESULT_ROOT=/data/yaominghao/gb/result/FedPLoRA
OUTPUT_ROOT="$RESULT_ROOT/eval_only_worst_indomain_20260723"
RUNNER=FedPLoRA-main/scripts/RunScripts/run_eval_only_matched_domain.sh
SUMMARIZER=FedPLoRA-main/scripts/Analysis/summarize_matched_domain_eval.py

mkdir -p "$OUTPUT_ROOT/d1" "$OUTPUT_ROOT/flowertune"

D1_METHOD_DIRS=(
  OS1_normal
  OS1_ffa
  OS1_flora
  OS1_flexlora
  OS1_ecolora
  OS1_fedsa_lora
  OS1_feddat
  OS1_yoco
  OS1_fedalt
  OS1_hydralora
  OS1_hilora
  OS1_fedlease
)

D1_RESULTS=()
for seed in 42 43 44; do
  base="$RESULT_ROOT/os_20260709_baseline_35c_dir05_r1_finaleval_seed${seed}/result_logs"
  for method_dir in "${D1_METHOD_DIRS[@]}"; do
    mapfile -t hits < <(find "$base/$method_dir" -maxdepth 1 -type f -name '*.json' | sort)
    if [[ "${#hits[@]}" -ne 1 ]]; then
      echo "Expected one result JSON, found ${#hits[@]}: $base/$method_dir" >&2
      exit 1
    fi
    D1_RESULTS+=("${hits[0]}")
  done
done

D1_OURS_DIRS=(
  "$RESULT_ROOT/v13_20260712_nx0_35c_dir05_r1_finaleval_seed42/result_logs/NX0_v13a_os_split42_train42"
  "$RESULT_ROOT/v13_20260711_nx1_35c_dir05_r1_finaleval_seed43/result_logs/NX1_v13a_os_split43_train43"
  "$RESULT_ROOT/v13_20260711_nx1_35c_dir05_r1_finaleval_seed44/result_logs/NX1_v13a_os_split44_train44"
)
for result_dir in "${D1_OURS_DIRS[@]}"; do
  mapfile -t hits < <(find "$result_dir" -maxdepth 1 -type f -name '*.json' | sort)
  if [[ "${#hits[@]}" -ne 1 ]]; then
    echo "Expected one result JSON, found ${#hits[@]}: $result_dir" >&2
    exit 1
  fi
  D1_RESULTS+=("${hits[0]}")
done

if [[ "${#D1_RESULTS[@]}" -ne 39 ]]; then
  echo "D1 list must contain 39 runs, found ${#D1_RESULTS[@]}" >&2
  exit 1
fi

FLOWER_METHOD_DIRS=(
  N9_flower_normal
  N9_flower_ecolora
  N9_flower_fedsa_lora
  N9_flower_fedalt
  N9_flower_hydralora
  N9_flower_fedlease
  N7_ours_flower_v13a
)

FLOWER_RESULTS=()
for seed in 42 43 44; do
  base="$RESULT_ROOT/order_0715/flowertune_20260715_core8_seed${seed}/result_logs"
  for method_dir in "${FLOWER_METHOD_DIRS[@]}"; do
    mapfile -t hits < <(find "$base/$method_dir" -maxdepth 1 -type f -name '*.json' | sort)
    if [[ "${#hits[@]}" -ne 1 ]]; then
      echo "Expected one result JSON, found ${#hits[@]}: $base/$method_dir" >&2
      exit 1
    fi
    FLOWER_RESULTS+=("${hits[0]}")
  done
done

if [[ "${#FLOWER_RESULTS[@]}" -ne 21 ]]; then
  echo "FlowerTune list must contain 21 runs, found ${#FLOWER_RESULTS[@]}" >&2
  exit 1
fi

printf '%s\n' "${D1_RESULTS[@]}" > "$OUTPUT_ROOT/d1_source_results.txt"
printf '%s\n' "${FLOWER_RESULTS[@]}" > "$OUTPUT_ROOT/flowertune_source_results.txt"
```

`FedP-OneShot` is intentionally excluded. The expected workload is 39 D1 runs
(12 baselines + ours, each with three seeds) and 21 FlowerTune-Mixed runs
(6 baselines + ours, each with three seeds), for 60 eval-only jobs in total.

## 3. Launch matched-domain eval-only on two GPUs

`EVAL_MAX_BATCHES=0` means evaluating the complete independent domain-test split.
No local training or server aggregation is executed.

```bash
nohup env \
  CUDA_VISIBLE_DEVICES=0 \
  EVAL_MAX_BATCHES=0 \
  MATCHED_DOMAIN_OUTPUT_ROOT="$OUTPUT_ROOT/d1" \
  bash "$RUNNER" "${D1_RESULTS[@]}" \
  > "$OUTPUT_ROOT/d1_eval.log" 2>&1 &
D1_PID=$!
echo "$D1_PID" > "$OUTPUT_ROOT/d1_eval.pid"

nohup env \
  CUDA_VISIBLE_DEVICES=1 \
  EVAL_MAX_BATCHES=0 \
  MATCHED_DOMAIN_OUTPUT_ROOT="$OUTPUT_ROOT/flowertune" \
  bash "$RUNNER" "${FLOWER_RESULTS[@]}" \
  > "$OUTPUT_ROOT/flowertune_eval.log" 2>&1 &
FLOWER_PID=$!
echo "$FLOWER_PID" > "$OUTPUT_ROOT/flowertune_eval.pid"

echo "D1 PID: $D1_PID"
echo "FlowerTune PID: $FLOWER_PID"
```

## 4. Verify completion and summarize three seeds

```bash
wait "$(cat "$OUTPUT_ROOT/d1_eval.pid")"
wait "$(cat "$OUTPUT_ROOT/flowertune_eval.pid")"

find "$OUTPUT_ROOT/d1" -type f -name '*_matched_domain.json' | sort \
  > "$OUTPUT_ROOT/d1_eval_results.txt"
find "$OUTPUT_ROOT/flowertune" -type f -name '*_matched_domain.json' | sort \
  > "$OUTPUT_ROOT/flowertune_eval_results.txt"

[[ "$(wc -l < "$OUTPUT_ROOT/d1_eval_results.txt")" -eq 39 ]]
[[ "$(wc -l < "$OUTPUT_ROOT/flowertune_eval_results.txt")" -eq 21 ]]

python "$SUMMARIZER" "$OUTPUT_ROOT/d1" \
  | tee "$OUTPUT_ROOT/d1_summary.tsv"
python "$SUMMARIZER" "$OUTPUT_ROOT/flowertune" \
  | tee "$OUTPUT_ROOT/flowertune_summary.tsv"
```

The two values needed for the main table are:

```text
in_domain_domain_test_token_accuracy
in_domain_domain_test_worst_token_accuracy
```

The first is the unweighted client-macro matched-domain accuracy. The second is the minimum
accuracy across matched domains after pooling all home-domain clients within each
domain; it is the proposed `Worst In-Dom.` column.
