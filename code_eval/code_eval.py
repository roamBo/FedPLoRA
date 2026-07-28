"""Repo-local fallback for ``evaluate.load("code_eval")``.

lm-evaluation-harness' MBPP task imports HuggingFace Evaluate's ``code_eval``
metric at task-load time.  On offline servers this metric is often missing from
the Evaluate cache, causing MBPP to fail before any model inference.  Evaluate
first checks for a local ``code_eval/code_eval.py`` under the current working
directory, so this file provides a small compatible pass@k implementation.

The metric executes generated Python code.  Only use it in the same isolated
environment where MBPP is allowed to run.
"""

from __future__ import annotations

import itertools
import multiprocessing as mp
import os
import queue
import traceback
from collections import Counter, defaultdict

import datasets
import evaluate
import numpy as np


_DESCRIPTION = "Compute pass@k for Python code-generation tasks."
_KWARGS_DESCRIPTION = """
Args:
    predictions: list of candidate-code lists.
    references: list of Python assertion strings.
    k: list of k values for pass@k.
    num_workers: accepted for API compatibility.
    timeout: per-candidate execution timeout in seconds.
"""
_WARNING = """
The code_eval metric executes untrusted model-generated Python code.  Set
HF_ALLOW_CODE_EVAL=1 only in an isolated/sandboxed environment.
"""


def _estimate_pass_at_k(num_samples, num_correct, k):
    def estimator(n: int, c: int, kk: int) -> float:
        if n - c < kk:
            return 1.0
        return 1.0 - np.prod(1.0 - kk / np.arange(n - c + 1, n + 1))

    if isinstance(num_samples, int):
        num_samples_it = itertools.repeat(num_samples, len(num_correct))
    else:
        assert len(num_samples) == len(num_correct)
        num_samples_it = iter(num_samples)
    return np.array(
        [estimator(int(n), int(c), int(k)) for n, c in zip(num_samples_it, num_correct)]
    )


def _execute_program(program: str, result_queue: mp.Queue) -> None:
    try:
        namespace = {"__name__": "__main__"}
        exec(program, namespace)  # noqa: S102 - explicitly required by MBPP/code_eval
    except BaseException as exc:  # noqa: BLE001 - report any candidate failure
        result_queue.put(
            {
                "passed": False,
                "result": repr(exc),
                "traceback": traceback.format_exc(limit=8),
            }
        )
        return
    result_queue.put({"passed": True, "result": "passed", "traceback": ""})


def _check_correctness(program: str, timeout: float, task_id: int, completion_id: int) -> dict:
    result_queue: mp.Queue = mp.Queue(maxsize=1)
    proc = mp.Process(target=_execute_program, args=(program, result_queue))
    proc.start()
    proc.join(timeout=float(timeout))
    if proc.is_alive():
        proc.terminate()
        proc.join(timeout=1.0)
        if proc.is_alive():
            proc.kill()
            proc.join(timeout=1.0)
        outcome = {"passed": False, "result": "timed out", "traceback": ""}
    else:
        try:
            outcome = result_queue.get_nowait()
        except queue.Empty:
            outcome = {
                "passed": proc.exitcode == 0,
                "result": f"exitcode={proc.exitcode}",
                "traceback": "",
            }
    return {
        "task_id": int(task_id),
        "completion_id": int(completion_id),
        **outcome,
    }


@evaluate.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class CodeEval(evaluate.Metric):
    def _info(self):
        return evaluate.MetricInfo(
            description=_DESCRIPTION,
            citation="",
            inputs_description=_KWARGS_DESCRIPTION,
            features=datasets.Features(
                {
                    "predictions": datasets.Sequence(datasets.Value("string")),
                    "references": datasets.Value("string"),
                }
            ),
            homepage="https://github.com/huggingface/evaluate/tree/main/metrics/code_eval",
            codebase_urls=[
                "https://github.com/huggingface/evaluate/tree/main/metrics/code_eval"
            ],
            reference_urls=["https://arxiv.org/abs/2107.03374"],
        )

    def _compute(self, predictions, references, k=None, num_workers=4, timeout=3.0):
        if os.getenv("HF_ALLOW_CODE_EVAL", "0") != "1":
            raise ValueError(_WARNING)
        if os.name == "nt":
            raise NotImplementedError("code_eval is not supported on Windows.")
        if k is None:
            k = [1, 10, 100]

        completion_id = Counter()
        results = defaultdict(list)
        for task_id, (candidates, test_case) in enumerate(zip(predictions, references)):
            if isinstance(candidates, str):
                candidates = [candidates]
            for candidate in candidates:
                program = str(candidate) + "\n" + str(test_case)
                result = _check_correctness(
                    program,
                    timeout=float(timeout),
                    task_id=int(task_id),
                    completion_id=int(completion_id[task_id]),
                )
                results[int(task_id)].append((int(completion_id[task_id]), result))
                completion_id[task_id] += 1

        total, correct = [], []
        for task_results in results.values():
            task_results.sort(key=lambda item: item[0])
            passed = [bool(row[1]["passed"]) for row in task_results]
            total.append(len(passed))
            correct.append(sum(passed))
        total_arr = np.array(total)
        correct_arr = np.array(correct)
        pass_at_k = {
            f"pass@{kk}": float(_estimate_pass_at_k(total_arr, correct_arr, int(kk)).mean())
            for kk in k
            if len(total_arr) and (total_arr >= int(kk)).all()
        }
        return pass_at_k, dict(results)
