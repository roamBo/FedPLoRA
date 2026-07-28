# code_eval fallback

This directory intentionally provides a repo-local fallback for
`evaluate.load("code_eval")`, which is imported by lm-evaluation-harness' MBPP
task.  Offline servers may not have the HuggingFace Evaluate metric cached, so
formal MBPP evaluation would otherwise fail before model inference.

The metric executes generated Python code.  Only run MBPP in an isolated
environment and only with `HF_ALLOW_CODE_EVAL=1`.
