"""
FDLoRA: dual global/local LoRA with adaptive fusion (arXiv 2024).
Conventional baseline here: FedAvg on all LoRA + heads (single shared adapter = global path only).
"""

from methods.fedavg_normal import aggregate_models_normal


def aggregate_models_fdlora(global_model, client_models, args):
    return aggregate_models_normal(global_model, client_models)
