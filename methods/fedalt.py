"""
FedALT (AAAI 2026): RoW LoRA + local LoRA + mixer (paper).
Conventional baseline in this repo: same communication as FedSA (aggregate A, local B);
input-dependent mixer / leave-one-out RoW can extend the custom forward later.
"""

from methods.fedsa_lora import aggregate_models_fedsa_lora


def aggregate_models_fedalt(global_model, client_uploads, args):
    return aggregate_models_fedsa_lora(global_model, client_uploads, args)
