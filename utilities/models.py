import torch
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    BitsAndBytesConfig,
)
from datasets import load_dataset
from tqdm import tqdm
import numpy as np
from peft import (
    get_peft_model,
    AdaLoraModel,
    AdaLoraConfig,
    TaskType,
    LoraConfig,
    prepare_model_for_kbit_training,
)
from utilities.data_utils import *
import argparse
from copy import deepcopy
import math


def _resolve_torch_dtype(args):
    dtype_name = getattr(args, "torch_dtype", "auto")
    if dtype_name == "bfloat16":
        return torch.bfloat16
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "float32":
        return torch.float32
    return torch.bfloat16 if torch.cuda.is_available() else torch.float32


def _resolve_target_modules(args, default_modules):
    target_modules = getattr(args, "target_modules", default_modules)
    if isinstance(target_modules, str):
        target_modules = [x.strip() for x in target_modules.split(",") if x.strip()]
    return target_modules or default_modules


def create_peft_causal_lm_model(args):
    dtype = _resolve_torch_dtype(args)
    attn = (getattr(args, "attn_implementation", None) or "").strip()
    extra_kw = {}
    if attn:
        extra_kw["attn_implementation"] = attn
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        device_map=None,
        trust_remote_code=getattr(args, "trust_remote_code", False),
        **extra_kw,
    )
    if getattr(args, "gradient_checkpointing", False):
        model.gradient_checkpointing_enable()
        if hasattr(model, "config"):
            model.config.use_cache = False

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        use_rslora=getattr(args, "rslora", False),
        target_modules=_resolve_target_modules(
            args,
            [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "up_proj",
                "down_proj",
                "gate_proj",
            ],
        ),
    )
    model = get_peft_model(model, lora_config)
    return model


def create_peft_causal_lm_ffa_model(args):
    model = create_peft_causal_lm_model(args)
    for name, param in model.named_parameters():
        if "lora_A" in name:
            param.requires_grad = False
    return model


def create_peft_model(num_labels, args):

    model = RobertaForSequenceClassification.from_pretrained(
        args.model, num_labels=num_labels
    )

    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        use_rslora=args.rslora,
        target_modules=["query", "value"],
    )

    model = get_peft_model(model, peft_config)

    return model


def init_fedplora_adapters(model):
    """
    FedP-LoRA adapter init (matches PEFT LoRA defaults): Kaiming-uniform lora_A,
    zeros for lora_B.

    If **both** A and B are all-zero, the LoRA branch output is always zero and
    gradients w.r.t. A and B in the product B·A vanish at the start, so AdamW
    would not update LoRA—only the classifier head would move. Training both A
    and B locally requires this (or any) non-degenerate init.
    """
    for name, param in model.named_parameters():
        if "lora_A" in name and name.endswith("weight"):
            torch.nn.init.kaiming_uniform_(param, a=math.sqrt(5))
        elif "lora_B" in name and name.endswith("weight"):
            with torch.no_grad():
                param.zero_()


def create_peft_FFA_model(num_labels, args):

    model = RobertaForSequenceClassification.from_pretrained(
        args.model, num_labels=num_labels
    )

    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        use_rslora=args.rslora,
        target_modules=["query", "value"],
    )
    model = get_peft_model(model, peft_config)

    # Make LoRA A matrices non-trainable
    for name, param in model.named_parameters():
        if "lora_A" in name:
            param.requires_grad = False

    return model


def create_peft_gpt2_model_e2e(args):
    model = GPT2LMHeadModel.from_pretrained("gpt2")

    # Define LoRA configuration for language modeling task
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,  # For language modeling
        inference_mode=False,
        r=args.lora_r,  # The dimension of the low-rank update matrices
        lora_alpha=args.lora_alpha,  # The scaling factor for LoRA layers
        lora_dropout=args.lora_dropout,  # Dropout to apply to LoRA layers
        target_modules=["c_attn", "c_proj"],  # Modules to apply LoRA
    )

    # Apply LoRA to the GPT-2 model
    model = get_peft_model(model, lora_config)
    return model


def create_peft_gpt2_model_e2e_ffa(args):
    model = GPT2LMHeadModel.from_pretrained("gpt2")

    # Define LoRA configuration for language modeling task
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,  # For language modeling
        inference_mode=False,
        r=args.lora_r,  # The dimension of the low-rank update matrices
        lora_alpha=args.lora_alpha,  # The scaling factor for LoRA layers
        lora_dropout=args.lora_dropout,  # Dropout to apply to LoRA layers
        target_modules=["c_attn", "c_proj"],  # Modules to apply LoRA
    )

    # Apply LoRA to the GPT-2 model
    model = get_peft_model(model, lora_config)
    for name, param in model.named_parameters():
        if "lora_A" in name:
            param.requires_grad = False
    return model
