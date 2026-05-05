import torch
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader, Subset
from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel,
    get_linear_schedule_with_warmup,
)
from tqdm import tqdm
import numpy as np
import pandas as pd
from peft import get_peft_model, LoraConfig, TaskType
import os
import json
from collections import defaultdict


def load_and_preprocess_data(task):

    if "mnli" in task:
        dataset = load_dataset("glue", "mnli")
    else:
        dataset = load_dataset("glue", task)

    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

    def tokenize_function(examples):

        # Handle different input formats
        if "premise" in examples and "hypothesis" in examples:
            # MNLI and similar tasks
            return tokenizer(
                examples["premise"],
                examples["hypothesis"],
                truncation=True,
                padding="max_length",
                max_length=128,
            )
        elif "question" in examples and "sentence" in examples:
            # QNLI and similar tasks
            return tokenizer(
                examples["question"],
                examples["sentence"],
                truncation=True,
                padding="max_length",
                max_length=128,
            )
        elif "sentence1" in examples and "sentence2" in examples:
            # MRPC, STS-B
            return tokenizer(
                examples["sentence1"],
                examples["sentence2"],
                truncation=True,
                padding="max_length",
                max_length=128,
            )
        elif "question1" in examples and "question2" in examples:
            # QQP
            return tokenizer(
                examples["question1"],
                examples["question2"],
                truncation=True,
                padding="max_length",
                max_length=128,
            )
        elif "sentence" in examples:
            # CoLA, SST-2
            return tokenizer(
                examples["sentence"],
                truncation=True,
                padding="max_length",
                max_length=128,
            )
        else:
            raise ValueError(f"Unexpected format for task {task}")

    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    if task == "cola":
        tokenized_datasets = tokenized_datasets.remove_columns(["sentence", "idx"])
    elif task == "sst2":
        tokenized_datasets = tokenized_datasets.remove_columns(["sentence", "idx"])
    elif task == "mrpc":
        tokenized_datasets = tokenized_datasets.remove_columns(
            ["sentence1", "sentence2", "idx"]
        )
    elif task == "qqp":
        tokenized_datasets = tokenized_datasets.remove_columns(
            ["question1", "question2", "idx"]
        )
    elif task == "stsb":
        tokenized_datasets = tokenized_datasets.remove_columns(
            ["sentence1", "sentence2", "idx"]
        )
    elif task == "qnli":
        tokenized_datasets = tokenized_datasets.remove_columns(
            ["question", "sentence", "idx"]
        )
    elif task == "rte":
        tokenized_datasets = tokenized_datasets.remove_columns(
            ["sentence1", "sentence2", "idx"]
        )
    elif task == "wnli":
        tokenized_datasets = tokenized_datasets.remove_columns(
            ["sentence1", "sentence2", "idx"]
        )
    elif task == "mnli_matched" or task == "mnli_mismatched" or task == "mnli":
        tokenized_datasets = tokenized_datasets.remove_columns(
            ["premise", "hypothesis", "idx"]
        )
    else:
        raise ValueError(f"Unexpected task {task}")

    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    tokenized_datasets.set_format("torch")

    if (
        task == "cola"
        or task == "sst2"
        or task == "mrpc"
        or task == "qqp"
        or task == "stsb"
        or task == "qnli"
        or task == "rte"
        or task == "wnli"
    ):
        train_dataset = tokenized_datasets["train"]
        val_dataset = tokenized_datasets["validation"]
        test_dataset = tokenized_datasets["test"]
    elif task == "mnli_matched":
        train_dataset = tokenized_datasets["train"]
        val_dataset = tokenized_datasets["validation_matched"]
        test_dataset = tokenized_datasets["test_matched"]
    elif task == "mnli_mismatched":
        train_dataset = tokenized_datasets["train"]
        val_dataset = tokenized_datasets["validation_mismatched"]
        test_dataset = tokenized_datasets["test_mismatched"]

    return train_dataset, val_dataset, test_dataset


def create_dataloader(dataset, args):
    return DataLoader(dataset, batch_size=args.batch_size, shuffle=False)


def dirichlet_label_skew_indices(labels, num_clients, alpha, rng):
    """
    Per-class Dirichlet partition: for each class, split indices across clients
    with proportions ~ Dir(alpha * 1_K). Smaller alpha => stronger label skew.

    Important: naive Dirichlet can create "single-class clients" (or extremely tiny
    clients). For binary tasks like CoLA this often causes degenerate training:
    models predict a single class -> ACC flat, MCC ~ 0 for many rounds, across
    *all* methods. To prevent this, we resample until basic constraints hold.
    """
    labels = np.asarray(labels)
    if labels.ndim > 1:
        labels = labels.reshape(-1)
    labels = labels.astype(np.int64, copy=False)

    classes = np.unique(labels)
    n_classes = int(classes.size)

    # Hard-coded safety constraints (no extra CLI args).
    # - ensure each client has at least MIN_SIZE samples
    # - ensure each client sees at least MIN_CLASSES unique labels (when possible)
    # - ensure at least MIN_PER_CLASS samples per seen class (small, but avoids zero)
    MIN_SIZE = 16
    MIN_CLASSES = 2
    MIN_PER_CLASS = 1
    MAX_TRIES = 200

    def _sample_once():
        client_indices = [[] for _ in range(num_clients)]
        for c in classes:
            idx_c = np.where(labels == c)[0]
            rng.shuffle(idx_c)
            if len(idx_c) == 0:
                continue
            proportions = rng.dirichlet(np.repeat(alpha, num_clients))
            counts = rng.multinomial(len(idx_c), proportions)
            start = 0
            for j in range(num_clients):
                end = start + int(counts[j])
                client_indices[j].extend(idx_c[start:end].tolist())
                start = end
        _redistribute_empty_clients(client_indices)
        return client_indices

    def _meets_constraints(client_indices):
        # total size constraint
        sizes = [len(x) for x in client_indices]
        if min(sizes) < MIN_SIZE:
            return False

        # For regression (no discrete labels) this function isn't used.
        # For classification: if we have >=2 classes overall, avoid single-class clients.
        if n_classes >= 2:
            for idxs in client_indices:
                sub = labels[np.asarray(idxs, dtype=np.int64)]
                uniq, cnts = np.unique(sub, return_counts=True)
                if uniq.size < min(MIN_CLASSES, n_classes):
                    return False
                if int(cnts.min()) < MIN_PER_CLASS:
                    return False
        return True

    last = None
    for _ in range(MAX_TRIES):
        cand = _sample_once()
        last = cand
        if _meets_constraints(cand):
            return cand

    # If constraints are impossible (too few samples or too extreme alpha),
    # fall back to the best-effort partition (still non-empty clients).
    return last


def _redistribute_empty_clients(client_indices):
    """Ensure no client has zero samples when total samples >= num_clients."""
    n_clients = len(client_indices)
    for _ in range(n_clients * n_clients):
        empty = [i for i, c in enumerate(client_indices) if len(c) == 0]
        if not empty:
            return
        j = max(range(n_clients), key=lambda k: len(client_indices[k]))
        if len(client_indices[j]) <= 1:
            return
        for i in empty:
            client_indices[i].append(client_indices[j].pop())


def print_client_label_distribution(labels, client_indices):
    """Print per-client class counts for inspecting non-IID strength."""
    labels = np.asarray(labels).reshape(-1).astype(np.int64, copy=False)
    classes = np.unique(labels)
    print("Client / label distribution (row=client, col=class count):")
    header = "client\t" + "\t".join(f"c{int(c)}" for c in classes)
    print(header)
    for i, idxs in enumerate(client_indices):
        if len(idxs) == 0:
            row = [0] * len(classes)
        else:
            sub = labels[np.array(idxs, dtype=np.int64)]
            row = [int((sub == c).sum()) for c in classes]
        print(f"{i}\t" + "\t".join(str(x) for x in row))
        print(f"  (n={len(idxs)})")


def create_client_dataloaders_nlg(dataset, args):
    client_data = [[] for _ in range(args.num_clients)]
    for data in dataset:
        client_idx = np.random.randint(args.num_clients)
        client_data[client_idx].append(data)
    return client_data


def create_client_dataloaders(dataset, args):
    partition = getattr(args, "partition", "iid")
    task = getattr(args, "task", "")

    # STS-B is regression; Dirichlet label skew does not apply — use IID.
    if task == "stsb" and partition == "dirichlet":
        print(
            "STS-B is regression; using IID partition instead of Dirichlet label skew."
        )
        partition = "iid"

    if partition == "dirichlet":
        labels = np.array(dataset["labels"], dtype=np.int64).reshape(-1)
        rng = np.random.default_rng(args.seed)
        alpha = float(getattr(args, "dirichlet_alpha", 1.0))
        client_idx_lists = dirichlet_label_skew_indices(
            labels, args.num_clients, alpha, rng
        )
        if getattr(args, "print_partition_stats", False):
            print("Training split — client / label distribution (Dirichlet):")
            print_client_label_distribution(labels, client_idx_lists)
        return [
            DataLoader(
                dataset.select(idxs),
                batch_size=args.batch_size,
                shuffle=True,
            )
            for idxs in client_idx_lists
        ]

    client_data = [[] for _ in range(args.num_clients)]
    for data in dataset:
        client_idx = np.random.randint(args.num_clients)
        client_data[client_idx].append(data)
    return [
        DataLoader(cd, batch_size=args.batch_size, shuffle=True) for cd in client_data
    ]


def create_client_val_dataloaders(val_dataset, args):
    """
    Partition validation set across clients (scheme B for PFL eval).
    Uses a different RNG seed offset from train split for reproducibility.
    """
    partition = getattr(args, "partition", "iid")
    task = getattr(args, "task", "")
    if task == "stsb" and partition == "dirichlet":
        partition = "iid"

    rng_val = np.random.default_rng(int(args.seed) + 1337)

    if partition == "dirichlet":
        labels = np.array(val_dataset["labels"], dtype=np.int64).reshape(-1)
        alpha = float(getattr(args, "dirichlet_alpha", 1.0))
        client_idx_lists = dirichlet_label_skew_indices(
            labels, args.num_clients, alpha, rng_val
        )
        if getattr(args, "print_partition_stats", False):
            print("Validation split — client / label distribution (Dirichlet):")
            print_client_label_distribution(labels, client_idx_lists)
        return [
            DataLoader(
                val_dataset.select(idxs),
                batch_size=args.batch_size,
                shuffle=False,
            )
            for idxs in client_idx_lists
        ]

    client_indices = [[] for _ in range(args.num_clients)]
    n = len(val_dataset)
    for idx in range(n):
        client_indices[int(rng_val.integers(0, args.num_clients))].append(idx)
    _redistribute_empty_clients(client_indices)
    return [
        DataLoader(
            val_dataset.select(idxs),
            batch_size=args.batch_size,
            shuffle=False,
        )
        for idxs in client_indices
    ]


def create_e2e_data():
    def preprocess_function(examples):
        inputs = examples["meaning_representation"]
        targets = examples["human_reference"]

        # Combine the input-output pair into a single text
        model_inputs = [
            f"{input_} -> {target} <|endoftext|>"
            for input_, target in zip(inputs, targets)
        ]
        only_inputs = [f"{input_} ->" for input_, target in zip(inputs, targets)]

        # Tokenize the combined inputs
        tokenized_inputs = tokenizer(
            model_inputs,
            max_length=512,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        tokenized_only_inputs = tokenizer(
            only_inputs,
            max_length=512,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # Labels are the same as input_ids but shift them for next-token prediction
        tokenized_inputs["labels"] = tokenized_inputs["input_ids"].clone()

        # Set the labels to -100 where attention mask is 0 (this will ignore padding in loss computation)
        tokenized_inputs["labels"][tokenized_inputs["attention_mask"] == 0] = -100
        # set the labels to -100 where meaning representation input ids are present
        tokenized_inputs["labels"][tokenized_only_inputs["attention_mask"] == 1] = -100

        return tokenized_inputs

    dataset = load_dataset("tuetschek/e2e_nlg")
    from transformers import GPT2Tokenizer

    # Load the GPT-2 tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = (
        tokenizer.eos_token
    )  # GPT-2 doesn't have a pad token, so we set it to the eos token
    tokenized_datasets = dataset.map(preprocess_function, batched=True)
    return (
        tokenized_datasets["train"],
        tokenized_datasets["validation"],
        tokenized_datasets["test"],
        tokenizer,
    )


class DomainSFTDataset(Dataset):
    def __init__(self, records, tokenizer, max_seq_length=2048):
        self.records = records
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        prompt = (record.get("prompt") or "").strip()
        response = (record.get("response") or "").strip()
        text = f"{prompt}\n{response}{self.tokenizer.eos_token or ''}"
        prompt_text = f"{prompt}\n"

        tok_all = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_seq_length,
            padding="max_length",
            return_tensors="pt",
        )
        tok_prompt = self.tokenizer(
            prompt_text,
            truncation=True,
            max_length=self.max_seq_length,
            padding="max_length",
            return_tensors="pt",
        )

        item = {k: v.squeeze(0) for k, v in tok_all.items()}
        labels = item["input_ids"].clone()
        labels[item["attention_mask"] == 0] = -100
        prompt_mask = tok_prompt["attention_mask"].squeeze(0).bool()
        labels[prompt_mask] = -100
        item["labels"] = labels
        return item


def _load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _write_jsonl(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_domain_benchmark_from_jsonl(
    input_path,
    output_dir,
    num_clients_per_domain=5,
    min_samples_per_client=50,
    val_ratio=0.1,
    test_ratio=0.1,
    seed=42,
):
    """
    Build a federated domain benchmark from a single JSONL file whose rows contain:
    - domain
    - prompt
    - response
    Optional:
    - source_id
    - metadata

    The script generates:
    - per-client train/val/local-test files
    - domain-level held-out test files
    - one mixed global test file
    - manifest files
    """
    rows = _load_jsonl(input_path)
    by_domain = defaultdict(list)
    for row in rows:
        domain = (row.get("domain") or "").strip()
        prompt = (row.get("prompt") or "").strip()
        response = (row.get("response") or "").strip()
        if not domain or not prompt or not response:
            continue
        by_domain[domain].append(row)

    rng = np.random.default_rng(seed)
    split_dir = os.path.join(output_dir, f"seed_{seed}")
    os.makedirs(split_dir, exist_ok=True)

    clients_manifest = []
    domain_stats = {}
    global_test = []

    client_train_rows = []
    client_val_rows = []
    client_local_test_rows = []
    domain_test_rows = []

    client_id = 0
    for domain, domain_rows in sorted(by_domain.items()):
        rng.shuffle(domain_rows)
        n = len(domain_rows)
        n_test = max(1, int(n * test_ratio))
        n_val = max(1, int(n * val_ratio))
        domain_test = domain_rows[:n_test]
        remain = domain_rows[n_test:]
        global_test.extend(domain_test)
        domain_test_rows.extend(domain_test)

        shards = [[] for _ in range(num_clients_per_domain)]
        for idx, row in enumerate(remain):
            shards[idx % num_clients_per_domain].append(row)

        kept_clients = 0
        for shard in shards:
            if len(shard) < min_samples_per_client:
                continue
            rng.shuffle(shard)
            n_local_test = max(1, int(len(shard) * test_ratio))
            n_local_val = max(1, int(len(shard) * val_ratio))
            local_test = shard[:n_local_test]
            local_val = shard[n_local_test : n_local_test + n_local_val]
            local_train = shard[n_local_test + n_local_val :]
            if len(local_train) < max(1, min_samples_per_client // 2):
                continue

            for row in local_train:
                row = dict(row)
                row["client_id"] = client_id
                client_train_rows.append(row)
            for row in local_val:
                row = dict(row)
                row["client_id"] = client_id
                client_val_rows.append(row)
            for row in local_test:
                row = dict(row)
                row["client_id"] = client_id
                client_local_test_rows.append(row)

            clients_manifest.append(
                {
                    "client_id": client_id,
                    "domain": domain,
                    "n_train": len(local_train),
                    "n_val": len(local_val),
                    "n_local_test": len(local_test),
                }
            )
            client_id += 1
            kept_clients += 1

        domain_stats[domain] = {
            "n_total": n,
            "n_domain_test": len(domain_test),
            "n_clients": kept_clients,
        }

    _write_jsonl(os.path.join(split_dir, "train.jsonl"), client_train_rows)
    _write_jsonl(os.path.join(split_dir, "val.jsonl"), client_val_rows)
    _write_jsonl(
        os.path.join(split_dir, "test_local.jsonl"), client_local_test_rows
    )
    _write_jsonl(os.path.join(split_dir, "test_domain.jsonl"), domain_test_rows)
    _write_jsonl(os.path.join(split_dir, "test_global.jsonl"), global_test)

    with open(os.path.join(split_dir, "clients.json"), "w", encoding="utf-8") as f:
        json.dump(clients_manifest, f, indent=2, ensure_ascii=False)
    with open(os.path.join(split_dir, "domain_stats.json"), "w", encoding="utf-8") as f:
        json.dump(domain_stats, f, indent=2, ensure_ascii=False)

    return {
        "split_dir": split_dir,
        "num_clients": len(clients_manifest),
        "domains": sorted(domain_stats.keys()),
    }


def load_domain_sft_benchmark(split_dir):
    train_rows = _load_jsonl(os.path.join(split_dir, "train.jsonl"))
    val_rows = _load_jsonl(os.path.join(split_dir, "val.jsonl"))
    test_local_rows = _load_jsonl(os.path.join(split_dir, "test_local.jsonl"))
    test_domain_rows = _load_jsonl(os.path.join(split_dir, "test_domain.jsonl"))
    test_global_rows = _load_jsonl(os.path.join(split_dir, "test_global.jsonl"))
    with open(os.path.join(split_dir, "clients.json"), "r", encoding="utf-8") as f:
        clients = json.load(f)
    with open(os.path.join(split_dir, "domain_stats.json"), "r", encoding="utf-8") as f:
        domain_stats = json.load(f)
    return {
        "train": train_rows,
        "val": val_rows,
        "test_local": test_local_rows,
        "test_domain": test_domain_rows,
        "test_global": test_global_rows,
        "clients": clients,
        "domain_stats": domain_stats,
    }


def group_rows_by_client(rows):
    by_client = defaultdict(list)
    for row in rows:
        by_client[int(row["client_id"])].append(row)
    return by_client


def group_rows_by_domain(rows):
    by_domain = defaultdict(list)
    for row in rows:
        by_domain[row["domain"]].append(row)
    return by_domain


def create_domain_client_dataloaders(rows, tokenizer, args):
    by_client = group_rows_by_client(rows)
    client_ids = sorted(by_client.keys())
    loaders = []
    for client_id in client_ids:
        ds = DomainSFTDataset(
            by_client[client_id],
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
        )
        loaders.append(
            DataLoader(
                ds,
                batch_size=args.batch_size,
                shuffle=True,
            )
        )
    return client_ids, loaders


def create_domain_eval_dataloader(rows, tokenizer, args):
    ds = DomainSFTDataset(rows, tokenizer=tokenizer, max_seq_length=args.max_seq_length)
    return DataLoader(ds, batch_size=args.batch_size, shuffle=False)
