"""Document-aware sampling, token-weighted evaluation, resource probes and snapshots."""

import math
import resource
import sys
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from kaleido.experiments import array_digest, write_json
from kaleido.lm.inference import synchronize
from kaleido.lm.model import DecoderConfig, DecoderLM
from kaleido.lm.tokenizer import EOS


class TokenCorpus:
    def __init__(self, documents, tokenizer):
        self.vocab_size = len(tokenizer)
        self.documents = [
            {
                **doc,
                "tokens": np.array(
                    [EOS] + tokenizer.encode(doc["text"], add_eos=True), dtype=np.int64
                ),
            }
            for doc in documents
        ]
        self.splits = {
            s: [d for d in self.documents if d["split"] == s] for s in ("train", "valid", "test")
        }
        if any(not group for group in self.splits.values()):
            raise ValueError("each corpus split requires at least one document")

    def sample(self, batch_size, context_length, rng):
        if min(batch_size, context_length) < 1:
            raise ValueError("batch and context must be positive")
        documents = self.splits["train"]
        weights = np.array([len(d["tokens"]) - 1 for d in documents], dtype=float)
        chosen = rng.choice(len(documents), size=batch_size, p=weights / weights.sum())
        x = np.full((batch_size, context_length), EOS, dtype=np.int64)
        y = np.full_like(x, -100)
        for row, index in enumerate(chosen):
            tokens = documents[index]["tokens"]
            start = int(rng.integers(max(1, len(tokens) - context_length)))
            count = min(context_length, len(tokens) - start - 1)
            x[row, :count] = tokens[start : start + count]
            y[row, :count] = tokens[start + 1 : start + count + 1]
        return x, y

    def windows(self, split, context_length):
        if split not in self.splits or context_length < 1:
            raise ValueError("invalid split/context")
        for document in self.splits[split]:
            tokens = document["tokens"]
            for offset in range(0, len(tokens) - 1, context_length):
                count = min(context_length, len(tokens) - 1 - offset)
                x = np.full(context_length, EOS, dtype=np.int64)
                y = np.full(context_length, -100, dtype=np.int64)
                x[:count] = tokens[offset : offset + count]
                y[:count] = tokens[offset + 1 : offset + 1 + count]
                yield document["id"], offset, x, y


def metrics_from_losses(losses, byte_count):
    losses = np.asarray(losses, dtype=np.float64)
    if losses.ndim != 1 or len(losses) < 1 or not np.isfinite(losses).all() or byte_count < 1:
        raise ValueError("evaluation requires finite token losses and a positive byte count")
    nll = float(losses.mean())
    return {
        "nll": nll,
        "perplexity": math.exp(nll),
        "target_tokens": len(losses),
        "nll_sum": float(losses.sum()),
        "bits_per_utf8_byte_including_eos_cost": float(losses.sum() / math.log(2) / byte_count),
    }


@torch.inference_mode()
def evaluate(model, corpus, split, *, batch_size=16):
    device = next(model.parameters()).device
    previous = model.training
    model.eval()
    losses = []
    windows = []
    per_document = defaultdict(lambda: {"nll_sum": 0.0, "target_tokens": 0})
    pending = []

    def consume(batch):
        x = np.stack([item[2] for item in batch])
        y = np.stack([item[3] for item in batch])
        logits, _ = model(torch.tensor(x, device=device))
        values = (
            F.cross_entropy(
                logits.reshape(-1, model.config.vocab_size),
                torch.tensor(y, device=device).reshape(-1),
                ignore_index=-100,
                reduction="none",
            )
            .reshape(y.shape)
            .cpu()
            .numpy()
            .astype(np.float64)
        )
        for i, (doc_id, offset, _x, _y) in enumerate(batch):
            valid = values[i, y[i] != -100]
            losses.extend(valid.tolist())
            windows.append(
                {
                    "document_id": doc_id,
                    "offset": offset,
                    "target_tokens": len(valid),
                    "nll_sum": float(valid.sum()),
                }
            )
            per_document[doc_id]["nll_sum"] += float(valid.sum())
            per_document[doc_id]["target_tokens"] += len(valid)

    try:
        for item in corpus.windows(split, model.config.context_length):
            pending.append(item)
            if len(pending) == batch_size:
                consume(pending)
                pending = []
        if pending:
            consume(pending)
    finally:
        model.train(previous)
    byte_count = sum(len(d["text"].encode()) for d in corpus.splits[split])
    return (
        metrics_from_losses(losses, byte_count),
        {"windows": windows, "documents": dict(per_document)},
        np.array(losses),
    )


def count_baselines(corpus, split, *, alpha=0.1):
    if not math.isfinite(alpha) or alpha <= 0:
        raise ValueError("baseline smoothing must be positive")
    counts = np.full(corpus.vocab_size, alpha)
    pairs = np.full((corpus.vocab_size, corpus.vocab_size), alpha)
    for document in corpus.splits["train"]:
        tokens = document["tokens"]
        np.add.at(counts, tokens[1:], 1)
        np.add.at(pairs, (tokens[:-1], tokens[1:]), 1)
    unigram = -np.log(counts / counts.sum())
    bigram = -np.log(pairs / pairs.sum(axis=1, keepdims=True))
    losses = {name: [] for name in ("unigram", "bigram")}
    for document in corpus.splits[split]:
        tokens = document["tokens"]
        losses["unigram"].extend(unigram[tokens[1:]].tolist())
        losses["bigram"].extend(bigram[tokens[:-1], tokens[1:]].tolist())
    byte_count = sum(len(d["text"].encode()) for d in corpus.splits[split])
    return {name: metrics_from_losses(values, byte_count) for name, values in losses.items()}


def save_weights(path, model):
    path = Path(path)
    temporary = path.with_suffix(".tmp.npz")
    state = {
        name: value.detach().cpu().numpy().copy() for name, value in model.state_dict().items()
    }
    np.savez_compressed(temporary, **state)
    temporary.replace(path)
    metadata = {
        "config": asdict(model.config),
        "arrays": {name: array_digest(value) for name, value in state.items()},
        "note": "inference weights only; optimizer/RNG training resume state is not included",
    }
    write_json(path.with_suffix(".json"), metadata)
    return metadata


def load_weights(path, *, device="cpu"):
    import json

    path = Path(path)
    metadata = json.loads(path.with_suffix(".json").read_text())
    arrays = dict(np.load(path, allow_pickle=False))
    if set(arrays) != set(metadata["arrays"]) or any(
        array_digest(value) != metadata["arrays"][name] for name, value in arrays.items()
    ):
        raise ValueError("checkpoint array digest mismatch")
    model = DecoderLM(DecoderConfig(**metadata["config"]))
    model.load_state_dict(
        {name: torch.from_numpy(value) for name, value in arrays.items()}, strict=True
    )
    return model.to(device).eval()


def _one_step(model, optimizer, x, y):
    optimizer.zero_grad(set_to_none=True)
    logits, _ = model(x)
    loss = F.cross_entropy(
        logits.reshape(-1, model.config.vocab_size), y.reshape(-1), ignore_index=-100
    )
    if not torch.isfinite(loss):
        raise FloatingPointError("non-finite pretraining loss")
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
    optimizer.step()
    return loss.detach(), norm.detach()


def fit(
    model,
    corpus,
    *,
    steps=1000,
    batch_size=16,
    learning_rate=0.0008,
    warmup_steps=50,
    eval_interval=100,
    seed=42,
    weight_decay=0.01,
):
    if (
        type(steps) is not int
        or steps < 1
        or type(batch_size) is not int
        or batch_size < 1
        or type(warmup_steps) is not int
        or not 0 <= warmup_steps < steps
        or type(eval_interval) is not int
        or eval_interval < 1
        or not math.isfinite(learning_rate)
        or learning_rate <= 0
        or not math.isfinite(weight_decay)
        or weight_decay < 0
    ):
        raise ValueError("invalid pretraining budget or optimizer configuration")
    device = next(model.parameters()).device
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=(0.9, 0.95),
        eps=1e-8,
        foreach=False,
    )
    rng = np.random.default_rng(seed + 100)
    initial, _, _ = evaluate(model, corpus, "valid", batch_size=batch_size)

    def snapshot():
        return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    best, best_step, best_nll = snapshot(), 0, initial["nll"]
    history = [{"step": 0, "valid_nll": best_nll, "valid_target_tokens": initial["target_tokens"]}]
    trained_tokens = 0
    start = time.perf_counter()
    model.train()
    for step in range(1, steps + 1):
        if step <= warmup_steps:
            lr = learning_rate * step / warmup_steps
        else:
            fraction = (step - warmup_steps) / (steps - warmup_steps)
            lr = learning_rate * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * fraction)))
        for group in optimizer.param_groups:
            group["lr"] = lr
        x, y = corpus.sample(batch_size, model.config.context_length, rng)
        trained_tokens += int(np.sum(y != -100))
        loss, norm = _one_step(
            model, optimizer, torch.tensor(x, device=device), torch.tensor(y, device=device)
        )
        if step % eval_interval == 0 or step == steps:
            validation, _, _ = evaluate(model, corpus, "valid", batch_size=batch_size)
            history.append(
                {
                    "step": step,
                    "valid_nll": validation["nll"],
                    "valid_target_tokens": validation["target_tokens"],
                    "last_batch_nll": float(loss.cpu()),
                    "gradient_norm_before_clip": float(norm.cpu()),
                    "learning_rate": lr,
                }
            )
            if validation["nll"] < best_nll:
                best, best_step, best_nll = snapshot(), step, validation["nll"]
    synchronize(device)
    seconds = time.perf_counter() - start
    model.load_state_dict(best)
    model.eval()
    return {
        "selected_step": best_step,
        "valid_nll": best_nll,
        "initial_valid_nll": initial["nll"],
        "steps": steps,
        "actual_training_target_tokens": trained_tokens,
        "training_seconds": seconds,
        "training_tokens_per_second_including_validation": trained_tokens / seconds,
        "optimizer": "AdamW",
        "betas": [0.9, 0.95],
        "weight_decay": weight_decay,
        "gradient_clip_norm": 1.0,
    }, history


def probe_resources(config, corpus, *, batch_size=16, repeats=5):
    if repeats < 1:
        raise ValueError("probe repetitions must be positive")
    candidates = [("cpu", 1), ("cpu", 4)]
    if torch.backends.mps.is_available():
        candidates.append(("mps", 1))
    x, y = corpus.sample(batch_size, config.context_length, np.random.default_rng(71))
    results = []
    original_threads = torch.get_num_threads()
    for device, threads in candidates:
        row = {"device": device, "cpu_threads": threads}
        try:
            torch.set_num_threads(threads)
            torch.manual_seed(123)
            model = DecoderLM(config).to(device)
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=0.0005, weight_decay=0.01, foreach=False
            )
            tx, ty = torch.tensor(x, device=device), torch.tensor(y, device=device)
            for _ in range(2):
                _one_step(model, optimizer, tx, ty)
            durations = []
            for _ in range(repeats):
                synchronize(device)
                start = time.perf_counter()
                _one_step(model, optimizer, tx, ty)
                synchronize(device)
                durations.append(time.perf_counter() - start)
            median = float(np.median(durations))
            row.update(
                status="ok",
                seconds=durations,
                median_step_seconds=median,
                actual_target_tokens=int(np.sum(y != -100)),
                target_tokens_per_second=float(np.sum(y != -100) / median),
                parameters=sum(p.numel() for p in model.parameters()),
                parameter_bytes=sum(p.numel() * p.element_size() for p in model.parameters()),
            )
            if device == "mps":
                row["mps_current_allocated_bytes"] = torch.mps.current_allocated_memory()
                row["mps_driver_allocated_bytes"] = torch.mps.driver_allocated_memory()
            del model, optimizer, tx, ty
        except (RuntimeError, NotImplementedError) as error:
            row.update(status="unavailable", error=str(error))
        results.append(row)
    torch.set_num_threads(original_threads)
    viable = [row for row in results if row["status"] == "ok"]
    if not viable:
        raise RuntimeError("no device completed a finite forward/backward/update probe")
    best = min(viable, key=lambda row: row["median_step_seconds"])
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (
        1 if sys.platform == "darwin" else 1024
    )
    return {
        "candidates": results,
        "selected": {"device": best["device"], "cpu_threads": best["cpu_threads"]},
        "process_peak_rss_bytes": rss,
        "rss_scope": "process high-water mark including imports/probes; not model-only peak",
        "warmup_steps": 2,
        "measured_steps": repeats,
    }
