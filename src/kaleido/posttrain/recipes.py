"""SFT -> online GRPO and fixed-SFT rejection-sampling control, with complete audits."""

import time

import numpy as np
import torch

from kaleido.lm.inference import synchronize
from kaleido.posttrain.model import (
    adapter_state,
    completion_log_probs,
    load_adapter,
    pack_completions,
    prompt_tokens,
    sample_completions,
    sft_loss,
    sft_tokens,
)
from kaleido.posttrain.objective import group_advantages, grpo_loss
from kaleido.posttrain.tasks import score_response, summarize, training_reward


def eos_ids(model, tokenizer):
    ids = model.generation_config.eos_token_id
    return [ids] if isinstance(ids, int) else list(ids or [tokenizer.eos_token_id])


def decode_completion(model, tokenizer, ids):
    # Remove only the real final stop token; never hide other special-token injections.
    content = ids[:-1] if ids and ids[-1] in eos_ids(model, tokenizer) else ids
    return tokenizer.decode(content, skip_special_tokens=False)


def optimizer_for(model, lr):
    return torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
        foreach=False,
    )


def update(model, optimizer, loss):
    if not torch.isfinite(loss):
        raise FloatingPointError("non-finite post-training objective")
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(
        [p for p in model.parameters() if p.requires_grad], 1.0, error_if_nonfinite=True
    )
    optimizer.step()
    return float(norm.detach().cpu())


def batch_from_rows(model, tokenizer, rows):
    encoded = [sft_tokens(tokenizer, row) for row in rows]
    return pack_completions(
        [x[0] for x in encoded],
        [x[1] for x in encoded],
        pad_id=tokenizer.pad_token_id,
        device=next(model.parameters()).device,
    )


@torch.no_grad()
def teacher_nll(model, tokenizer, rows, *, batch_size=8):
    total, count = 0.0, 0
    for start in range(0, len(rows), batch_size):
        batch = batch_from_rows(model, tokenizer, rows[start : start + batch_size])
        logp = completion_log_probs(model, batch)
        total += float((-(logp * batch["mask"]).sum()).cpu())
        count += int(batch["mask"].sum())
    return total / count, count


def evaluate_policy(model, tokenizer, rows, *, batch_size=8, max_new_tokens=32, seed=9000):
    if not rows or batch_size < 1:
        raise ValueError("evaluation requires rows and positive batch size")
    start = time.perf_counter()
    records = []
    prompt_count = 0
    generated_count = 0
    for offset in range(0, len(rows), batch_size):
        group = rows[offset : offset + batch_size]
        prompts = [prompt_tokens(tokenizer, row) for row in group]
        sequences, truncated = sample_completions(
            model,
            prompts,
            pad_id=tokenizer.pad_token_id,
            eos_ids=eos_ids(model, tokenizer),
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            seed=seed,
        )
        for row, prompt, ids, cut in zip(group, prompts, sequences, truncated, strict=True):
            response = decode_completion(model, tokenizer, ids)
            records.append(
                {
                    "id": row["id"],
                    "family": row["family"],
                    "response": response,
                    "completion_ids": ids,
                    "prompt_tokens": len(prompt),
                    "generated_tokens": len(ids),
                    **score_response(row, response, truncated=cut),
                }
            )
            prompt_count += len(prompt)
            generated_count += len(ids)
    nll, target_count = teacher_nll(model, tokenizer, rows, batch_size=batch_size)
    synchronize(next(model.parameters()).device)
    return {
        **summarize(records),
        "response_nll": nll,
        "teacher_target_tokens": target_count,
        "generated_tokens": generated_count,
        "prompt_tokens": prompt_count,
        "wall_seconds": time.perf_counter() - start,
    }, records


def select_key(metrics):
    return metrics["strict_accuracy"], -metrics["response_nll"]


def train_sft(
    model,
    tokenizer,
    train,
    valid,
    *,
    steps=64,
    batch_size=4,
    lr=0.0002,
    eval_interval=16,
    max_new_tokens=32,
    seed=42,
    initial_validation=None,
    emit=None,
):
    optimizer = optimizer_for(model, lr)
    rng = np.random.default_rng(seed + 100)
    initial = (
        initial_validation
        or evaluate_policy(model, tokenizer, valid, max_new_tokens=max_new_tokens)[0]
    )
    best, best_step, best_metrics = adapter_state(model), 0, initial
    history = [{"step": 0, "validation": initial}]
    validation_samples = []
    tokens = 0
    start = time.perf_counter()
    for step in range(1, steps + 1):
        rows = [
            train[int(i)]
            for i in rng.choice(len(train), size=min(batch_size, len(train)), replace=False)
        ]
        batch = batch_from_rows(model, tokenizer, rows)
        optimizer.zero_grad(set_to_none=True)
        loss = sft_loss(completion_log_probs(model, batch), batch["mask"])
        norm = update(model, optimizer, loss)
        tokens += int(batch["mask"].sum())
        if step % eval_interval == 0 or step == steps:
            metrics, records = evaluate_policy(
                model, tokenizer, valid, max_new_tokens=max_new_tokens
            )
            history.append(
                {
                    "step": step,
                    "validation": metrics,
                    "last_loss": float(loss.detach().cpu()),
                    "gradient_norm": norm,
                }
            )
            validation_samples.append({"step": step, "samples": records})
            if select_key(metrics) > select_key(best_metrics):
                best, best_step, best_metrics = adapter_state(model), step, metrics
            if emit:
                emit(
                    {
                        "method": "sft",
                        "step": step,
                        "validation_accuracy": metrics["strict_accuracy"],
                    }
                )
    load_adapter(model, best)
    return (
        {
            "selected_step": best_step,
            "selected_validation": best_metrics,
            "updates": steps,
            "update_completion_tokens": tokens,
            "wall_seconds": time.perf_counter() - start,
        },
        history,
        validation_samples,
    )


def rollout_schedule(train, *, iterations, prompts_per_iteration, seed):
    rng = np.random.default_rng(seed + 200)
    return [
        [train[int(i)] for i in rng.choice(len(train), size=prompts_per_iteration, replace=False)]
        for _ in range(iterations)
    ]


def collect_rollout(model, tokenizer, rows, *, group_size, max_new_tokens, seed):
    expanded = [row for row in rows for _ in range(group_size)]
    prompts = [prompt_tokens(tokenizer, row) for row in expanded]
    sequences, truncated = sample_completions(
        model,
        prompts,
        pad_id=tokenizer.pad_token_id,
        eos_ids=eos_ids(model, tokenizer),
        max_new_tokens=max_new_tokens,
        temperature=1.0,
        seed=seed,
    )
    records = []
    rewards = []
    for index, (row, ids, cut) in enumerate(zip(expanded, sequences, truncated, strict=True)):
        response = decode_completion(model, tokenizer, ids)
        reward = training_reward(row, response, truncated=cut)
        independent = score_response(row, response, truncated=cut)
        if reward != independent["reward"]:
            raise AssertionError("reward and independent verifier disagree")
        records.append(
            {
                "id": row["id"],
                "family": row["family"],
                "group": index // group_size,
                "response": response,
                "completion_ids": ids,
                "training_reward": reward,
                **independent,
            }
        )
        rewards.append(reward)
    batch = pack_completions(
        prompts, sequences, pad_id=tokenizer.pad_token_id, device=next(model.parameters()).device
    )
    return batch, rewards, records


def train_grpo(
    model,
    tokenizer,
    train,
    valid,
    reference,
    *,
    iterations=24,
    prompts_per_iteration=2,
    group_size=4,
    epochs=2,
    lr=0.00005,
    eval_interval=4,
    max_new_tokens=32,
    beta=0.04,
    clip=0.2,
    seed=42,
    initial_validation=None,
    emit=None,
):
    device = next(model.parameters()).device
    optimizer = optimizer_for(model, lr)
    initial = (
        initial_validation
        or evaluate_policy(model, tokenizer, valid, max_new_tokens=max_new_tokens)[0]
    )
    best, best_step, best_metrics = adapter_state(model), 0, initial
    history = [{"iteration": 0, "validation": initial}]
    rollouts = []
    validation_samples = []
    targets = 0
    generated = 0
    updates = 0
    start = time.perf_counter()
    schedule = rollout_schedule(
        train, iterations=iterations, prompts_per_iteration=prompts_per_iteration, seed=seed
    )
    for iteration, rows in enumerate(schedule, 1):
        batch, rewards, records = collect_rollout(
            model,
            tokenizer,
            rows,
            group_size=group_size,
            max_new_tokens=max_new_tokens,
            seed=seed + 300 + iteration,
        )
        advantages, group_stats = group_advantages(rewards, group_size)
        advantages = advantages.to(device)
        with torch.no_grad():
            old = completion_log_probs(model, batch).detach()
            current = adapter_state(model, cpu=False)
            try:
                load_adapter(model, reference)
                ref = completion_log_probs(model, batch).detach()
            finally:
                load_adapter(model, current)
        generated += int(batch["mask"].sum())
        diagnostics = []
        for _epoch in range(epochs):
            optimizer.zero_grad(set_to_none=True)
            loss, stats = grpo_loss(
                completion_log_probs(model, batch),
                old,
                ref,
                batch["mask"],
                advantages,
                clip=clip,
                beta=beta,
            )
            norm = update(model, optimizer, loss)
            targets += int(batch["mask"].sum())
            updates += 1
            diagnostics.append({"loss": float(loss.detach().cpu()), "gradient_norm": norm, **stats})
        history.append({"iteration": iteration, **group_stats, "updates": diagnostics})
        rollouts.append(
            {
                "iteration": iteration,
                "question_ids": [r["id"] for r in rows],
                "sampling_seed": seed + 300 + iteration,
                "samples": records,
                "group_stats": group_stats,
            }
        )
        if iteration % eval_interval == 0 or iteration == iterations:
            metrics, samples = evaluate_policy(
                model, tokenizer, valid, max_new_tokens=max_new_tokens
            )
            history[-1]["validation"] = metrics
            validation_samples.append({"iteration": iteration, "samples": samples})
            if select_key(metrics) > select_key(best_metrics):
                best, best_step, best_metrics = adapter_state(model), iteration, metrics
            if emit:
                emit(
                    {
                        "method": "grpo",
                        "iteration": iteration,
                        "validation_accuracy": metrics["strict_accuracy"],
                        **group_stats,
                    }
                )
    load_adapter(model, best)
    return (
        {
            "selected_iteration": best_step,
            "selected_validation": best_metrics,
            "updates": updates,
            "generated_completion_tokens": generated,
            "update_completion_tokens": targets,
            "zero_variance_group_fraction": float(
                np.mean([r["group_stats"]["zero_variance_group_fraction"] for r in rollouts])
            ),
            "wall_seconds": time.perf_counter() - start,
        },
        history,
        rollouts,
        validation_samples,
    )


def accepted_completions(rollouts):
    seen = set()
    accepted = []
    for rollout in rollouts:
        for sample in rollout["samples"]:
            if sample["training_reward"] != 1:
                continue
            key = sample["id"], tuple(sample["completion_ids"])
            if key in seen:
                continue
            seen.add(key)
            accepted.append(sample)
    return accepted


def train_rejection_sft(
    model,
    tokenizer,
    train,
    valid,
    *,
    iterations=24,
    prompts_per_iteration=2,
    group_size=4,
    updates=48,
    batch_size=8,
    lr=0.00005,
    eval_interval=8,
    max_new_tokens=32,
    seed=42,
    initial_validation=None,
    emit=None,
):
    # Sampling policy is the fixed SFT checkpoint for the entire collection phase.
    start = time.perf_counter()
    rollouts = []
    schedule = rollout_schedule(
        train, iterations=iterations, prompts_per_iteration=prompts_per_iteration, seed=seed
    )
    generated = 0
    for iteration, rows in enumerate(schedule, 1):
        batch, _rewards, records = collect_rollout(
            model,
            tokenizer,
            rows,
            group_size=group_size,
            max_new_tokens=max_new_tokens,
            seed=seed + 300 + iteration,
        )
        generated += int(batch["mask"].sum())
        rollouts.append(
            {
                "iteration": iteration,
                "question_ids": [r["id"] for r in rows],
                "sampling_seed": seed + 300 + iteration,
                "samples": records,
            }
        )
    accepted = accepted_completions(rollouts)
    lookup = {r["id"]: r for r in train}
    initial = (
        initial_validation
        or evaluate_policy(model, tokenizer, valid, max_new_tokens=max_new_tokens)[0]
    )
    best, best_step, best_metrics = adapter_state(model), 0, initial
    history = [{"step": 0, "validation": initial}]
    validation_samples = []
    targets = 0
    if not accepted:
        return (
            {
                "status": "no_accepted_completions",
                "accepted_unique": 0,
                "updates": 0,
                "generated_completion_tokens": generated,
                "update_completion_tokens": 0,
                "selected_step": 0,
                "selected_validation": initial,
                "wall_seconds": time.perf_counter() - start,
            },
            history,
            rollouts,
            validation_samples,
        )
    optimizer = optimizer_for(model, lr)
    rng = np.random.default_rng(seed + 400)
    for step in range(1, updates + 1):
        samples = [accepted[int(i)] for i in rng.integers(0, len(accepted), size=batch_size)]
        prompts = [prompt_tokens(tokenizer, lookup[s["id"]]) for s in samples]
        batch = pack_completions(
            prompts,
            [s["completion_ids"] for s in samples],
            pad_id=tokenizer.pad_token_id,
            device=next(model.parameters()).device,
        )
        optimizer.zero_grad(set_to_none=True)
        loss = sft_loss(completion_log_probs(model, batch), batch["mask"])
        norm = update(model, optimizer, loss)
        targets += int(batch["mask"].sum())
        if step % eval_interval == 0 or step == updates:
            metrics, records = evaluate_policy(
                model, tokenizer, valid, max_new_tokens=max_new_tokens
            )
            history.append(
                {
                    "step": step,
                    "validation": metrics,
                    "last_loss": float(loss.detach().cpu()),
                    "gradient_norm": norm,
                }
            )
            validation_samples.append({"step": step, "samples": records})
            if select_key(metrics) > select_key(best_metrics):
                best, best_step, best_metrics = adapter_state(model), step, metrics
            if emit:
                emit(
                    {
                        "method": "rejection_sft",
                        "step": step,
                        "accepted_unique": len(accepted),
                        "validation_accuracy": metrics["strict_accuracy"],
                    }
                )
    load_adapter(model, best)
    return (
        {
            "status": "completed",
            "accepted_unique": len(accepted),
            "accepted_raw": sum(s["training_reward"] == 1 for r in rollouts for s in r["samples"]),
            "updates": updates,
            "generated_completion_tokens": generated,
            "update_completion_tokens": targets,
            "selected_step": best_step,
            "selected_validation": best_metrics,
            "wall_seconds": time.perf_counter() - start,
        },
        history,
        rollouts,
        validation_samples,
    )


def probe(model, tokenizer, train, *, batch_size=4, lr=0.0002):
    snapshot = adapter_state(model, cpu=False)
    device = next(model.parameters()).device
    batch = batch_from_rows(model, tokenizer, train[:batch_size])
    optimizer = optimizer_for(model, lr)
    durations = []
    try:
        for _ in range(3):
            optimizer.zero_grad(set_to_none=True)
            synchronize(device)
            start = time.perf_counter()
            loss = sft_loss(completion_log_probs(model, batch), batch["mask"])
            norm = update(model, optimizer, loss)
            synchronize(device)
            durations.append(time.perf_counter() - start)
    finally:
        load_adapter(model, snapshot)
        model.zero_grad(set_to_none=True)
    prompt = prompt_tokens(tokenizer, train[0])
    native = model.generate(
        torch.tensor([prompt], device=device),
        attention_mask=torch.ones((1, len(prompt)), device=device, dtype=torch.long),
        do_sample=False,
        max_new_tokens=16,
        eos_token_id=eos_ids(model, tokenizer),
        pad_token_id=tokenizer.pad_token_id,
        repetition_penalty=1.0,
    )[0, len(prompt) :].tolist()
    actual, _ = sample_completions(
        model,
        [prompt],
        pad_id=tokenizer.pad_token_id,
        eos_ids=eos_ids(model, tokenizer),
        max_new_tokens=16,
    )
    if native != actual[0]:
        raise AssertionError("native generation and raw-policy sampler disagree")
    result = {
        "update_seconds": durations,
        "warmup_steps": 1,
        "median_warm_update_seconds": float(np.median(durations[1:])),
        "completion_tokens": int(batch["mask"].sum()),
        "prompt_width": batch["prompt_width"],
        "native_greedy_match": True,
        "base_parameters": sum(p.numel() for p in model.parameters() if not p.requires_grad),
        "adapter_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "device": str(device),
        "last_gradient_norm": norm,
    }
    if device.type == "mps":
        result.update(
            mps_current_allocated_bytes=torch.mps.current_allocated_memory(),
            mps_driver_allocated_bytes=torch.mps.driver_allocated_memory(),
        )
    return result
