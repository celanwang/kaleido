"""Cached generation is checked against the exact cropped-window baseline."""

import math
import time

import torch


def synchronize(device):
    if torch.device(device).type == "mps":
        torch.mps.synchronize()
    elif torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


@torch.inference_mode()
def generate(
    model,
    prompt,
    *,
    max_new_tokens=64,
    temperature=1.0,
    top_k=None,
    seed=0,
    use_cache=True,
    eos_id=None,
):
    if model.training:
        raise ValueError("generation requires eval mode")
    if (
        type(max_new_tokens) is not int
        or max_new_tokens < 0
        or not math.isfinite(temperature)
        or temperature < 0
    ):
        raise ValueError("invalid generation length or temperature")
    if top_k is not None and (type(top_k) is not int or not 1 <= top_k <= model.config.vocab_size):
        raise ValueError("top_k must be within the vocabulary")
    if not prompt or any(
        type(t) is not int or not 0 <= t < model.config.vocab_size for t in prompt
    ):
        raise ValueError("prompt must contain valid integer IDs")
    if eos_id is not None and (
        type(eos_id) is not int or not 0 <= eos_id < model.config.vocab_size
    ):
        raise ValueError("invalid EOS ID")
    device = next(model.parameters()).device
    tokens = torch.tensor([prompt], dtype=torch.long, device=device)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    cache = None
    rebuilds = 0
    for _ in range(max_new_tokens):
        if not use_cache:
            logits, _ = model(tokens[:, -model.config.context_length :])
        elif cache is None or cache[0][0].shape[2] >= model.config.context_length:
            if cache is not None:
                rebuilds += 1
            logits, cache = model(tokens[:, -model.config.context_length :], use_cache=True)
        else:
            logits, cache = model(tokens[:, -1:], past_key_values=cache, use_cache=True)
        scores = logits[0, -1].float()
        if temperature == 0:
            following = int(scores.argmax())
        else:
            scores = scores / temperature
            if top_k is not None:
                threshold = scores.topk(top_k).values[-1]
                scores = scores.masked_fill(scores < threshold, float("-inf"))
            probability = scores.softmax(-1).cpu()
            following = int(torch.multinomial(probability, 1, generator=generator))
        tokens = torch.cat((tokens, torch.tensor([[following]], device=device)), dim=1)
        if following == eos_id:
            break
    return tokens[0].tolist(), {
        "cache_rebuilds": rebuilds,
        "generated_tokens": tokens.shape[1] - len(prompt),
    }


@torch.inference_mode()
def cache_equivalence(model, tokens, *, chunk_sizes=(3, 1, 4)):
    if model.training or tokens.shape[1] > model.config.context_length:
        raise ValueError("cache comparison requires eval mode and a fitting sequence")
    full, _ = model(tokens)
    cache = None
    outputs = []
    start = 0
    index = 0
    while start < tokens.shape[1]:
        size = chunk_sizes[index % len(chunk_sizes)]
        if type(size) is not int or size < 1:
            raise ValueError("chunks must have positive lengths")
        end = min(tokens.shape[1], start + size)
        logits, cache = model(tokens[:, start:end], past_key_values=cache, use_cache=True)
        outputs.append(logits)
        start = end
        index += 1
    cached = torch.cat(outputs, dim=1)
    difference = float((full - cached).abs().max())
    return {
        "max_abs_logit_error": difference,
        "greedy_agreement": bool(torch.equal(full.argmax(-1), cached.argmax(-1))),
        "cache_bytes": sum(v.numel() * v.element_size() for pair in cache for v in pair),
    }


def benchmark_decode(model, prompt, *, new_tokens=32, repeats=5):
    if type(repeats) is not int or repeats < 1 or new_tokens < 1:
        raise ValueError("benchmark requires positive repeat and token counts")
    result = {}
    device = next(model.parameters()).device
    # Fixed greedy workload, no EOS stop; each call starts with a fresh cache.
    for cached in (False, True):
        for _ in range(2):
            generate(model, prompt, max_new_tokens=new_tokens, temperature=0, use_cache=cached)
        durations = []
        last = None
        for _ in range(repeats):
            synchronize(device)
            start = time.perf_counter()
            ids, metadata = generate(
                model, prompt, max_new_tokens=new_tokens, temperature=0, use_cache=cached
            )
            synchronize(device)
            durations.append(time.perf_counter() - start)
            last = ids
        result["cached" if cached else "naive"] = {
            "seconds": durations,
            "median_seconds": sorted(durations)[len(durations) // 2],
            "token_ids": last,
            "metadata": metadata,
        }
    if result["cached"]["token_ids"] != result["naive"]["token_ids"]:
        raise RuntimeError("greedy decoding changed with cache")
    result["speedup"] = result["naive"]["median_seconds"] / result["cached"]["median_seconds"]
    result["generated_tokens"] = new_tokens
    return result
