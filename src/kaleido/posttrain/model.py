"""Native-template Qwen, explicit frozen-base LoRA and completion-only likelihoods."""

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from kaleido.experiments import array_digest, write_json
from kaleido.posttrain.tasks import SYSTEM
from kaleido.repro.integrity import local_path
from kaleido.serialization import load_object


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


class LoRALinear(nn.Module):
    def __init__(self, base, *, rank=8, alpha=16.0, generator=None):
        super().__init__()
        if type(rank) is not int or rank < 1 or not np.isfinite(alpha) or alpha <= 0:
            raise ValueError("LoRA rank/alpha must be positive")
        if base.weight.dtype != torch.float32:
            raise ValueError("P4 explicitly uses float32 base and adapters")
        self.base = base.requires_grad_(False)
        self.scale = alpha / rank
        a = torch.randn(rank, base.in_features, generator=generator) * base.in_features**-0.5
        self.lora_a = nn.Parameter(a.to(base.weight.device))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device))

    def forward(self, x):
        return self.base(x) + F.linear(F.linear(x, self.lora_a), self.lora_b) * self.scale


def inject_lora(model, *, rank=8, alpha=16.0, seed=42):
    model.requires_grad_(False)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    targets = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear) and name.rsplit(".", 1)[-1] in ("q_proj", "v_proj")
    ]
    if not targets:
        raise ValueError("model has no q_proj/v_proj linear targets")
    for name, module in targets:
        parent, attribute = name.rsplit(".", 1)
        setattr(
            model.get_submodule(parent),
            attribute,
            LoRALinear(module, rank=rank, alpha=alpha, generator=generator),
        )
    return [name for name, _ in targets]


def adapter_state(model, *, cpu=True):
    return {
        name: (p.detach().cpu().clone() if cpu else p.detach().clone())
        for name, p in model.named_parameters()
        if p.requires_grad
    }


def load_adapter(model, state):
    current = {name: p for name, p in model.named_parameters() if p.requires_grad}
    if set(state) != set(current):
        raise ValueError("adapter parameter keys differ")
    values = {
        name: torch.as_tensor(value, device=current[name].device, dtype=current[name].dtype)
        for name, value in state.items()
    }
    if any(
        value.shape != current[name].shape or not torch.isfinite(value).all()
        for name, value in values.items()
    ):
        raise ValueError("adapter shapes/values invalid")
    with torch.no_grad():
        for name, value in values.items():
            current[name].copy_(value)


def save_adapter(path, model, *, metadata):
    path = Path(path)
    state = {k: v.numpy() for k, v in adapter_state(model).items()}
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **state)
    temporary.replace(path)
    info = {**metadata, "arrays": {k: array_digest(v) for k, v in state.items()}}
    write_json(path.with_suffix(".json"), info)
    return info


def read_adapter(path):
    path = Path(path)
    info = json.loads(path.with_suffix(".json").read_text())
    state = dict(np.load(path, allow_pickle=False))
    if set(state) != set(info["arrays"]) or any(
        array_digest(v) != info["arrays"][k] for k, v in state.items()
    ):
        raise ValueError("adapter checkpoint digest mismatch")
    return state, info


def _pinned_files(spec):
    expected = spec.get("files_sha256")
    required = {
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "LICENSE",
        "merges.txt",
        "vocab.json",
    }
    if not isinstance(expected, dict) or set(expected) != required:
        raise ValueError("model spec must pin all native tokenizer/config/license file hashes")
    if spec.get("weights_file") != "model.safetensors":
        raise ValueError("this model spec requires one model.safetensors checkpoint")
    files = {spec["weights_file"]: spec.get("weights_sha256"), **expected}
    if any(
        not isinstance(v, str) or re.fullmatch(r"[0-9a-f]{64}", v) is None for v in files.values()
    ):
        raise ValueError("model spec requires canonical SHA256 checksums")
    return files


def validate_snapshot(directory, spec):
    files = _pinned_files(spec)
    for name, checksum in files.items():
        path = local_path(directory, name)
        if not path.is_file() or file_sha256(path) != checksum:
            raise ValueError(f"pinned model file checksum mismatch: {name}")
    return files


def load_pinned(spec_path, *, device="mps", download=True):
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError
    from transformers import AutoModelForCausalLM, AutoTokenizer

    spec = load_object(spec_path)
    if (
        spec.get("trust_remote_code") is not False
        or not isinstance(spec.get("revision"), str)
        or re.fullmatch(r"[0-9a-f]{40}", spec["revision"]) is None
    ):
        raise ValueError("model must use a fixed commit and no remote code")
    expected_files = _pinned_files(spec)
    if not isinstance(spec.get("repo_id"), str) or not spec["repo_id"]:
        raise ValueError("model spec requires a repository ID")
    options = {
        "repo_id": spec["repo_id"],
        "revision": spec["revision"],
        "allow_patterns": list(expected_files),
    }
    try:
        directory = Path(snapshot_download(**options, local_files_only=True))
    except LocalEntryNotFoundError:
        if not download:
            raise
        directory = None
    required = tuple(expected_files)
    if directory is None or any(not (directory / name).is_file() for name in required):
        if not download:
            raise FileNotFoundError("pinned snapshot is incomplete in local cache")
        directory = Path(snapshot_download(**options, local_files_only=False))
    validate_snapshot(directory, spec)
    tokenizer = AutoTokenizer.from_pretrained(
        directory, local_files_only=True, trust_remote_code=False, padding_side="left"
    )
    if tokenizer.pad_token_id is None or tokenizer.eos_token_id is None:
        raise ValueError("native tokenizer must define pad and EOS tokens")
    template = tokenizer.get_chat_template()
    model = (
        AutoModelForCausalLM.from_pretrained(
            directory,
            local_files_only=True,
            trust_remote_code=False,
            dtype=torch.float32,
            attn_implementation="sdpa",
        )
        .to(device)
        .eval()
    )
    provenance = {
        **spec,
        "chat_template_sha256": hashlib.sha256(template.encode()).hexdigest(),
        "files": {
            p.name: file_sha256(p)
            for p in directory.iterdir()
            if p.is_file() and p.name != "model.safetensors"
        },
        "base_parameters": sum(p.numel() for p in model.parameters()),
        "dtype": "float32",
        "device": str(device),
    }
    return model, tokenizer, provenance


def prompt_tokens(tokenizer, row):
    return list(
        tokenizer.apply_chat_template(
            [{"role": "system", "content": SYSTEM}, {"role": "user", "content": row["prompt"]}],
            tokenize=True,
            add_generation_prompt=True,
            return_dict=False,
        )
    )


def sft_tokens(tokenizer, row):
    prompt = prompt_tokens(tokenizer, row)
    full = list(
        tokenizer.apply_chat_template(
            [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": row["prompt"]},
                {"role": "assistant", "content": row["completion"]},
            ],
            tokenize=True,
            return_dict=False,
        )
    )
    if full[: len(prompt)] != prompt:
        raise ValueError("native template does not preserve generation prefix")
    completion = full[len(prompt) :]
    if tokenizer.eos_token_id not in completion:
        raise ValueError("native assistant template has no EOS")
    completion = completion[: completion.index(tokenizer.eos_token_id) + 1]
    return prompt, completion


def pack_completions(prompts, completions, *, pad_id, device="cpu"):
    if (
        len(prompts) != len(completions)
        or not prompts
        or any(not p or not c for p, c in zip(prompts, completions, strict=True))
    ):
        raise ValueError("training requires matching non-empty prompts and completions")
    width = max(map(len, prompts))
    tail = max(map(len, completions))
    ids = torch.full((len(prompts), width + tail), pad_id, dtype=torch.long, device=device)
    attention = torch.zeros_like(ids)
    mask = torch.zeros((len(prompts), tail), dtype=torch.bool, device=device)
    for i, (prompt, completion) in enumerate(zip(prompts, completions, strict=True)):
        ids[i, width - len(prompt) : width] = torch.tensor(prompt, device=device)
        ids[i, width : width + len(completion)] = torch.tensor(completion, device=device)
        attention[i, width - len(prompt) : width + len(completion)] = 1
        mask[i, : len(completion)] = True
    positions = (attention.cumsum(-1) - 1).clamp(min=0)
    return {
        "input_ids": ids[:, :-1],
        "attention_mask": attention[:, :-1],
        "position_ids": positions[:, :-1],
        "targets": ids[:, width:],
        "mask": mask,
        "prompt_width": width,
    }


def completion_log_probs(model, batch):
    count = batch["targets"].shape[1]
    logits = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        position_ids=batch["position_ids"],
        use_cache=False,
        logits_to_keep=count,
    ).logits
    if logits.shape[:2] != batch["targets"].shape:
        raise ValueError("native logits slicing changed")
    logits = logits.float()
    selected = logits.gather(-1, batch["targets"].unsqueeze(-1)).squeeze(-1)
    return selected - torch.logsumexp(logits, dim=-1)


def sft_loss(log_probs, mask):
    if log_probs.shape != mask.shape or mask.dtype != torch.bool or not mask.any():
        raise ValueError("SFT requires a non-empty completion mask")
    return -(log_probs * mask).sum() / mask.sum()


@torch.inference_mode()
def sample_completions(
    model, prompts, *, pad_id, eos_ids, max_new_tokens=32, temperature=0.0, seed=42
):
    if (
        not prompts
        or any(not p for p in prompts)
        or type(max_new_tokens) is not int
        or max_new_tokens < 1
        or not np.isfinite(temperature)
        or temperature < 0
    ):
        raise ValueError("invalid prompt or generation budget")
    device = next(model.parameters()).device
    model.eval()
    width = max(map(len, prompts))
    ids = torch.full((len(prompts), width), pad_id, dtype=torch.long, device=device)
    attention = torch.zeros_like(ids)
    for i, prompt in enumerate(prompts):
        ids[i, -len(prompt) :] = torch.tensor(prompt, device=device)
        attention[i, -len(prompt) :] = 1
    generator = torch.Generator(device="cpu").manual_seed(seed)
    finished = torch.zeros(len(prompts), dtype=torch.bool, device=device)
    sequences = [[] for _ in prompts]
    past = None
    current = ids
    for _ in range(max_new_tokens):
        positions = (attention.cumsum(-1) - 1).clamp(min=0)
        output = model(
            input_ids=current,
            attention_mask=attention,
            position_ids=positions[:, -current.shape[1] :],
            past_key_values=past,
            use_cache=True,
            logits_to_keep=1,
        )
        logits = output.logits[:, -1, :].float()
        past = output.past_key_values
        if temperature == 0:
            following = logits.argmax(-1)
        else:
            following = (
                torch.multinomial((logits / temperature).softmax(-1).cpu(), 1, generator=generator)
                .squeeze(-1)
                .to(device)
            )
        already = finished.cpu().tolist()
        next_ids = following.cpu().tolist()
        for i, token in enumerate(next_ids):
            if not already[i]:
                sequences[i].append(token)
        was_finished = finished.clone()
        for eos in eos_ids:
            finished |= following == eos
        if bool(finished.all()):
            break
        following = torch.where(was_finished, torch.full_like(following, pad_id), following)
        current = following[:, None]
        attention = torch.cat((attention, (~was_finished).long()[:, None]), dim=1)
    return sequences, [not value for value in finished.cpu().tolist()]
