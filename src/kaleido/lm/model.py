"""Explicit pre-LN decoder blocks using PyTorch tensors, not a pretrained model."""

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class DecoderConfig:
    vocab_size: int = 512
    context_length: int = 128
    width: int = 128
    layers: int = 2
    heads: int = 4
    dropout: float = 0.0

    def __post_init__(self):
        for name in ("vocab_size", "context_length", "width", "layers", "heads"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.width % self.heads or not 0 <= self.dropout < 1:
            raise ValueError("width must divide into heads; dropout must lie in [0,1)")


class CausalAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.qkv = nn.Linear(config.width, 3 * config.width)
        self.output = nn.Linear(config.width, config.width)

    def forward(self, x, past=None):
        batch, time, width = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q, k, v = (
            part.reshape(batch, time, self.config.heads, width // self.config.heads).transpose(1, 2)
            for part in (q, k, v)
        )
        offset = 0 if past is None else past[0].shape[2]
        if past is not None:
            k = torch.cat((past[0], k), dim=2)
            v = torch.cat((past[1], v), dim=2)
        scores = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])
        visible = (
            torch.arange(k.shape[2], device=x.device)[None, :]
            <= offset + torch.arange(time, device=x.device)[:, None]
        )
        probability = F.softmax(scores.masked_fill(~visible, float("-inf")), dim=-1)
        probability = F.dropout(probability, p=self.config.dropout, training=self.training)
        result = (probability @ v).transpose(1, 2).contiguous().reshape(batch, time, width)
        return self.output(result), (k, v)


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.width)
        self.attention = CausalAttention(config)
        self.ln2 = nn.LayerNorm(config.width)
        self.up = nn.Linear(config.width, 4 * config.width)
        self.down = nn.Linear(4 * config.width, config.width)
        self.dropout = config.dropout

    def forward(self, x, past=None):
        attended, present = self.attention(self.ln1(x), past)
        x = x + F.dropout(attended, p=self.dropout, training=self.training)
        x = x + F.dropout(
            self.down(F.gelu(self.up(self.ln2(x)), approximate="tanh")),
            p=self.dropout,
            training=self.training,
        )
        return x, present


class DecoderLM(nn.Module):
    def __init__(self, config: DecoderConfig):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.width)
        self.position_embedding = nn.Embedding(config.context_length, config.width)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.layers)])
        self.final_norm = nn.LayerNorm(config.width)

        def initialize(module):
            if isinstance(module, (nn.Linear, nn.Embedding)):
                nn.init.normal_(module.weight, std=0.02)
                if isinstance(module, nn.Linear) and module.bias is not None:
                    nn.init.zeros_(module.bias)

        self.apply(initialize)

    def forward(self, tokens, *, past_key_values=None, use_cache=False):
        if tokens.ndim != 2 or min(tokens.shape) < 1 or tokens.dtype != torch.long:
            raise ValueError("tokens must be a non-empty int64 (batch,time) tensor")
        if torch.any(tokens < 0) or torch.any(tokens >= self.config.vocab_size):
            raise ValueError("token ID outside vocabulary")
        if (past_key_values is not None and not use_cache) or (use_cache and self.training):
            raise ValueError("KV cache requires evaluation mode and use_cache=True")
        offset = 0
        if past_key_values is not None:
            if len(past_key_values) != self.config.layers:
                raise ValueError("cache layer count mismatch")
            lengths = []
            for key, value in past_key_values:
                if (
                    key.ndim != 4
                    or key.shape != value.shape
                    or key.shape[:2] != (tokens.shape[0], self.config.heads)
                    or key.shape[3] != self.config.width // self.config.heads
                    or key.device != tokens.device
                    or value.device != tokens.device
                    or key.dtype != self.token_embedding.weight.dtype
                    or value.dtype != key.dtype
                ):
                    raise ValueError("cache shape, device or dtype mismatch")
                lengths.append(key.shape[2])
            if len(set(lengths)) != 1:
                raise ValueError("cache positions must agree across layers")
            offset = lengths[0]
        if offset + tokens.shape[1] > self.config.context_length:
            raise ValueError("context limit exceeded; rebuild cache with a fresh cropped window")
        positions = torch.arange(offset, offset + tokens.shape[1], device=tokens.device)
        x = self.token_embedding(tokens) + self.position_embedding(positions)
        x = F.dropout(x, p=self.config.dropout, training=self.training)
        presents = []
        for i, block in enumerate(self.blocks):
            x, present = block(x, None if past_key_values is None else past_key_values[i])
            if use_cache:
                presents.append(tuple(t.detach() for t in present))
        logits = F.linear(self.final_norm(x), self.token_embedding.weight)
        return logits, tuple(presents) if use_cache else None
