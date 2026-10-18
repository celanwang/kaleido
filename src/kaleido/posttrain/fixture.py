"""Explicit random tiny Qwen fixture for offline CI, never formal P4 evidence."""

import hashlib

import torch


class FixtureTokenizer:
    pad_token_id = 0
    eos_token_id = 2

    def get_chat_template(self):
        return "fixture-only-byte-chat-v1"

    def encode(self, text):
        return [b + 3 for b in text.encode()]

    def decode(self, ids, *, skip_special_tokens=False):
        return "".join(
            chr(i - 3) if i >= 3 else "" if skip_special_tokens else f"<special:{i}>" for i in ids
        )

    def apply_chat_template(
        self, messages, *, tokenize=True, add_generation_prompt=False, return_dict=True
    ):
        content = (
            "".join(m["role"] + ":" + m["content"] + "\n" for m in messages[:2]) + "assistant:"
        )
        ids = [1] + self.encode(content)
        if len(messages) > 2:
            ids += self.encode(messages[2]["content"]) + [2]
        return {"input_ids": ids} if return_dict else ids


def load_fixture(*, seed=42):
    from transformers import Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(seed)
    model = (
        Qwen2ForCausalLM(
            Qwen2Config(
                vocab_size=259,
                hidden_size=8,
                intermediate_size=16,
                num_hidden_layers=1,
                num_attention_heads=2,
                num_key_value_heads=1,
                max_position_embeddings=1024,
                pad_token_id=0,
                eos_token_id=2,
            )
        )
        .float()
        .eval()
    )
    tokenizer = FixtureTokenizer()
    return (
        model,
        tokenizer,
        {
            "kind": "random_fixture_not_pretrained",
            "chat_template_sha256": hashlib.sha256(
                tokenizer.get_chat_template().encode()
            ).hexdigest(),
            "device": "cpu",
            "dtype": "float32",
        },
    )
