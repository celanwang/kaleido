import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from kaleido.posttrain.model import (  # noqa:E402
    LoRALinear,
    adapter_state,
    completion_log_probs,
    inject_lora,
    load_adapter,
    pack_completions,
    prompt_tokens,
    read_adapter,
    sample_completions,
    save_adapter,
    sft_loss,
    sft_tokens,
)


def tiny():
    torch.manual_seed(7)
    torch.set_num_threads(1)
    config = transformers.Qwen2Config(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
        pad_token_id=0,
        eos_token_id=2,
        attn_implementation="sdpa",
    )
    return transformers.Qwen2ForCausalLM(config).float().eval()


def test_lora_zero_init_forward_gradients_and_frozen_base():
    torch.manual_seed(3)
    base = torch.nn.Linear(4, 3)
    layer = LoRALinear(base, rank=2, alpha=4)
    x = torch.randn(5, 4)
    torch.testing.assert_close(layer(x), base(x), atol=0, rtol=0)
    before = base.weight.detach().clone()
    layer(x).square().sum().backward()
    assert layer.lora_a.grad.abs().sum() == 0 and layer.lora_b.grad.abs().sum() > 0
    assert base.weight.grad is None
    torch.optim.AdamW([layer.lora_a, layer.lora_b], lr=0.01).step()
    torch.testing.assert_close(base.weight, before, atol=0, rtol=0)


def test_completion_shift_padding_eos_mask_and_likelihood_match_full_native_forward():
    model = tiny()
    inject_lora(model, rank=2, alpha=4)
    batch = pack_completions([[3, 4, 5], [6]], [[7, 2], [8, 9, 2]], pad_id=2)
    actual = completion_log_probs(model, batch)
    full = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        position_ids=batch["position_ids"],
        use_cache=False,
    ).logits
    start = batch["prompt_width"] - 1
    expected = (
        full[:, start : start + 3]
        .log_softmax(-1)
        .gather(-1, batch["targets"].unsqueeze(-1))
        .squeeze(-1)
    )
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
    assert batch["mask"].tolist() == [[True, True, False], [True, True, True]]
    loss = sft_loss(actual, batch["mask"])
    loss.backward()
    assert any(p.grad is not None for p in model.parameters() if p.requires_grad)
    assert all(p.grad is None for p in model.parameters() if not p.requires_grad)


def test_native_generation_padding_batch_equivalence_and_checkpoint(tmp_path):
    model = tiny()
    prompts = [[3, 4, 5], [6, 7]]
    sequences, truncated = sample_completions(
        model, prompts, pad_id=0, eos_ids=[2], max_new_tokens=5
    )
    for prompt, actual in zip(prompts, sequences, strict=True):
        generated = model.generate(
            torch.tensor([prompt]),
            max_new_tokens=5,
            do_sample=False,
            eos_token_id=2,
            pad_token_id=0,
        ).tolist()[0][len(prompt) :]
        assert generated == actual
    inject_lora(model, rank=2, alpha=4)
    state = adapter_state(model)
    path = tmp_path / "adapter.npz"
    save_adapter(path, model, metadata={"rank": 2})
    restored, metadata = read_adapter(path)
    load_adapter(model, restored)
    for name, value in adapter_state(model).items():
        np.testing.assert_array_equal(value.numpy(), state[name].numpy())
    assert metadata["rank"] == 2 and len(truncated) == 2
    bad = dict(state)
    bad[next(iter(bad))] = torch.ones(1)
    with pytest.raises(ValueError):
        load_adapter(model, bad)


def test_chat_template_explicit_list_and_eos_boundary():
    class NativeTemplate:
        eos_token_id = 2

        def apply_chat_template(
            self, messages, *, tokenize, return_dict=True, add_generation_prompt=False
        ):
            ids = [1, 11, 22, 3] if add_generation_prompt else [1, 11, 22, 3, 10, 20, 2, 30]
            return {"input_ids": ids} if return_dict else ids

    tokenizer = NativeTemplate()
    row = {"prompt": "task", "completion": "<answer>1</answer>"}
    assert prompt_tokens(tokenizer, row) == [1, 11, 22, 3]
    assert sft_tokens(tokenizer, row) == ([1, 11, 22, 3], [10, 20, 2])
