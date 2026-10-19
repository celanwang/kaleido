"""Export an actually trained P3 decoder, with eager/graph numerical and decoding checks."""

import shutil
import tempfile
from dataclasses import asdict
from pathlib import Path

from kaleido.experiments import provenance, write_json
from kaleido.repro.integrity import file_info


def export_decoder(weights, tokenizer_path, output, *, root):
    import torch
    from torch.nn import functional as F

    from kaleido.lm.inference import generate
    from kaleido.lm.tokenizer import EOS, ByteBPE
    from kaleido.lm.training import load_weights

    class ForwardGraph(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, tokens):
            # Input range/shape validation stays in the runtime, outside the tensor graph.
            model = self.model
            x = model.token_embedding(tokens) + model.position_embedding(
                torch.arange(tokens.shape[1], device=tokens.device)
            )
            for block in model.blocks:
                x = block(x)[0]
            return F.linear(model.final_norm(x), model.token_embedding.weight)

    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("export destination exists; choose a new directory")
    weights, tokenizer_path = Path(weights), Path(tokenizer_path)
    model = load_weights(weights, device="cpu").requires_grad_(False)
    tokenizer = ByteBPE.load(tokenizer_path)
    if model.config.vocab_size != len(tokenizer) or model.config.context_length < 8:
        raise ValueError("export requires a matching tokenizer and context length >=8")
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with tempfile.TemporaryDirectory(prefix=".kaleido-export-", dir=output.parent) as temporary:
            staging = Path(temporary) / "bundle"
            staging.mkdir()
            graph = torch.export.export(
                ForwardGraph(model).eval(),
                (torch.ones(1, 8, dtype=torch.long),),
                strict=True,
                dynamic_shapes=(
                    {1: torch.export.Dim("time", min=1, max=model.config.context_length)},
                ),
            )
            torch.export.save(graph, staging / "model.pt2")
            shutil.copyfile(tokenizer_path, staging / "tokenizer.json")
            loaded = torch.export.load(staging / "model.pt2").module()
            errors = []
            generator = torch.Generator().manual_seed(2026)
            lengths = sorted(
                {1, 2, 7, model.config.context_length - 1, model.config.context_length}
            )
            for length in lengths:
                ids = torch.randint(len(tokenizer), (1, length), generator=generator)
                with torch.inference_mode():
                    expected = model(ids)[0]
                    actual = loaded(ids)
                error = float((expected - actual).abs().max())
                torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
                if not torch.equal(actual.argmax(-1), expected.argmax(-1)):
                    raise AssertionError("export changed greedy decisions")
                errors.append(
                    {"length": length, "max_abs_logit_error": error, "greedy_agreement": True}
                )
            examples = []
            for prompt in ("Alice was", "The little girl", "Once upon a time " * 20):
                ids = tokenizer.encode(prompt)
                expected, _ = generate(
                    model, ids, max_new_tokens=12, temperature=0, use_cache=False, eos_id=EOS
                )
                tokens = list(ids)
                completion = []
                with torch.inference_mode():
                    for _ in range(12):
                        following = int(
                            loaded(
                                torch.tensor(
                                    [tokens[-model.config.context_length :]], dtype=torch.long
                                )
                            )[0, -1].argmax()
                        )
                        completion.append(following)
                        tokens.append(following)
                        if following == EOS:
                            break
                if completion != expected[len(ids) :]:
                    raise AssertionError("export changed cropped-window generation")
                examples.append(
                    {
                        "request": {"prompt": prompt, "max_new_tokens": 12},
                        "completion_ids": completion,
                        "completion": tokenizer.decode(completion, errors="replace"),
                    }
                )
            write_json(
                staging / "validation.json",
                {
                    "logits": errors,
                    "generation_cases": len(examples),
                    "tolerance": {"atol": 1e-5, "rtol": 1e-5},
                    "passed": True,
                    "scope": (
                        "CPU float32 dynamic full-forward export; "
                        "not cache, quantization or a speedup claim"
                    ),
                },
            )
            write_json(staging / "examples.json", {"cases": examples})
            write_json(
                staging / "manifest.json",
                {
                    "schema_version": 1,
                    "kind": "kaleido_decoder_export",
                    "status": "completed",
                    "runtime": {
                        "torch": torch.__version__.split("+")[0],
                        "device": "cpu",
                        "batch_size": 1,
                        "decoding": "greedy_cropped_window",
                    },
                    "model_config": asdict(model.config),
                    "source": {
                        "weights": file_info(weights),
                        "metadata": file_info(weights.with_suffix(".json")),
                        "tokenizer": file_info(tokenizer_path),
                        "environment": provenance(Path(root)),
                    },
                    "files": {p.name: file_info(p) for p in sorted(staging.iterdir())},
                },
            )
            # Loader and serving adapter are independently checked before publishing the directory.
            from kaleido.deployment.runtime import Predictor

            predictor = Predictor(staging)
            for example in examples:
                value = predictor.predict(example["request"])
                if value["completion_ids"] != example["completion_ids"]:
                    raise AssertionError("runtime disagrees with validated export")
            staging.rename(output)
    finally:
        torch.set_num_threads(previous)
    return output
