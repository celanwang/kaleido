"""Load a verified export and serve bounded greedy JSON-lines requests locally."""

import json
from pathlib import Path

from kaleido.lm.tokenizer import EOS, ByteBPE
from kaleido.repro.integrity import verify_index

MAX_LINE_BYTES = 65536
MAX_PROMPT_BYTES = 16384
MAX_NEW_TOKENS = 128


class Predictor:
    def __init__(self, directory):
        import torch

        directory = Path(directory)
        manifest = json.loads((directory / "manifest.json").read_text())
        if (
            manifest.get("schema_version") != 1
            or manifest.get("kind") != "kaleido_decoder_export"
            or manifest.get("status") != "completed"
        ):
            raise ValueError("unsupported or incomplete inference bundle")
        if set(manifest) != {
            "schema_version",
            "kind",
            "status",
            "runtime",
            "model_config",
            "source",
            "files",
        }:
            raise ValueError("bundle manifest fields differ from the declared format")
        expected = {"model.pt2", "tokenizer.json", "validation.json", "examples.json"}
        if set(manifest["files"]) != expected or {
            p.name for p in directory.iterdir()
        } != expected | {"manifest.json"}:
            raise ValueError("bundle file set differs from the declared format")
        result = verify_index(directory, {"schema_version": 1, "files": manifest["files"]})
        if not result["ok"]:
            raise ValueError(f"bundle integrity failed: {result['errors']}")
        runtime = manifest["runtime"]
        if (
            runtime["torch"] != torch.__version__.split("+")[0]
            or runtime["device"] != "cpu"
            or runtime["batch_size"] != 1
            or runtime["decoding"] != "greedy_cropped_window"
        ):
            raise ValueError("bundle runtime does not match this supported CPU/PyTorch protocol")
        self.context_length = manifest["model_config"]["context_length"]
        if type(self.context_length) is not int or not 2 <= self.context_length <= 4096:
            raise ValueError("invalid exported context length")
        self.tokenizer = ByteBPE.load(directory / "tokenizer.json")
        if manifest["model_config"]["vocab_size"] != len(self.tokenizer):
            raise ValueError("tokenizer vocabulary differs from the exported graph")
        self.model = torch.export.load(directory / "model.pt2").module()
        self.manifest = manifest

    def logits(self, ids):
        import torch

        if (
            not isinstance(ids, list)
            or not 1 <= len(ids) <= self.context_length
            or any(type(t) is not int or not 0 <= t < len(self.tokenizer) for t in ids)
        ):
            raise ValueError("logits requires a fitting list of valid integer token IDs")
        with torch.inference_mode():
            result = self.model(torch.tensor([ids], dtype=torch.long))
        if result.shape != (1, len(ids), len(self.tokenizer)) or not torch.isfinite(result).all():
            raise RuntimeError("exported graph returned invalid logits")
        return result

    def predict(self, request):
        if not isinstance(request, dict) or set(request) - {"prompt", "max_new_tokens"}:
            raise ValueError("request must contain prompt and optional max_new_tokens")
        prompt = request.get("prompt")
        limit = request.get("max_new_tokens", 32)
        if (
            not isinstance(prompt, str)
            or not prompt
            or len(prompt.encode()) > MAX_PROMPT_BYTES
            or type(limit) is not int
            or not 0 <= limit <= MAX_NEW_TOKENS
        ):
            raise ValueError(
                "prompt must be non-empty bounded UTF-8 text; token budget must be 0..128"
            )
        tokens = self.tokenizer.encode(prompt)
        initial = len(tokens)
        completion = []
        for _ in range(limit):
            token = int(self.logits(tokens[-self.context_length :])[0, -1].argmax())
            completion.append(token)
            tokens.append(token)
            if token == EOS:
                break
        return {
            "status": "completed",
            "prompt_tokens": initial,
            "initial_cropped_tokens": max(0, initial - self.context_length),
            "completion_ids": completion,
            "completion": self.tokenizer.decode(completion, errors="replace"),
            "generated_tokens": len(completion),
            "stopped_on_eos": bool(completion and completion[-1] == EOS),
            "decoding": "greedy_cropped_window",
            "device": "cpu",
        }


def serve(predictor, input_stream, output_stream):
    """One result per line; malformed requests fail individually and set the process exit code."""
    failed = False
    request_id = 0
    while line := input_stream.readline(MAX_LINE_BYTES + 1):
        request_id += 1
        try:
            if len(line) > MAX_LINE_BYTES:
                while not line.endswith(b"\n"):
                    line = input_stream.readline(MAX_LINE_BYTES + 1)
                    if not line:
                        break
                raise ValueError("JSON line exceeds 65536 bytes")
            request = json.loads(line)
            result = predictor.predict(request)
        except (ValueError, TypeError, RuntimeError) as error:
            failed = True
            result = {"status": "failed", "error": type(error).__name__, "message": str(error)}
        output_stream.write(
            json.dumps({"request_id": request_id, **result}, ensure_ascii=False, allow_nan=False)
            + "\n"
        )
        output_stream.flush()
    return 1 if failed else 0
