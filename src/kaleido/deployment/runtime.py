"""Load a verified export and serve bounded greedy JSON-lines requests locally."""

import json
import math
from pathlib import Path

from kaleido.lm.tokenizer import EOS, ByteBPE
from kaleido.repro.integrity import verify_index
from kaleido.serialization import load_object

MAX_LINE_BYTES = 65536
MAX_PROMPT_BYTES = 16384
MAX_NEW_TOKENS = 128


class Predictor:
    def __init__(self, directory):
        import torch

        directory = Path(directory)
        manifest = load_object(directory / "manifest.json")
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
        if not isinstance(manifest["files"], dict):
            raise ValueError("bundle files must be a mapping")
        expected = {"model.pt2", "tokenizer.json", "validation.json", "examples.json"}
        if set(manifest["files"]) != expected or {
            p.name for p in directory.iterdir()
        } != expected | {"manifest.json"}:
            raise ValueError("bundle file set differs from the declared format")
        result = verify_index(directory, {"schema_version": 1, "files": manifest["files"]})
        if not result["ok"]:
            raise ValueError(f"bundle integrity failed: {result['errors']}")
        runtime = manifest["runtime"]
        if not isinstance(runtime, dict) or set(runtime) != {
            "torch",
            "device",
            "batch_size",
            "decoding",
        }:
            raise ValueError("invalid bundle runtime fields")
        if (
            type(runtime["batch_size"]) is not int
            or runtime["torch"] != torch.__version__.split("+")[0]
            or runtime["device"] != "cpu"
            or runtime["batch_size"] != 1
            or runtime["decoding"] != "greedy_cropped_window"
        ):
            raise ValueError("bundle runtime does not match this supported CPU/PyTorch protocol")
        config = manifest["model_config"]
        if not isinstance(config, dict) or set(config) != {
            "vocab_size",
            "context_length",
            "width",
            "layers",
            "heads",
            "dropout",
        }:
            raise ValueError("invalid model configuration fields")
        if (
            any(
                type(config[k]) is not int or config[k] < 1
                for k in ("vocab_size", "context_length", "width", "layers", "heads")
            )
            or not 8 <= config["context_length"] <= 4096
            or config["width"] % config["heads"]
            or type(config["dropout"]) not in (int, float)
            or not math.isfinite(config["dropout"])
            or not 0 <= config["dropout"] < 1
        ):
            raise ValueError("invalid exported model dimensions")
        self.context_length = config["context_length"]
        validation = load_object(directory / "validation.json")
        examples = load_object(directory / "examples.json")
        if (
            validation.get("passed") is not True
            or type(validation.get("generation_cases")) is not int
            or validation["generation_cases"] < 1
            or not isinstance(examples.get("cases"), list)
            or len(examples["cases"]) != validation["generation_cases"]
        ):
            raise ValueError("bundle has no completed generation validation")
        self.tokenizer = ByteBPE.load(directory / "tokenizer.json")
        if manifest["model_config"]["vocab_size"] != len(self.tokenizer):
            raise ValueError("tokenizer vocabulary differs from the exported graph")
        exported = torch.export.load(directory / "model.pt2")
        state = exported.state_dict
        dimensions = {
            "model.token_embedding.weight": (len(self.tokenizer), config["width"]),
            "model.position_embedding.weight": (self.context_length, config["width"]),
        }
        if any(k not in state or tuple(state[k].shape) != shape for k, shape in dimensions.items()):
            raise ValueError("manifest dimensions disagree with the exported embeddings")
        if any(
            not isinstance(v, torch.Tensor) or v.device.type != "cpu" or v.dtype != torch.float32
            for v in state.values()
        ):
            raise ValueError("bundle must contain CPU float32 parameters")
        self.model = exported.module()
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
