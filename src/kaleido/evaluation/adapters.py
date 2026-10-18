"""Small explicit adapters; no hidden downloads or automatic protocol substitutions."""

import json
from pathlib import Path

from kaleido.evaluation.schema import digest


class ArchiveAdapter:
    def __init__(self, model_id, records, *, source, protocol):
        if len({r["id"] for r in records}) != len(records):
            raise ValueError("duplicate archived IDs")
        self.model_id = model_id
        self.records = {r["id"]: r for r in records}
        self.provenance = {
            "kind": "frozen_responses",
            "source": source,
            "records_sha256": digest(records),
        }
        self.protocol = protocol

    def generate(self, cases, protocol):
        if protocol.to_dict() != self.protocol.to_dict():
            raise ValueError("archive cannot change its generation protocol")
        return [
            {
                "id": c.id,
                "response": self.records[c.id]["response"],
                "truncated": self.records[c.id]["truncated"],
                "completion_ids": self.records[c.id].get("completion_ids", []),
            }
            for c in cases
        ]


class QwenAdapter:
    def __init__(self, model_id, model, tokenizer, state, *, source):
        self.model_id = model_id
        self.model = model
        self.tokenizer = tokenizer
        self.state = state
        self.provenance = {"kind": "native_qwen_lora", "source": source}

    def generate(self, cases, protocol):
        from kaleido.posttrain.model import load_adapter, sample_completions
        from kaleido.posttrain.recipes import decode_completion, eos_ids

        load_adapter(self.model, self.state)
        records = []
        for offset in range(0, len(cases), protocol.batch_size):
            batch = cases[offset : offset + protocol.batch_size]
            prompts = [
                list(
                    self.tokenizer.apply_chat_template(
                        [
                            {"role": "system", "content": c.system},
                            {"role": "user", "content": c.prompt},
                        ],
                        tokenize=True,
                        add_generation_prompt=True,
                        return_dict=False,
                    )
                )
                for c in batch
            ]
            outputs, truncated = sample_completions(
                self.model,
                prompts,
                pad_id=self.tokenizer.pad_token_id,
                eos_ids=eos_ids(self.model, self.tokenizer),
                max_new_tokens=protocol.max_new_tokens,
                temperature=protocol.temperature,
                seed=protocol.seed,
            )
            for case, prompt, ids, cut in zip(batch, prompts, outputs, truncated, strict=True):
                records.append(
                    {
                        "id": case.id,
                        "response": decode_completion(self.model, self.tokenizer, ids),
                        "completion_ids": ids,
                        "truncated": cut,
                        "prompt_tokens": len(prompt),
                        "generated_tokens": len(ids),
                    }
                )
        return records


class DecoderLikelihoodAdapter:
    def __init__(self, model_id, weights_path, tokenizer_path):
        self.model_id = model_id
        self.weights_path = Path(weights_path)
        self.tokenizer_path = Path(tokenizer_path)

    def evaluate(self, documents, *, split="test"):
        from kaleido.lm.tokenizer import ByteBPE
        from kaleido.lm.training import TokenCorpus, evaluate, load_weights

        tokenizer = ByteBPE.load(self.tokenizer_path)
        corpus = TokenCorpus(documents, tokenizer)
        model = load_weights(self.weights_path, device="cpu")
        metrics, detail, losses = evaluate(model, corpus, split, batch_size=16)
        mapping = {d["id"]: d for d in corpus.splits[split]}
        rows = [
            {
                "id": name,
                "model_id": self.model_id,
                "nll_sum": value["nll_sum"],
                "tokens": value["target_tokens"],
                "utf8_bytes": len(mapping[name]["text"].encode()),
            }
            for name, value in detail["documents"].items()
        ]
        return {
            "model_id": self.model_id,
            "metrics": metrics,
            "documents": rows,
            "token_losses": losses,
            "tokenizer_sha256": digest(json.loads(self.tokenizer_path.read_text())),
            "weight_arrays": json.loads(self.weights_path.with_suffix(".json").read_text())[
                "arrays"
            ],
            "protocol": "chapter-local blocks; shifted targets once; EOS included; padding ignored",
        }
