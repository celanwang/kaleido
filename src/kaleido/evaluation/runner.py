import time

from kaleido.evaluation.schema import align_predictions, digest, freeze_cases
from kaleido.evaluation.scoring import score, stratified_summary


def evaluate(adapter, cases, protocol):
    cases = list(cases)
    case_sha = freeze_cases(cases)
    start = time.perf_counter()
    predictions = align_predictions(cases, adapter.generate(cases, protocol))
    if freeze_cases(cases) != case_sha:
        raise ValueError("adapter mutated frozen case metadata")
    records = []
    for case, prediction in zip(cases, predictions, strict=True):
        if not isinstance(prediction.get("truncated"), bool):
            raise ValueError("prediction must declare truncation")
        records.append(
            {
                **prediction,
                "model_id": adapter.model_id,
                "suite": case.suite,
                "family": case.family,
                "difficulty": case.difficulty,
                **score(case, prediction["response"], truncated=prediction["truncated"]),
            }
        )
    identity = {
        "model_id": adapter.model_id,
        "source": adapter.provenance,
        "case_sha256": case_sha,
        "protocol": protocol.to_dict(),
        "protocol_sha256": digest(protocol.to_dict()),
    }
    return {
        "identity": identity,
        "identity_sha256": digest(identity),
        "summary": stratified_summary(records),
        "records": records,
        "wall_seconds": time.perf_counter() - start,
    }
