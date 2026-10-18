import pytest

from kaleido.posttrain.tasks import (
    FAMILIES,
    dataset_digest,
    independent_oracle,
    make_tasks,
    score_response,
    training_reward,
    wilson_interval,
)


def test_structural_splits_frozen_data_and_independent_oracles():
    tasks = make_tasks()
    assert dataset_digest(tasks) == dataset_digest(make_tasks())
    assert len(tasks) == 192 and len({r["id"] for r in tasks}) == 192
    for row in tasks:
        assert row["family"] in FAMILIES[row["split"]]
        assert independent_oracle(row) == row["answer"]
        assert 0 <= row["answer"] < row["operands"]["m"]
    assert len(set.union(*(set(v) for v in FAMILIES.values()))) == 8


def test_nonexecuting_strict_verifier_rejects_reward_hacks_and_truncation():
    row = make_tasks(train_per_family=1, heldout_per_family=1)[0]
    answer = row["completion"]
    assert score_response(row, answer)["reward"] == 1
    for value in (
        answer + " extra",
        answer + answer,
        "<answer>01</answer>",
        '<answer>__import__("os")</answer>',
        "<answer>" + "9" * 10000 + "</answer>",
    ):
        assert score_response(row, value)["reward"] == 0
    assert score_response(row, answer, truncated=True)["reward"] == 0
    assert score_response(row, "reasoning " + answer)["numerical_correct"]
    assert score_response(row, str(row["answer"]))["reward"] == 0
    assert training_reward(row, answer) == 1
    changed = {**row, "answer": (row["answer"] + 1) % row["operands"]["m"]}
    assert training_reward(changed, answer) == 0
    assert score_response(changed, answer)["reward"] == 1


def test_wilson_and_invalid_config_boundaries():
    assert wilson_interval(0, 10)[0] == 0
    assert wilson_interval(10, 10)[1] == 1
    assert wilson_interval(5, 10)[0] < 0.5 < wilson_interval(5, 10)[1]
    with pytest.raises(ValueError):
        make_tasks(train_per_family=0)
    with pytest.raises(ValueError):
        wilson_interval(1, 0)
