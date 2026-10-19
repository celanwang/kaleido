# Frozen public evaluation source

`raw/gsm8k-test.jsonl` is the complete upstream test file at the exact Git revision in `sources.json`.
The upstream MIT license is retained as `raw/GSM8K-LICENSE.txt`; source bytes are hash checked.
The runner selects 32 integer-answer rows with a fixed seed, without looking at model outputs.
Only questions reach the model; reference solution text is not included in prompts.

This is a small direct-answer subset with a custom `<answer>` protocol and fixed token budget.
It is not the canonical full-GSM8K chain-of-thought evaluation and cannot be compared directly to leaderboard scores.
No claim is made that Qwen pretraining did not include the public dataset.

The additional novel-structure and reference-color cases are generated locally from frozen rules/seeds.
All P5 cases are evaluation-only; no P4 model or hyperparameter is updated after seeing them.

The color prompt/parser contract was repaired after inspecting the initial run. Both versions are retained
in the P5 protocol audit; the repaired results are descriptive, not untouched confirmatory testing.
