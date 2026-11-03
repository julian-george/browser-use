# Beaker: browser-use Agent on BU Bench V2.1

Optimizes this fork's Python `Agent` harness (system prompt, per-step messages, DOM
serialization, tools, loop policy) on [BU Bench V2.1](https://github.com/browser-use/benchmark)
(release v2.1.1, commit `af6c7f7`). Each case is scored by the benchmark's own findings judge:
the weighted rubric score, 0–100 per task with partial credit, reported here as 0–1.

## Winning changes must be general

This fork is public. Accept only harness changes that would help on any website and any task.
A prompt, tool description or rule that mentions a benchmark task, site, answer or rubric item
leaks benchmark text into a public PR, and it is overfitting. Reject those candidates.

## Nothing from the benchmark is committed

- The benchmark asks that decrypted tasks and rubrics stay private, and its repo has no
  license. `bu_bench.py` downloads the pinned `BU_Bench_V2.enc` and judge files at runtime into
  a temp cache, checks their pinned sha256s, and imports them. Their code is not copied here.
- Dataset rows hold only ids and sha stamps (`input = {task_id, revision, task_sha}`,
  `expected = {rubric_sha, weights_sha}`). `load_cases` decrypts in memory and checks that the
  stamps match the row. The rubric goes only to the scorer, never to the candidate.
- `splits.json`: fixed-seed 140 train / 60 test task ids, plus a 6 + 3 quickstart subset.
  `build_dataset.py {full|quickstart} {smoke|upload}` builds rows in a temp dir.

## How a case runs

This mirrors the benchmark's `run_eval.run_task` (`--executor browser-use` path):

1. Each case gets a fresh headless Chromium with a fresh profile, stopped in `finally`.
2. `Agent(..., use_judge=False)` and `max_steps=100`. Only the benchmark judge scores.
3. Timeout is `task_timeout_s`, default 1800 (upstream uses 3600). Set it with launch `extra`.
   After a timeout or exception, the partial history is still judged.
4. The judge evidence (steps, up to 50 screenshots, output files) travels in
   `CaseResult.context`. It is capped at 4 MiB (upstream allows 8 MB of screenshots) because
   hosted judge calls go straight to Beaker's gateway, which rejects larger request bodies.
5. If no step was produced (browser launch, provider or CDP failure), the case raises
   `RetryableCaseError`. A blocked site or a CAPTCHA is a real outcome, and the rubric scores it.

## Models

- Agent: `gpt-6-luna` with upstream's executor settings, except `medium` reasoning instead of
  `xhigh`, which took about 16 s per step. This applies when the run selects no model. A
  selected model goes through `inference_target`, which also covers page extraction.
- Judge: `openai:gpt-6-luna` through `scoring_inference_target()` with upstream's judge settings.
  `scorer_model` in `beaker.yaml` keeps it the same for the baseline and every candidate.
- Upstream's published scores use `gpt-5.6-luna` as the judge, so scores here are **not
  directly comparable** to theirs.

## Launch

```bash
beaker run trigger --dataset <name@revision>
```

The judge is pinned by `scorer_model: openai:gpt-6-luna` on the integration in `beaker.yaml`.

The live web makes scores noisy. Compare candidates against a repeated baseline, not a single run.
