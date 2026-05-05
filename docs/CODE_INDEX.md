# Code index

This note maps the released code to the main method and experiment blocks discussed in the paper revision.

## Benchmark construction

- `scripts/egms_autolabel_trend_qa_v5_8.py`
  - train-prefix label construction
  - trend / seasonality / step labels
  - released `eval` fields and tolerances
  - conditional step-task base series view

## Query rendering

- `scripts/render_prompt.py`
  - compact prompt renderer with summary + anchor blocks
- `scripts/render_prompt_routeA_a3.py`
  - routeA-a3 rendering with step-candidate evidence blocks

## Deterministic analyzers

- `scripts/predict_b1_from_series.py`
  - seasonality-presence decision
- `scripts/predict_b2_from_series.py`
  - harmonic seasonal-amplitude estimator
- `scripts/predict_c1_from_series.py`
  - direct step-time predictor
- `scripts/predict_c1_none_guard_from_series.py`
  - one-sided weak-evidence `none` guard for C1
- `scripts/predict_c2_from_routeA_a3_prompt.py`
  - step-magnitude extraction from structured routeA-a3 evidence

## Hybrid composition

- `scripts/merge_preds_taskwise.py`
  - task-wise unconditional merge
- `scripts/merge_preds_taskwise_conditional.py`
  - task-wise conditional merge using `use_alt`
- `scripts/apply_gate_consistency_v3.py`
  - full gate logic
- `scripts/apply_gate_v3_consistency.py`
  - gate application wrapper used in the experiment chain

## Baselines and local inference

- `scripts/run_baseline_v3.py`
  - deterministic reference pipeline
- `scripts/step_residual_detector_v2.py`
  - step detector used by `run_baseline_v3.py`
- `scripts/run_llm.py`
  - shared parsing / validation helpers and legacy API runner
- `scripts/run_llm_hf_local.py`
  - local Hugging Face inference entry point
- `scripts/run_local_hybrid_once.py`
  - single-run local hybrid pipeline
- `scripts/run_b1_local_hybrid_selected.py`
  - selected robustness reruns for the local hybrid setup

## Scoring

- `scripts/score.py`
  - current scorer used in the manuscript revision workflow

## Additional experiments in the revision

- `scripts/build_robustness_subset.py`
  - robustness subset construction
- `scripts/summarize_robustness_scores.py`
  - robustness summary tables
- `scripts/build_natural_subset.py`
  - natural-prior subset construction
- `scripts/analyze_eval_slices.py`
  - tile / regime / metadata slice analysis
- `scripts/run_spatial_priority_a123.py`
  - A1/A2/A3 spatial package
- `scripts/run_spatial_b3_package.py`
  - B3 cross-tile spatial package
- `scripts/compare_hybrid_vs_rules.py`
  - hybrid vs deterministic comparison
- `scripts/bootstrap_ci_main_v1.py`
  - bootstrap confidence intervals
- `scripts/make_acl_tables_from_scores_v3.py`
  - paper-facing aggregate table generation

## Intentionally omitted from the curated release

- large score directories and rendered artifacts
- provider-specific batch scripts for remote proprietary APIs
- temporary debugging scripts and one-off probes
- manuscript files themselves
