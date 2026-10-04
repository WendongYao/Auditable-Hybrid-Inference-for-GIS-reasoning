# Auditable Hybrid Inference for GIS Reasoning

This repository contains the core code used in the EGMSQA study on auditable hybrid inference for GIS-oriented spatio-temporal reasoning over EGMS ground-displacement time series.

The release is intentionally compact. It includes the dataset-construction script used for the released benchmark, the prompt renderers, the deterministic task-specific analyzers, the task-wise merge and gate logic, the scorer, and the main experiment / analysis scripts used in the paper revision. Large data files, score artifacts, plots, temporary work directories, and provider-specific batch runners are not included.

## Repository layout

- `scripts/`
  - benchmark construction, prompt rendering, deterministic analyzers, hybrid orchestration, scoring, and analysis scripts
- `baselines/`
  - baseline detector helper retained for reference
- `docs/`
  - code index and experiment map for the paper
- `data/`
  - placeholder directory; the actual JSONL datasets are not redistributed in this repository

## Core code paths

- Dataset construction
  - `scripts/egms_autolabel_trend_qa_v5_8.py`
- Prompt rendering
  - `scripts/render_prompt.py`
  - `scripts/render_prompt_routeA_a3.py`
- Deterministic analyzers
  - `scripts/predict_b1_from_series.py`
  - `scripts/predict_b2_from_series.py`
  - `scripts/predict_c1_from_series.py`
  - `scripts/predict_c1_none_guard_from_series.py`
  - `scripts/predict_c2_from_routeA_a3_prompt.py`
- Task-wise merge and gate
  - `scripts/merge_preds_taskwise.py`
  - `scripts/merge_preds_taskwise_conditional.py`
  - `scripts/apply_gate_consistency_v3.py`
  - `scripts/apply_gate_v3_consistency.py`
- Baseline and local hybrid runs
  - `scripts/run_baseline_v3.py`
  - `scripts/run_llm_hf_local.py`
  - `scripts/run_local_hybrid_once.py`
  - `scripts/run_b1_local_hybrid_selected.py`
- Evaluation and analysis
  - `scripts/score.py`
  - `scripts/build_natural_subset.py`
  - `scripts/build_robustness_subset.py`
  - `scripts/summarize_robustness_scores.py`
  - `scripts/analyze_eval_slices.py`
  - `scripts/run_spatial_priority_a123.py`
  - `scripts/run_spatial_b3_package.py`
  - `scripts/compare_hybrid_vs_rules.py`
  - `scripts/bootstrap_ci_main_v1.py`
  - `scripts/make_acl_tables_from_scores_v3.py`

## Environment

The local experiments in the paper revision were run in a Conda environment with Python 3.10+ and a CUDA-enabled PyTorch installation. A minimal dependency list is provided in `requirements.txt`.

For local HF runs, install a CUDA-compatible build of `torch` first, then install the remaining packages:

```bash
pip install -r requirements.txt
```

Optional packages used only for local quantized inference:

- `accelerate`
- `bitsandbytes`

## Expected data format

Most scripts operate on EGMSQA-style JSONL files where each line contains at least:

- `id`
- `task_name`
- `question`
- `context.series`
- `context.static`
- `eval`

The release does not bundle the full benchmark data. See `data/README.md` for the expected file roles.

## Minimal workflows

### 1. Build labels / released benchmark rows

```bash
python scripts/egms_autolabel_trend_qa_v5_8.py --help
```

### 2. Render prompts

```bash
python scripts/render_prompt_routeA_a3.py --help
```

### 3. Run the deterministic baseline

```bash
python scripts/run_baseline_v3.py --help
```

### 4. Run a local HF front end and the full hybrid chain

```bash
python scripts/run_local_hybrid_once.py --help
```

### 5. Score predictions

```bash
python scripts/score.py --help
```

### 6. Run robustness / natural-prior / spatial analyses

```bash
python scripts/build_robustness_subset.py --help
python scripts/build_natural_subset.py --help
python scripts/run_spatial_priority_a123.py --help
python scripts/run_spatial_b3_package.py --help
```

## Notes on this release

- This repository is a curated code release, not a verbatim dump of the working directory.
- Provider-specific API runners, temporary scripts, and large intermediate artifacts were intentionally omitted.
- The scorer retained here is `scripts/score.py`, which matches the current release workflow discussed in the manuscript revision.

## Paper-facing code map

See `docs/CODE_INDEX.md` for a section-by-section mapping between manuscript claims and the released scripts.

## Revision replay package (2026-10-04)

Download [the curated offline replay bundle](releases/egmsqa_revision_replay_20261004.zip) and verify its [SHA256](releases/egmsqa_revision_replay_20261004.sha256). It includes the 30-series GPT matched comparison, 15-series conditional sensitivity experiment, four audit cases, and the 42-series Gemini development audit. These are different experimental settings, not a cross-model leaderboard or independent deployment validation.

Extract the archive, create a Python 3.12 environment, and run:

```sh
python -m pip install -r requirements-replay.txt
python replay.py --mode all
```

For the reference-input isolation and intervention/sensitivity extension, use a fresh extraction and run `python replay_evidence.py`. No API key or model call is required. The first command verifies 180 reconstructed deterministic answers, four audit cases and five tables; the second verifies the definition reference (252/252), 504 frozen prompts, paired intervention tables, and cached fallback sensitivity. Full instructions, limitations, exact prompts, saved responses and per-file hashes are inside the archive. This curated package does not contain every historical experiment.
