# Auditable Hybrid Inference for GIS Reasoning

## Reproduce the revision results

The manuscript appendix retains supporting scientific results and figures. Installation, replay and integrity-check instructions are collected here. The [supplementary provenance record](docs/APPENDIX_PROCEDURES_20261016.md) preserves longer descriptions compressed from the paper.

1. Download the [pinned October 5 archive](https://github.com/WendongYao/Auditable-Hybrid-Inference-for-GIS-reasoning/raw/8124b2a9d2d98b7c39f7176c1e89b925836da572/releases/egmsqa_revision_replay_20261005.zip).
2. Check the archive SHA256 before extraction. Expected value: `620499a33580d45dc977abb46d0cd547a6ae667d385639bf72783d90bd32ebdb`.

   ```powershell
   Get-FileHash ./egmsqa_revision_replay_20261005.zip -Algorithm SHA256
   ```

3. Extract a **separate fresh copy for each replay command** below. From the extracted bundle directory, create a Python 3.12 environment and install pinned replay dependencies:

   ```sh
   python -m venv .venv
   # Windows:
   .venv/Scripts/python -m pip install -r requirements-replay.txt
   # macOS/Linux: substitute .venv/bin/python in all commands.
   ```

4. Choose the replay that answers your question:

   | Command (run inside a fresh bundle extraction) | What it checks |
   | --- | --- |
   | `.venv/Scripts/python replay.py --mode cases` | Four saved audit paths |
   | `.venv/Scripts/python replay.py --mode all` | 180 matched deterministic answers, four cases and five tables |
   | `.venv/Scripts/python replay_evidence.py` | Evidence comparison, reference-input isolation, prompt reconstruction and sensitivity outputs |
   | `.venv/Scripts/python replay_interventions.py` | Eight intervention policies over five saved settings (5,076 questions), the 360-question condition audit and calendar diagnostics |

No API key or new model call is used. The replay environment was verified with Python 3.12.14. A successful replay confirms the archived outputs; it does not independently validate geological events, make repeated perturbations independent observations, or reproduce all historical experiments. Fresh inference is separate and can vary even at temperature zero. Exact prompts, retained answer text, finish reasons, rule records, manifests and export-transformation notes are inside the archive. Opaque provider continuation signatures are omitted.

**Scoring-version notice (16 October 2026).** Use the corrected October 16 archive below for current manuscript results. Earlier dated archives remain immutable historical records and retain earlier scores. The new evaluation fixes inclusive decimal tolerance comparisons and binding of alternative numeric targets; neither is a method improvement.

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

## Intervention and calendar replay extension (2026-10-05)

The [October 5 archive](releases/egmsqa_revision_replay_20261005.zip) extends the earlier package with frozen 360-question inputs and saved responses, B/F condition checks, retrospective simple controls on five stored settings, C1 calendar diagnostics, and development/execution records. Verify its [SHA256](releases/egmsqa_revision_replay_20261005.sha256).

From a fresh extraction, run `python -m pip install -r requirements-replay.txt`, then `python replay_interventions.py`. Expected output is PASS for 5,076 questions across the stored settings, all simple-control predictions/metrics, 360 independent condition checks and calendar tables. These are not 5,076 independent locations. E/BF/Q-check/H-direct give 309/323/326/323 correct on the 360-question setting; the added simple controls are retrospective, not newly frozen validation. No model calls or API keys are used.

The older `replay.py --mode all` and `replay_evidence.py` commands remain available; use a separate fresh extraction for each command. Provider continuation signatures are omitted from the new response exports; visible response text, finish reasons and parsed fields are retained, with original local file hashes recorded in PROVIDER_EXPORT_NOTES.json. The archive remains a curated subset of the research, not a full historical workspace dump.

## Corrected final-revision replay (2026-10-16)

Download [the corrected replay archive](releases/egmsqa_revision_replay_20261016.zip) and verify its [SHA256](releases/egmsqa_revision_replay_20261016.sha256). It supersedes earlier accuracy summaries for the included studies. No model calls, predictions, prompts, selection or intervention thresholds changed. Numeric tolerance comparisons use decimal arithmetic; the alternative-label adapter now uses the intended alternative numeric answer instead of the original numeric target (111 B2 and 636 C2 records).

Create an isolated Python environment and install `requirements-replay.txt`. Use a separate fresh archive extraction for each replay command:

```sh
python scripts/test_decimal_scoring.py
python replay.py --mode all
python replay_evidence.py
python replay_interventions.py
python replay_alternative_labels.py
```

Current E/BF/Q-check/H-direct counts are **314/328/331/328 out of 360**. On the 252-question development audit, evidence corrects 21 errors and introduces 6. The matched GPT Full score is **56.11%**, below the same-tool deterministic control's **61.67%**. Alternative step-label macro agreement is **42.52% Direct, 41.49% Full, 68.50% deterministic baseline**. These unfavorable findings remain in the manuscript. Labels measure operational agreement, not independent physical truth; the simpler controls are retrospective.

The bundle includes corrected historical summaries, per-question decimal correction records, 3,027 paired alternative-label records and 24 system/definition checks. Original freeze documents retain historical numbers. The package is a curated reproducibility release, not the complete historical workspace. No credentials are needed.

Public-download verification of pinned commit `81bcfe3` passed all four replay commands and all 5,331 manifest entries in a new isolated environment. See [the verification record](releases/egmsqa_revision_replay_20261016.verification.json).
