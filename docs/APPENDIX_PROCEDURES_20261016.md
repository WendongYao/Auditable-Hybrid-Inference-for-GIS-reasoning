# Supplementary implementation and provenance record

This record preserves longer protocol descriptions moved or compressed from the manuscript on 16 October 2026. Scientific results, figures, task definitions and limitations remain in the paper. For executable commands, use the repository README and the pinned replay archive. These historical paragraphs describe the October 5 release, not a newly rerun experiment.

## 1

```tex
\section{Reproducibility details}
\label{app:repro}

\paragraph{Models and decoding.}
Recorded endpoint names and token budgets are reported in Section~\ref{sec:eval_models} and Table~\ref{tab:model_versions}. For the historical experiments, per-request prompt hashes and complete decoding settings were not retained in the saved prediction records; we do not claim identical decoding settings or immutable provider snapshots.

\paragraph{Prompts and outputs.}
Saved direct answers are retained as upstream inputs. Post-hoc predictors produce single-task JSONL files merged in the order B1, B2, C1, C2. The verified gate enforces both B1/B2 and C1/C2 none constraints. Revised artifacts are stored separately from historical outputs. The signed-output renderer is versioned, but the historical tables are not prompt-matched reruns. The separate 30-series experiment records exact new prompts and model snapshots.

\paragraph{Controlled evaluation by ID sets.}
To ensure strict comparability across variants, we evaluate every post-hoc variant on an identical per-backbone ID set derived from that backbone's base predictions (\textbf{ids\_\{tag\}.jsonl}).
Scoring is performed with \textbf{scripts/score.py} in \textbf{intersection} mode, i.e., only instances present in both gold and predictions are scored.

\paragraph{Baselines.}
We preserve historical Baseline-v3 with A1 threshold 0.5 and add a control using threshold 0.2 on the same saved slopes. Both are evaluated before and after the same deterministic updates. Their remaining differences from the released labels include observation support and estimator details.
```

## 2

```tex
Our main benchmark is the \textbf{dev\_balanced} split of \textbf{EGMSQA}, comprising 28{,}000 structured query instances. Task counts follow the scorer configuration used throughout the study, with trend tasks contributing 6{,}000 instances each and seasonality and step-related tasks contributing 4{,}000 instances each. To ensure strictly controlled comparison across framework variants, all post-hoc components are applied to exactly the same per-backbone ID set derived from the corresponding base prediction file. Scoring is performed by \textbf{scripts/score.py} in \textbf{intersection} mode, so that each comparison reflects matched prediction sets rather than changes in instance coverage. Prediction coverage differences are negligible in the main \textbf{dev\_balanced} comparisons: all reported systems produce complete predictions except \textbf{Llama4-Maverick}, which misses a single prediction (27{,}999/28{,}000) in both the direct and full-hybrid runs. Appendix Table~\ref{tab:coverage_appendix} reports the exact coverage values extracted from \textbf{scripts/score.py}.

Beyond this controlled benchmark, we also examine deployment-oriented conditions that are especially relevant to GIS use. We analyze the natural \textbf{dev.jsonl} distribution through bucket-reweighted prior-shift estimates for all complete backbones, a direct full-split rerun of \textbf{Baseline-v3} over all 797{,}106 QA instances, and a direct supporting rerun on a 49{,}998-example natural-prior subset using a fully local representative setup. In addition, we run dedicated cross-tile analyses on representative systems by EGMS tile, deformation regime, and \textbf{rmse\_mm} bin, together with error-cluster summaries that expose how local spatial behavior changes under hybrid validation.

This protocol serves two purposes. First, it isolates the effect of task-wise deterministic updates and bounded consistency control from differences in instance composition. Second, paired transitions identify which component changes each saved answer while keeping the compared records fixed. Task definitions and scoring rules are summarized in Table~\ref{tab:task_spec}.
```

## 3

```tex
The offline entry is \texttt{python replay.py --mode cases} for four audit paths, or \texttt{python replay.py --mode all} for all 180 matched questions and reconstruction of five matched/uncertainty result tables. It requires Python 3.12.14 with pinned NumPy, pandas, Matplotlib and tqdm dependencies. No API key is read. Fresh inference is distinct from replay, and provider outputs can vary even at temperature zero. The uncertainty-table replay uses saved perturbed rule outputs; the original propagation experiment reran every affected branch.

\textbf{Verified revision release.} The current curated package is available in the confirmed repository under \texttt{releases/egmsqa\_revision\_replay\_20261005.zip}, at commit \texttt{8124b2a}. We downloaded the archive from that pinned public commit, verified its SHA256 and 5,251 manifest entries, and ran all three README replay commands in a newly created Python environment. The original replay regenerated 180 deterministic answers, four cases and five tables; the evidence replay reproduced its 11 tables or prediction files. The intervention extension verified eight policies across five saved settings (5,076 questions), the 360-question audit and the calendar diagnostics. No model API was used. Provider-internal opaque thought signatures were omitted from the exported response envelopes; original-file hashes and this transformation are documented, while answer text and scoring fields are retained. The full commit, archive hash and verification record accompany the archive. This curated release does not imply reproduction of the entire historical experiment collection.
```

## 4

```tex
In addition to the main \textbf{dev\_balanced} benchmark, we conducted a supporting local validation on a natural-prior subset sampled from the full \textbf{dev.jsonl}. The purpose of this experiment is not to replace the main cross-backbone evaluation, but to test whether the same qualitative ordering of systems can still be observed when the task and tile distributions follow the natural data prior more closely.

The subset contains 49{,}998 examples grouped into 8{,}333 complete \textbf{pid} bundles, each preserving the full six-task set \textbf{A1/A2/B1/B2/C1/C2}. This construction keeps tile composition and key bucket prevalences close to the full natural split, including 0.6293 uplifting \textbf{A1} cases versus 0.6316 in the full natural set, 0.8498 seasonal \textbf{B1} cases versus 0.8500, and 0.8524 step-positive \textbf{C1} cases versus 0.8497. Because each selected \textbf{pid} contributes exactly one instance per task, macro and micro aggregation coincide on this subset.

To keep the experiment fully local and computationally manageable, we instantiated the query-facing model with \textbf{Qwen/Qwen2.5-3B-Instruct} under 4-bit quantization and reconstructed the same task-wise hybrid chain used elsewhere in the paper, including deterministic replacements for \textbf{B1/B2/C1/C2} and the final consistency layer. This experiment should therefore be interpreted as a supporting natural-prior validation under a representative local setup. It does not substitute for a full direct natural-split rerun across all main-paper backbones, but it provides direct evidence that the framework-level behavior observed on \textbf{dev\_balanced} is not solely an artifact of balanced-split evaluation.


% =========================
% 6 Framework Validation Results
% =========================
```

## 5

```tex
To localize which modules contribute most to the final gain, Table~\ref{tab:abl_components} reports cross-backbone averages over the five query-facing backbones included in the main-backbone summary: Gemini-3 Flash, Gemini-3 Pro, DeepSeek-V3.2, GPT-5.2, and Llama4-Maverick. For each backbone and each variant, we first compute macro/micro and per-task accuracies from the scored outputs of that backbone, and then report the unweighted arithmetic mean across backbones; we do not pool predictions across models before scoring. Per-backbone component summaries are reported in Appendix Table~\ref{tab:abl_per_backbone}. The seasonality-presence and seasonality-amplitude modules yield large improvements on their respective targets, while the step-magnitude module produces the dominant gains for abrupt-change quantification. The step-time control module contributes more modestly, which is consistent with the fact that step-time inference is discrete, definition-sensitive, and strongly affected by no-step handling.

This historical attribution analysis describes the saved systems and their task-specific updates. It does not establish a causal benefit from the merge order. The matched studies show that rule replacement can also remove correct LLM answers, and consistency constraints can help or harm. Evidence inputs and post-hoc interventions must therefore be evaluated separately (Section~\ref{sec:evidence_audit}).
```