# PROTOCOL.md — Locked Analytical Decisions

This document records every fixed constant, design choice, and load-bearing parameter
in the VLM-fairness study.  These values are frozen at time of submission.  Do **not**
change them without updating the version tag below and adding an entry to the changelog.

**Version:** 1.0.0  
**Locked at:** Phase 2 full run (September 2025)

---

## 1. Dataset — RFW Pair Sampling

| Parameter | Value | Notes |
|-----------|-------|-------|
| Dataset | [Racial Faces in the Wild (RFW)](http://www.whdeng.cn/RFW/index.html) | Test set only |
| Ethnicities | African, Asian, Caucasian, Indian | 4 groups |
| Pair types | genuine (same identity), impostor (different identity) | 50/50 split |
| Pairs per ethnicity per type | 30 | 120 genuine + 120 impostor = 240 total |
| Sampling seed | `42` | `random.seed(42)` in `rfw_fairness_baseline.ipynb` |
| Sampling method | Stratified random without replacement per ethnicity × label | Locked in `pairs_locked.csv` |

---

## 2. Verifier Models

| Parameter | Value | Notes |
|-----------|-------|-------|
| Model 1 | ArcFace (InsightFace `buffalo_l`) | Identity verification |
| Model 2 | AdaFace (`ir_101` backbone) | Identity verification |
| ArcFace margin threshold | **0.3133** | Similarity score margin; locked from pilot calibration |
| AdaFace margin threshold | From pilot calibration | See `rfw_fairness_baseline.ipynb` |
| Image preprocessing | MTCNN alignment (AdaFace); InsightFace built-in (ArcFace) | |
| Verifier score columns | `verifier_score_margin_arcface`, `verifier_score_margin_adaface` | In `pairs_locked.csv` |

---

## 3. VLM Generation

| Parameter | Value | Notes |
|-----------|-------|-------|
| Model 1 | `LLaVA-NeXT-Mistral-7B` | `llava-hf/llava-v1.6-mistral-7b-hf` |
| Model 2 | `Qwen2.5-VL-7B-Instruct` | `Qwen/Qwen2.5-VL-7B-Instruct` |
| Prompt variants | V1: `direct_feature_comparison`, V2: `chain_of_thought` | See prompts in `rfw-phase-2-run.ipynb` |
| Total generations | 240 pairs × 2 variants × 2 models = **960** | LLaVA had 1 dropped row (239 in V2) |
| Verdict detection | Literal keyword match, then semantic fallback | `verdict_tier` column |
| Inference environment | Kaggle GPU (P100) | |
| LLaVA pilot mean inference time | **39.7 s** per generation | Fallback constant if pilot CSV missing |

---

## 4. Annotation — Phase 3

| Parameter | Value | Notes |
|-----------|-------|-------|
| Annotator assignment seed | **1337** | `np.random.default_rng(seed=1337)` in `assign_annotators.py` |
| Kappa subset | 5 pairs × 4 ethnicities × 2 labels = **40 pairs** | Double-coded for inter-rater reliability |
| Annotator A subset | 12 pairs × 4 ethnicities × 2 labels = **96 pairs** | |
| Annotator B subset | 13 pairs × 4 ethnicities × 2 labels = **104 pairs** | |
| Annotation labels | `hallucinated`, `accurate`, `unverifiable` | `label_chics` column |
| Claim decomposition | `scripts/extract_claims.py` | Atomic descriptive claims only; verdicts stripped |

---

## 5. Statistical Analysis

### 5.1 Chi-Square Test

| Parameter | Value |
|-----------|-------|
| Test | Pearson Chi-square of independence |
| Contingency table | 4 ethnicities × 2 outcomes (correct / incorrect) per slice |
| Implementation | `scipy.stats.chi2_contingency` |
| Uncorrected α | 0.05 |

### 5.2 FDR Correction

| Parameter | Value | Notes |
|-----------|-------|-------|
| Method | Benjamini-Hochberg (`fdr_bh`) | |
| Implementation | `statsmodels.stats.multitest.multipletests` |
| Corrected α | **0.05** | |
| **Grouping** | **Pooled across all 6 tests in one pass** | 2 models × 3 slices (All / V1 / V2) |
| Tests included | LLaVA-All, LLaVA-V1, LLaVA-V2, Qwen-All, Qwen-V1, Qwen-V2 | |

> **Critical detail:** FDR correction is applied to all 6 p-values simultaneously,
> NOT per-model.  Running per-model (3 tests × 2) yields different `p_adj` values.
> This pooled call is what `scripts/run_analysis.py` and notebook cell 34 both implement.

### 5.3 Slices Analyzed

| Slice label | Description |
|-------------|-------------|
| `All` | All prompt variants pooled per model |
| `V1: direct_feature_comparison` | Prompt variant 1 only |
| `V2: chain_of_thought` | Prompt variant 2 only |

---

## 6. Reproducibility Notes

- **Large raw files** (`qwen_full_results.csv`, `llava_next_full_results.csv`, annotation CSVs) are
  hosted on [HuggingFace Datasets](https://huggingface.co/datasets/chidroopakanaparthy/vlm-fairness-outputs) *(link TBD post-submission)*. Download them into `results_data/` before running `scripts/run_analysis.py`.
- **`pairs_locked.csv`** (the exact 240 pairs used) is also hosted on HuggingFace; it is the ground truth for all downstream analysis.
- Running `python scripts/run_analysis.py` with the downloaded CSVs regenerates all three files in `results/` with zero diff against the committed versions.

---

## Changelog

| Version | Date | Change |
|---------|------|--------|
| 1.0.0 | 2025-09 | Initial lock — Phase 2 full run complete |
