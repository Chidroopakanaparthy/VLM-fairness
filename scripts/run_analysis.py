"""
run_analysis.py — VLM Fairness Statistical Analysis
=====================================================
Reproduces the locked analysis from rfw-phase-2-run.ipynb (cells 33–34).

Pipeline:
  1. Load Qwen and LLaVA-NeXT full-run CSVs (local paths).
  2. Filter to rows with a parseable verdict_polarity (same / different).
  3. Compute per-ethnicity accuracy, TPR, FPR, TNR for each model × prompt-variant slice.
  4. Run a 4×2 Chi-square test of independence per slice (ethnicity × correct/incorrect).
  5. Apply Benjamini-Hochberg FDR correction across all 6 tests in one pooled pass
     (2 models × 3 slices: All / V1 / V2).  This matches cell 34 exactly.

Outputs (written to results/):
  vlm_summary_metrics.csv   — per-ethnicity accuracy metrics
  vlm_chi2_results.csv      — raw Chi-square results
  vlm_chi2_results_fdr.csv  — BH-corrected results (paper numbers)

FDR grouping note (PROTOCOL.md §FDR):
  multipletests is called once on all 6 p-values together, NOT per-model.
  This matches the notebook and is the number cited in the paper.

Usage:
  python scripts/run_analysis.py \
      --qwen   results_data/qwen_full_results.csv \
      --llava  results_data/llava_next_full_results.csv \
      --outdir results/
"""

import argparse
import os
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from statsmodels.stats.multitest import multipletests

# ---------------------------------------------------------------------------
# Constants (locked — see PROTOCOL.md)
# ---------------------------------------------------------------------------
RACES            = ["African", "Asian", "Caucasian", "Indian"]
VALID_POLARITIES = ["same", "different"]
POLARITY_MAP     = {"same": "genuine", "different": "impostor"}
FDR_ALPHA        = 0.05
FDR_METHOD       = "fdr_bh"   # Benjamini-Hochberg


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------

def load_and_merge(qwen_path: str, llava_path: str) -> pd.DataFrame:
    qwen_df  = pd.read_csv(qwen_path)
    llava_df = pd.read_csv(llava_path)
    df = pd.concat([qwen_df, llava_df], ignore_index=True)

    # Keep only rows with a parseable verdict
    df = df[df["verdict_polarity"].isin(VALID_POLARITIES)].copy()

    df["predicted_label"] = df["verdict_polarity"].map(POLARITY_MAP)
    df["is_correct"]      = (df["predicted_label"] == df["ground_truth_label"]).astype(int)

    n_models = df["model_name"].nunique()
    print(f"Loaded {len(df):,} rows across {n_models} model(s).")
    print(df.groupby(["model_name", "prompt_variant"]).size().rename("n_rows").to_string())
    return df


def analyze_slice(
    slice_df: pd.DataFrame,
    model_label: str,
    prompt_variant_label: str,
    summary_rows: list,
    chi2_rows: list,
) -> None:
    """Compute per-ethnicity metrics and a 4×2 Chi-square for one model × variant slice."""
    contingency   = []
    race_accs     = {}

    for race in RACES:
        r_df = slice_df[slice_df["ethnicity"] == race]
        if len(r_df) == 0:
            continue

        gen_df = r_df[r_df["ground_truth_label"] == "genuine"]
        imp_df = r_df[r_df["ground_truth_label"] == "impostor"]

        acc = r_df["is_correct"].mean()
        tpr = gen_df["is_correct"].mean() if len(gen_df) > 0 else 0.0
        tnr = imp_df["is_correct"].mean() if len(imp_df) > 0 else 0.0

        summary_rows.append({
            "Race":           race,
            "Model":          model_label,
            "Prompt_Variant": prompt_variant_label,
            "Accuracy":       round(acc, 4),
            "TPR":            round(tpr, 4),
            "FPR":            round(1.0 - tnr, 4),
            "TNR":            round(tnr, 4),
            "N_pairs":        len(r_df),
            "N_genuine":      len(gen_df),
            "N_impostor":     len(imp_df),
        })

        n_correct   = int(r_df["is_correct"].sum())
        n_incorrect = len(r_df) - n_correct
        contingency.append([n_correct, n_incorrect])
        race_accs[race] = acc

    if len(contingency) == 4:
        chi2, p_value, dof, _ = chi2_contingency(np.array(contingency))
        best_race  = max(race_accs, key=race_accs.get)
        worst_race = min(race_accs, key=race_accs.get)

        chi2_rows.append({
            "Model":          model_label,
            "Prompt_Variant": prompt_variant_label,
            "Chi2":           round(chi2, 4),
            "p_value":        round(p_value, 6),
            "DoF":            dof,
            "Significant":    "Yes" if p_value < 0.05 else "No",
            "Best_Race":      best_race,
            "Best_Acc":       round(race_accs[best_race], 4),
            "Worst_Race":     worst_race,
            "Worst_Acc":      round(race_accs[worst_race], 4),
        })


def apply_fdr(chi2_df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply BH-FDR correction pooled across all 6 tests in one pass.
    Replicates cell 34 of rfw-phase-2-run.ipynb exactly.
    """
    p_values = chi2_df["p_value"].values
    reject, pvals_corrected, _, _ = multipletests(p_values, alpha=FDR_ALPHA, method=FDR_METHOD)

    chi2_df = chi2_df.copy()
    chi2_df["p_value_fdr"]    = [round(p, 6) for p in pvals_corrected]
    chi2_df["Significant_FDR"] = ["Yes" if r else "No" for r in reject]

    # Insert FDR columns right after 'Significant' (matches notebook column order)
    cols    = chi2_df.columns.tolist()
    sig_idx = cols.index("Significant")
    new_cols = (
        cols[: sig_idx + 1]
        + ["p_value_fdr", "Significant_FDR"]
        + [c for c in cols[sig_idx + 1 :] if c not in ("p_value_fdr", "Significant_FDR")]
    )
    return chi2_df[new_cols]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(qwen_path: str, llava_path: str, outdir: str) -> None:
    os.makedirs(outdir, exist_ok=True)

    df = load_and_merge(qwen_path, llava_path)

    summary_rows: list = []
    chi2_rows:    list = []

    for model_name in sorted(df["model_name"].unique()):
        m_df = df[df["model_name"] == model_name]

        # Pooled across variants
        analyze_slice(m_df, model_name, "All", summary_rows, chi2_rows)

        # Per-variant
        for variant in sorted(m_df["prompt_variant"].unique()):
            v_df     = m_df[m_df["prompt_variant"] == variant]
            v_label  = (
                v_df["prompt_variant_label"].iloc[0]
                if "prompt_variant_label" in v_df.columns
                else f"Variant {variant}"
            )
            analyze_slice(v_df, model_name, f"V{variant}: {v_label}", summary_rows, chi2_rows)

    summary_df = pd.DataFrame(summary_rows)
    chi2_df    = pd.DataFrame(chi2_rows)
    fdr_df     = apply_fdr(chi2_df)

    # Write outputs
    summary_path = os.path.join(outdir, "vlm_summary_metrics.csv")
    chi2_path    = os.path.join(outdir, "vlm_chi2_results.csv")
    fdr_path     = os.path.join(outdir, "vlm_chi2_results_fdr.csv")

    summary_df.to_csv(summary_path, index=False)
    chi2_df.to_csv(chi2_path,       index=False)
    fdr_df.to_csv(fdr_path,         index=False)

    print("\n" + "=" * 80)
    print("VLM CHI-SQUARE DISPARITY RESULTS")
    print("=" * 80)
    print(chi2_df.to_string(index=False))

    print("\n" + "=" * 80)
    print(f"BH-FDR CORRECTED (alpha={FDR_ALPHA}, method={FDR_METHOD!r})")
    print("Correction applied pooled across all 6 tests — see PROTOCOL.md §FDR")
    print("=" * 80)
    print(fdr_df.to_string(index=False))

    print(f"\nOutputs written to {outdir}")
    print(f"  {summary_path}")
    print(f"  {chi2_path}")
    print(f"  {fdr_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VLM fairness statistical analysis.")
    parser.add_argument(
        "--qwen",
        default="results_data/qwen_full_results.csv",
        help="Path to qwen_full_results.csv (default: results_data/qwen_full_results.csv)",
    )
    parser.add_argument(
        "--llava",
        default="results_data/llava_next_full_results.csv",
        help="Path to llava_next_full_results.csv (default: results_data/llava_next_full_results.csv)",
    )
    parser.add_argument(
        "--outdir",
        default="results/",
        help="Directory to write output CSVs (default: results/)",
    )
    args = parser.parse_args()
    main(args.qwen, args.llava, args.outdir)
