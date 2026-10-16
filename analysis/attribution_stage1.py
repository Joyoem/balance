import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

RUN = Path(os.environ["RUN"])
INFILE = RUN / "per_expression_enriched.csv"
OUTDIR = RUN / "attribution_stage1"
OUTDIR.mkdir(exist_ok=True)

SEED = 20261016
N_BOOT = 2000
rng = np.random.default_rng(SEED)

df = pd.read_csv(INFILE)

# Derived model-output variables needed by attribution.
# They are defined here directly from the validated raw diagnostic data,
# so attribution does not depend on the slice-analysis output.
df["pred_target_area_ratio"] = (
    df["pred_area_px"] / df["target_area_px"].clip(lower=1)
)
df["fail50"] = df["iou"] < 0.50

# ============================================================
# A. BASIC ASSOCIATIONS AMONG DATASET VARIABLES
# ============================================================

assoc_cols = [
    "target_area_ratio_source",
    "same_category_distractors",
    "word_count",
]

corr = df[assoc_cols].corr(method="spearman")
corr.to_csv(OUTDIR / "metadata_spearman.csv")

print("===== METADATA SPEARMAN CORRELATIONS =====")
print(corr.round(3).to_string())


# ============================================================
# B. WITHIN-OBJECT EXPRESSION-LENGTH TEST
#
# Same seg_id:
#   same image
#   same target
#   same target area
#   same category
#   same distractors
#
# Only referring expression differs.
# ============================================================

valid_refs = []

for seg_id, g in df.groupby("seg_id"):
    if g["word_count"].nunique() < 2:
        continue

    min_len = g["word_count"].min()
    max_len = g["word_count"].max()

    short = g[g["word_count"] == min_len]
    long = g[g["word_count"] == max_len]

    valid_refs.append({
        "seg_id": seg_id,

        "short_len": min_len,
        "long_len": max_len,
        "length_delta": max_len - min_len,

        "short_iou": short["iou"].mean(),
        "long_iou": long["iou"].mean(),
        "delta_iou_long_minus_short":
            long["iou"].mean() - short["iou"].mean(),

        "short_pred_target_ratio":
            short["pred_target_area_ratio"].mean(),
        "long_pred_target_ratio":
            long["pred_target_area_ratio"].mean(),

        "delta_pred_target_ratio":
            long["pred_target_area_ratio"].mean()
            - short["pred_target_area_ratio"].mean(),

        "target_area_ratio":
            g["target_area_ratio_source"].iloc[0],

        "same_category_distractors":
            g["same_category_distractors"].iloc[0],
    })

pairs = pd.DataFrame(valid_refs)
pairs.to_csv(
    OUTDIR / "within_object_short_vs_long.csv",
    index=False
)

delta = pairs["delta_iou_long_minus_short"].to_numpy()

boot_delta = np.empty(N_BOOT)

for b in range(N_BOOT):
    idx = rng.integers(
        0,
        len(delta),
        size=len(delta)
    )
    boot_delta[b] = delta[idx].mean()

within_summary = {
    "num_refs_with_length_variation":
        int(len(pairs)),

    "mean_short_iou":
        float(pairs["short_iou"].mean()),

    "mean_long_iou":
        float(pairs["long_iou"].mean()),

    "mean_delta_iou_long_minus_short":
        float(delta.mean()),

    "delta_iou_ci_low":
        float(np.quantile(boot_delta, .025)),

    "delta_iou_ci_high":
        float(np.quantile(boot_delta, .975)),

    "fraction_long_worse":
        float((delta < 0).mean()),

    "mean_length_delta_words":
        float(pairs["length_delta"].mean()),

    "mean_delta_pred_target_ratio":
        float(
            pairs["delta_pred_target_ratio"].mean()
        ),
}

print("\n===== WITHIN-OBJECT SHORT vs LONG =====")
print(json.dumps(within_summary, indent=2))


# ============================================================
# C. WITHIN-OBJECT FIXED-EFFECT SLOPE FOR WORD COUNT
#
# Equivalent idea:
# remove each object's own mean IoU and own mean word count.
#
# This eliminates all object-level constant confounders.
# ============================================================

per_ref_stats = []

for seg_id, g in df.groupby("seg_id"):
    if g["word_count"].nunique() < 2:
        continue

    x = g["word_count"].to_numpy(dtype=float)
    y = g["iou"].to_numpy(dtype=float)

    xc = x - x.mean()
    yc = y - y.mean()

    per_ref_stats.append({
        "seg_id": seg_id,
        "sxx": float(np.sum(xc * xc)),
        "sxy": float(np.sum(xc * yc)),
    })

fe = pd.DataFrame(per_ref_stats)

beta_fe = fe["sxy"].sum() / fe["sxx"].sum()

boot_beta = np.empty(N_BOOT)

arr = fe[["sxx", "sxy"]].to_numpy()

for b in range(N_BOOT):
    idx = rng.integers(
        0,
        len(arr),
        size=len(arr)
    )

    sample = arr[idx].sum(axis=0)

    boot_beta[b] = sample[1] / sample[0]

fixed_effect_summary = {
    "beta_iou_per_additional_word":
        float(beta_fe),

    "beta_percentage_points_per_word":
        float(beta_fe * 100),

    "ci_low_percentage_points_per_word":
        float(np.quantile(boot_beta, .025) * 100),

    "ci_high_percentage_points_per_word":
        float(np.quantile(boot_beta, .975) * 100),
}

print("\n===== WITHIN-OBJECT FIXED EFFECT =====")
print(json.dumps(fixed_effect_summary, indent=2))


# ============================================================
# D. REF-LEVEL MULTIVARIABLE ATTRIBUTION
#
# One row per object, so refs with more sentences do not get
# extra statistical weight.
#
# predictors:
#   log(target area)
#   log(1 + distractors)
#   mean expression length
#
# All standardized for comparable coefficients.
# ============================================================

ref_df = (
    df.groupby("seg_id")
    .agg(
        mean_iou=("iou", "mean"),
        target_area_ratio=(
            "target_area_ratio_source", "first"
        ),
        distractors=(
            "same_category_distractors", "first"
        ),
        mean_word_count=("word_count", "mean"),
        fail_rate=("fail50", "mean"),
    )
    .reset_index()
)

raw_X = np.column_stack([
    np.log(
        ref_df["target_area_ratio"].to_numpy()
    ),
    np.log1p(
        ref_df["distractors"].to_numpy()
    ),
    ref_df["mean_word_count"].to_numpy(),
])

predictor_names = [
    "log_target_area",
    "log1p_distractors",
    "mean_expression_length",
]

mu = raw_X.mean(axis=0)
sd = raw_X.std(axis=0, ddof=0)

Z = (raw_X - mu) / sd

X = np.column_stack([
    np.ones(len(Z)),
    Z
])

y = ref_df["mean_iou"].to_numpy()

coef = np.linalg.lstsq(
    X,
    y,
    rcond=None
)[0]

boot_coef = np.empty(
    (N_BOOT, len(coef))
)

for b in range(N_BOOT):
    idx = rng.integers(
        0,
        len(ref_df),
        size=len(ref_df)
    )

    boot_coef[b] = np.linalg.lstsq(
        X[idx],
        y[idx],
        rcond=None
    )[0]

reg_rows = []

for j, name in enumerate(
    ["intercept"] + predictor_names
):
    reg_rows.append({
        "predictor": name,
        "coefficient_iou":
            coef[j],
        "coefficient_percentage_points":
            coef[j] * 100,
        "ci_low_percentage_points":
            np.quantile(
                boot_coef[:, j], .025
            ) * 100,
        "ci_high_percentage_points":
            np.quantile(
                boot_coef[:, j], .975
            ) * 100,
    })

reg = pd.DataFrame(reg_rows)

reg.to_csv(
    OUTDIR / "ref_level_multivariable_regression.csv",
    index=False
)

print("\n===== REF-LEVEL MULTIVARIABLE MODEL =====")
print(
    reg.round(3).to_string(index=False)
)


# ============================================================
# E. SAVE MASTER SUMMARY
# ============================================================

summary = {
    "design": {
        "dataset": "RefCOCO+ val",
        "model": "DETRIS-B",
        "bootstrap_unit":
            "seg_id/referring object",
        "bootstrap_repetitions":
            N_BOOT,

        "interpretation":
            "Associational attribution only; "
            "not causal inference."
    },

    "within_object_short_long":
        within_summary,

    "within_object_fixed_effect":
        fixed_effect_summary,

    "multivariable_predictors":
        predictor_names,
}

with open(
    OUTDIR / "attribution_stage1_summary.json",
    "w"
) as f:
    json.dump(summary, f, indent=2)

print("\nSaved to:", OUTDIR)
