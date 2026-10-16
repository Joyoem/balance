import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

RUN = Path(os.environ["RUN"])
INFILE = RUN / "per_expression_enriched.csv"
OUTDIR = RUN / "slice_analysis"
OUTDIR.mkdir(exist_ok=True)

SEED = 20261016
N_BOOT = 1000
rng = np.random.default_rng(SEED)

df = pd.read_csv(INFILE)

# ---------------------------------------------------------
# 1. Freeze slice definitions using DATASET METADATA ONLY.
#    No IoU/model-output values are used to define bins.
# ---------------------------------------------------------

# target size is ref-level, so do NOT let refs with more sentences
# have more influence on thresholds.
refs = (
    df.drop_duplicates("seg_id")
      [["seg_id", "target_area_ratio_source",
        "same_category_distractors"]]
      .copy()
)

q33 = float(refs["target_area_ratio_source"].quantile(1/3))
q67 = float(refs["target_area_ratio_source"].quantile(2/3))

thresholds = {
    "target_area_ratio_q33": q33,
    "target_area_ratio_q67": q67,
    "target_area_definition":
        "relative-low <= q33; relative-mid q33-q67; relative-high > q67",
    "distractor_definition":
        "1; 2-4; 5+ same-category distractors",
    "expression_length_definition":
        "1-2; 3-5; 6+ whitespace-token words",
    "bootstrap_unit": "seg_id",
    "bootstrap_repetitions": N_BOOT,
    "seed": SEED,
}

with open(OUTDIR / "slice_definitions.json", "w") as f:
    json.dump(thresholds, f, indent=2)

print("===== FROZEN SLICE DEFINITIONS =====")
print(json.dumps(thresholds, indent=2))


def area_bin(x):
    if x <= q33:
        return "relative-low"
    elif x <= q67:
        return "relative-mid"
    return "relative-high"


def distractor_bin(x):
    if x == 1:
        return "1"
    elif x <= 4:
        return "2-4"
    return "5+"


def length_bin(x):
    if x <= 2:
        return "1-2"
    elif x <= 5:
        return "3-5"
    return "6+"


df["target_area_bin"] = \
    df["target_area_ratio_source"].apply(area_bin)

df["distractor_bin"] = \
    df["same_category_distractors"].apply(distractor_bin)

df["length_bin"] = \
    df["word_count"].apply(length_bin)

# Model-output diagnostic variables. These do NOT define slices.
df["pred_target_area_ratio"] = (
    df["pred_area_px"] / df["target_area_px"].clip(lower=1)
)

df["undersegment"] = df["pred_target_area_ratio"] < 0.5
df["oversegment"] = df["pred_target_area_ratio"] > 1.5
df["fail50"] = df["iou"] < 0.50
df["severe_fail10"] = df["iou"] < 0.10


# ---------------------------------------------------------
# 2. Cluster bootstrap by seg_id
# ---------------------------------------------------------

def cluster_bootstrap_ci(g):
    # Collapse expressions to per-ref sufficient statistics.
    r = (
        g.groupby("seg_id")
         .agg(
             n=("iou", "size"),
             sum_iou=("iou", "sum"),
             sum_I=("intersection", "sum"),
             sum_U=("union", "sum"),
             fail=("fail50", "sum"),
         )
         .reset_index(drop=True)
    )

    n_refs = len(r)

    miou_boot = np.empty(N_BOOT)
    oiou_boot = np.empty(N_BOOT)
    fail_boot = np.empty(N_BOOT)

    arr = r[["n", "sum_iou", "sum_I", "sum_U", "fail"]].to_numpy()

    for b in range(N_BOOT):
        idx = rng.integers(0, n_refs, size=n_refs)
        s = arr[idx].sum(axis=0)

        n, sum_iou, sum_I, sum_U, fail = s

        miou_boot[b] = sum_iou / n
        oiou_boot[b] = sum_I / sum_U
        fail_boot[b] = fail / n

    return {
        "miou_ci_low": np.quantile(miou_boot, .025),
        "miou_ci_high": np.quantile(miou_boot, .975),
        "oiou_ci_low": np.quantile(oiou_boot, .025),
        "oiou_ci_high": np.quantile(oiou_boot, .975),
        "fail50_ci_low": np.quantile(fail_boot, .025),
        "fail50_ci_high": np.quantile(fail_boot, .975),
    }


def summarize(axis, gname, g):
    ci = cluster_bootstrap_ci(g)

    return {
        "axis": axis,
        "group": str(gname),

        "refs": g["seg_id"].nunique(),
        "queries": len(g),

        "mIoU": g["iou"].mean() * 100,
        "mIoU_ci_low": ci["miou_ci_low"] * 100,
        "mIoU_ci_high": ci["miou_ci_high"] * 100,

        "oIoU": g["intersection"].sum()
                 / g["union"].sum() * 100,
        "oIoU_ci_low": ci["oiou_ci_low"] * 100,
        "oIoU_ci_high": ci["oiou_ci_high"] * 100,

        "fail50_pct": g["fail50"].mean() * 100,
        "fail50_ci_low": ci["fail50_ci_low"] * 100,
        "fail50_ci_high": ci["fail50_ci_high"] * 100,

        "severe_fail10_pct":
            g["severe_fail10"].mean() * 100,

        "median_pred_target_area_ratio":
            g["pred_target_area_ratio"].median(),

        "undersegment_pct":
            g["undersegment"].mean() * 100,

        "oversegment_pct":
            g["oversegment"].mean() * 100,
    }


rows = []

axes = [
    ("target_area", "target_area_bin"),
    ("same_category_distractors", "distractor_bin"),
    ("expression_length", "length_bin"),
]

orders = {
    "target_area": [
        "relative-low",
        "relative-mid",
        "relative-high"
    ],
    "same_category_distractors": [
        "1", "2-4", "5+"
    ],
    "expression_length": [
        "1-2", "3-5", "6+"
    ],
}

for axis_name, column in axes:
    for group_name in orders[axis_name]:
        g = df[df[column] == group_name].copy()

        if len(g) == 0:
            continue

        rows.append(
            summarize(axis_name, group_name, g)
        )

result = pd.DataFrame(rows)
result.to_csv(OUTDIR / "slice_summary.csv", index=False)

# ---------------------------------------------------------
# 3. Save failures for later qualitative inspection.
#    We do NOT cherry-pick them yet.
# ---------------------------------------------------------

failures = (
    df[df["iou"] < 0.5]
    .sort_values(["iou", "seg_id", "sent_idx"])
)

failure_cols = [
    "seg_id",
    "img_name",
    "sent_idx",
    "expression",
    "category_name",
    "target_area_ratio_source",
    "same_category_distractors",
    "word_count",
    "iou",
    "pred_target_area_ratio",
    "undersegment",
    "oversegment",
]

failures[failure_cols].to_csv(
    OUTDIR / "failures_iou_lt_050.csv",
    index=False
)

severe = (
    df[df["iou"] < 0.1]
    .sort_values(["iou", "seg_id", "sent_idx"])
)

severe[failure_cols].to_csv(
    OUTDIR / "severe_failures_iou_lt_010.csv",
    index=False
)

# Save enriched dataframe with frozen bins.
df.to_csv(
    OUTDIR / "per_expression_with_frozen_slices.csv",
    index=False
)

# ---------------------------------------------------------
# 4. Human-readable output
# ---------------------------------------------------------

display_cols = [
    "axis", "group",
    "refs", "queries",
    "mIoU", "mIoU_ci_low", "mIoU_ci_high",
    "oIoU",
    "fail50_pct",
    "severe_fail10_pct",
    "median_pred_target_area_ratio",
    "undersegment_pct",
    "oversegment_pct",
]

print("\n===== SLICE RESULTS =====")
print(
    result[display_cols]
    .round(2)
    .to_string(index=False)
)

print("\nSaved to:", OUTDIR)
