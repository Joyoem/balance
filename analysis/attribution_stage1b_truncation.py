import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from utils.simple_tokenizer import SimpleTokenizer


RUN = Path(os.environ["RUN"])
INFILE = RUN / "per_expression_enriched.csv"
OUTDIR = RUN / "attribution_stage1b_truncation"
OUTDIR.mkdir(exist_ok=True)

CONTEXT_LEN = 17
N_BOOT = 2000
SEED = 20261016

rng = np.random.default_rng(SEED)
tok = SimpleTokenizer()

df = pd.read_csv(INFILE)

# ---------------------------------------------------------
# Reconstruct ORIGINAL CLIP token length before DETRIS
# truncates to context_length=17.
#
# +2 = <startoftext> + <endoftext>
# ---------------------------------------------------------

df["raw_clip_token_count"] = df["expression"].apply(
    lambda x: len(tok.encode(str(x))) + 2
)

df["is_truncated"] = (
    df["raw_clip_token_count"] > CONTEXT_LEN
)

df["tokens_removed"] = np.maximum(
    df["raw_clip_token_count"] - CONTEXT_LEN,
    0
)

print("===== TRUNCATION SANITY CHECK =====")

print("queries:", len(df))
print(
    "truncated:",
    int(df["is_truncated"].sum()),
    f"({df['is_truncated'].mean()*100:.2f}%)"
)

print("\nRaw CLIP token count:")
print(
    df["raw_clip_token_count"]
    .describe(
        percentiles=[
            .01,.05,.10,.25,.50,.75,.90,.95,.99
        ]
    )
)

# Verify our reconstructed tokenization matches diagnostic
# output whenever there was no truncation.
nontr = df[~df["is_truncated"]]

max_diff = np.abs(
    nontr["raw_clip_token_count"] -
    nontr["clip_token_count"]
).max()

print(
    "\nmax token-count disagreement "
    "for non-truncated queries:",
    max_diff
)

assert max_diff == 0, \
    "Tokenizer reconstruction disagrees with inference."


# ---------------------------------------------------------
# Basic performance by truncation
# ---------------------------------------------------------

def simple_summary(g):
    return {
        "queries": int(len(g)),
        "refs": int(g["seg_id"].nunique()),
        "mean_iou": float(g["iou"].mean()),
        "fail50": float((g["iou"] < .5).mean()),
        "severe_fail10": float((g["iou"] < .1).mean()),
    }


trunc_summary = {
    "not_truncated":
        simple_summary(df[~df["is_truncated"]]),
    "truncated":
        simple_summary(df[df["is_truncated"]]),
}

print("\n===== PERFORMANCE BY TRUNCATION =====")
print(json.dumps(trunc_summary, indent=2))


# ---------------------------------------------------------
# Long-expression bucket: separate truncated vs nontruncated
# ---------------------------------------------------------

long_df = df[df["word_count"] >= 6]

long_summary = {
    "long_nontruncated":
        simple_summary(
            long_df[~long_df["is_truncated"]]
        ),
    "long_truncated":
        simple_summary(
            long_df[long_df["is_truncated"]]
        ),
}

print("\n===== 6+ WORD EXPRESSIONS =====")
print(json.dumps(long_summary, indent=2))


# ---------------------------------------------------------
# WITHIN-OBJECT TEST, excluding ALL truncated expressions.
#
# This asks:
# same object + same image + same distractors,
# among expressions that DETRIS saw in full,
# does the longer expression still do worse?
# ---------------------------------------------------------

rows = []

for seg_id, g in nontr.groupby("seg_id"):

    if g["word_count"].nunique() < 2:
        continue

    min_len = g["word_count"].min()
    max_len = g["word_count"].max()

    short = g[g["word_count"] == min_len]
    long = g[g["word_count"] == max_len]

    short_iou = short["iou"].mean()
    long_iou = long["iou"].mean()

    rows.append({
        "seg_id": seg_id,
        "short_len": min_len,
        "long_len": max_len,
        "length_delta": max_len - min_len,
        "short_iou": short_iou,
        "long_iou": long_iou,
        "delta_iou_long_minus_short":
            long_iou - short_iou,
    })

pairs = pd.DataFrame(rows)

pairs.to_csv(
    OUTDIR /
    "within_object_nontruncated_short_vs_long.csv",
    index=False
)

delta = (
    pairs["delta_iou_long_minus_short"]
    .to_numpy()
)

boot = np.empty(N_BOOT)

for b in range(N_BOOT):
    idx = rng.integers(
        0,
        len(delta),
        size=len(delta)
    )
    boot[b] = delta[idx].mean()


pair_summary = {
    "num_refs":
        int(len(pairs)),

    "mean_short_iou":
        float(pairs["short_iou"].mean()),

    "mean_long_iou":
        float(pairs["long_iou"].mean()),

    "mean_delta_long_minus_short":
        float(delta.mean()),

    "ci_low":
        float(np.quantile(boot, .025)),

    "ci_high":
        float(np.quantile(boot, .975)),

    "fraction_long_worse":
        float((delta < 0).mean()),

    "fraction_long_worse_by_10pp":
        float((delta < -.10).mean()),

    "fraction_long_worse_by_30pp":
        float((delta < -.30).mean()),

    "fraction_long_better_by_10pp":
        float((delta > .10).mean()),

    "delta_p01":
        float(np.quantile(delta, .01)),

    "delta_p05":
        float(np.quantile(delta, .05)),

    "delta_p25":
        float(np.quantile(delta, .25)),

    "delta_median":
        float(np.quantile(delta, .50)),

    "delta_p75":
        float(np.quantile(delta, .75)),

    "delta_p95":
        float(np.quantile(delta, .95)),

    "delta_p99":
        float(np.quantile(delta, .99)),
}

print(
    "\n===== NON-TRUNCATED WITHIN-OBJECT "
    "SHORT vs LONG ====="
)
print(json.dumps(pair_summary, indent=2))


# ---------------------------------------------------------
# Fixed-effect slope on NON-TRUNCATED expressions only.
# ---------------------------------------------------------

fe_rows = []

for seg_id, g in nontr.groupby("seg_id"):

    if g["word_count"].nunique() < 2:
        continue

    x = g["word_count"].to_numpy(dtype=float)
    y = g["iou"].to_numpy(dtype=float)

    xc = x - x.mean()
    yc = y - y.mean()

    fe_rows.append({
        "seg_id": seg_id,
        "sxx": float(np.sum(xc * xc)),
        "sxy": float(np.sum(xc * yc)),
    })

fe = pd.DataFrame(fe_rows)

beta = fe["sxy"].sum() / fe["sxx"].sum()

arr = fe[["sxx", "sxy"]].to_numpy()

boot_beta = np.empty(N_BOOT)

for b in range(N_BOOT):
    idx = rng.integers(
        0,
        len(arr),
        size=len(arr)
    )

    sample = arr[idx].sum(axis=0)

    boot_beta[b] = (
        sample[1] / sample[0]
    )

fe_summary = {
    "num_refs":
        int(len(fe)),

    "beta_iou_per_word":
        float(beta),

    "beta_percentage_points_per_word":
        float(beta * 100),

    "ci_low_percentage_points":
        float(
            np.quantile(boot_beta, .025) * 100
        ),

    "ci_high_percentage_points":
        float(
            np.quantile(boot_beta, .975) * 100
        ),
}

print(
    "\n===== NON-TRUNCATED FIXED EFFECT ====="
)
print(json.dumps(fe_summary, indent=2))


# ---------------------------------------------------------
# Save query-level data for later cross-model comparison
# ---------------------------------------------------------

df.to_csv(
    OUTDIR /
    "per_expression_with_truncation_metadata.csv",
    index=False
)

summary = {
    "context_length": CONTEXT_LEN,
    "truncation": trunc_summary,
    "long_expression": long_summary,
    "nontruncated_within_object":
        pair_summary,
    "nontruncated_fixed_effect":
        fe_summary,
}

with open(
    OUTDIR / "summary.json",
    "w"
) as f:
    json.dump(summary, f, indent=2)

print("\nSaved:", OUTDIR)
