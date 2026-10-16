import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from refer import REFER


RUN = Path(os.environ["RUN"])
ROOT = Path.home() / "projects/ris_discovery/DETRIS"
OUT = Path(f"/data/{os.environ['USER']}/ris_discovery/manifests")
OUT.mkdir(parents=True, exist_ok=True)

print("Loading RefCOCO+...")
refer = REFER(
    str(ROOT / "datasets"),
    dataset="refcoco+",
    splitBy="unc"
)

val_ids = refer.getRefIds(split="val")
print("val refs:", len(val_ids))

records = []

for k, ref_id in enumerate(val_ids):
    ref_id = int(ref_id)

    ref = refer.Refs[ref_id]
    ann = refer.refToAnn[ref_id]
    img = refer.Imgs[ref["image_id"]]

    h = int(img["height"])
    w = int(img["width"])

    # Regenerate mask directly from original RefCOCO+ annotation
    mask = refer.getMask(ref)["mask"] > 0
    mask_area_px = int(mask.sum())
    mask_area_ratio = mask_area_px / float(h * w)

    # Original COCO annotation area
    ann_area = float(ann["area"])
    ann_area_ratio = ann_area / float(h * w)

    # Original COCO bbox
    x, y, bw, bh = ann["bbox"]
    bbox_area_ratio = float(bw * bh) / float(h * w)

    # All annotated COCO instances in this image
    image_anns = refer.imgToAnns[ref["image_id"]]

    same_category_instances = sum(
        int(a["category_id"] == ref["category_id"])
        for a in image_anns
    )

    same_category_distractors = max(
        0, same_category_instances - 1
    )

    total_instances = len(image_anns)

    category_name = refer.Cats[ref["category_id"]]

    for sent_idx, sent in enumerate(ref["sentences"]):
        records.append({
            "seg_id": ref_id,
            "sent_idx": sent_idx,
            "sent_id": int(sent["sent_id"]),
            "expression": sent["sent"].strip(),

            "image_id": int(ref["image_id"]),
            "img_name": img["file_name"],

            "category_id_coco": int(ref["category_id"]),
            "category_name": category_name,

            "ori_h_source": h,
            "ori_w_source": w,

            "target_area_px_source": mask_area_px,
            "target_area_ratio_source": mask_area_ratio,
            "annotation_area_ratio": ann_area_ratio,
            "bbox_area_ratio_source": bbox_area_ratio,

            "total_instances_image": total_instances,
            "same_category_instances": same_category_instances,
            "same_category_distractors": same_category_distractors,
        })

    if (k + 1) % 500 == 0:
        print(f"{k+1}/{len(val_ids)} refs")

meta = pd.DataFrame(records)

meta_path = OUT / "refcocop_val_expression_metadata.csv"
meta.to_csv(meta_path, index=False)

print("\n===== SOURCE METADATA =====")
print("expressions:", len(meta))
print("refs:", meta.seg_id.nunique())
print("images:", meta.image_id.nunique())

print("\nTarget-area ratio percentiles:")
print(
    meta["target_area_ratio_source"]
    .describe(percentiles=[.01, .05, .10, .25, .50, .75, .90, .95, .99])
)

print("\nAbsolute size counts:")
for threshold in [0.001, 0.005, 0.01, 0.02, 0.05, 0.10]:
    n = (
        meta["target_area_ratio_source"] < threshold
    ).sum()

    print(
        f"< {threshold*100:4.1f}% : "
        f"{n:5d} expressions "
        f"({n/len(meta)*100:6.2f}%)"
    )

print("\nSame-category distractors:")
print(
    meta["same_category_distractors"]
    .value_counts()
    .sort_index()
    .head(15)
)

# --------------------------------------------------
# Merge with DETRIS diagnostic results
# --------------------------------------------------

diag_path = RUN / "per_expression.csv"
diag = pd.read_csv(diag_path)

merged = diag.merge(
    meta,
    on=["seg_id", "sent_idx", "expression"],
    how="left",
    validate="one_to_one",
    suffixes=("_detris", "_source")
)

assert len(merged) == len(diag), \
    f"merge changed row count: {len(diag)} -> {len(merged)}"

assert merged["image_id"].notna().all(), \
    "some DETRIS rows did not match source metadata"

# Integrity checks
area_diff = np.abs(
    merged["target_area_ratio"] -
    merged["target_area_ratio_source"]
)

bbox_diff = np.abs(
    merged["bbox_area_ratio"] -
    merged["bbox_area_ratio_source"]
)

validation = {
    "num_expressions_diag": int(len(diag)),
    "num_expressions_metadata": int(len(meta)),
    "num_refs": int(meta.seg_id.nunique()),

    "target_area_max_abs_diff": float(area_diff.max()),
    "target_area_mean_abs_diff": float(area_diff.mean()),

    "bbox_area_max_abs_diff": float(bbox_diff.max()),
    "bbox_area_mean_abs_diff": float(bbox_diff.mean()),

    "target_area_source_min": float(
        meta.target_area_ratio_source.min()
    ),

    "target_area_source_p01": float(
        meta.target_area_ratio_source.quantile(.01)
    ),

    "target_area_source_p05": float(
        meta.target_area_ratio_source.quantile(.05)
    ),
}

with open(RUN / "metadata_validation.json", "w") as f:
    json.dump(validation, f, indent=2)

merged.to_csv(
    RUN / "per_expression_enriched.csv",
    index=False
)

print("\n===== INTEGRITY CHECK =====")
print(json.dumps(validation, indent=2))

print("\nSaved:")
print(meta_path)
print(RUN / "per_expression_enriched.csv")
print(RUN / "metadata_validation.json")
