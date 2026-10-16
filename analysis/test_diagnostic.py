import argparse
import csv
import json
import os
import time
import warnings
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

import utils.config as config
from model import build_segmenter
from utils.dataset import RefDataset, tokenize

warnings.filterwarnings("ignore")
cv2.setNumThreads(0)


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--path", required=True)
    p.add_argument("--diag-out", required=True)
    p.add_argument("--max-refs", type=int, default=0)
    p.add_argument("--warmup-queries", type=int, default=50)
    p.add_argument("--opts", nargs=argparse.REMAINDER, default=None)
    cli = p.parse_args()

    cfg = config.load_cfg_from_cfg_file(cli.config)
    cfg.path = cli.path

    if cli.opts is not None:
        cfg = config.merge_cfg_from_list(cfg, cli.opts)

    cfg.diag_out = cli.diag_out
    cfg.max_refs = cli.max_refs
    cfg.warmup_queries = cli.warmup_queries
    return cfg


@torch.no_grad()
def run_diagnostic(loader, model, args, ann_map):
    model.eval()

    rows = []
    cum_I = 0
    cum_U = 0
    query_idx = 0

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    ev_start = torch.cuda.Event(enable_timing=True)
    ev_end = torch.cuda.Event(enable_timing=True)

    wall_start = time.perf_counter()

    for ref_idx, (img, param) in enumerate(loader):
        img = img.cuda(non_blocking=True)

        seg_id = int(param["seg_id"][0].item())
        meta = ann_map[seg_id]

        mask_u8 = cv2.imread(
            param["mask_dir"][0],
            flags=cv2.IMREAD_GRAYSCALE
        )
        if mask_u8 is None:
            raise RuntimeError(f"Could not read mask: {param['mask_dir'][0]}")

        gt = mask_u8 > 0

        h, w = [int(x) for x in param["ori_size"].numpy()[0]]
        mat = param["inverse"].numpy()[0]

        target_area = int(gt.sum())
        target_area_ratio = target_area / float(h * w)

        x1, y1, x2, y2 = meta["bbox"]
        bbox_area = max(0, x2 - x1) * max(0, y2 - y1)
        bbox_area_ratio = bbox_area / float(h * w)

        for sent_idx, sent_batch in enumerate(param["sents"]):
            if isinstance(sent_batch, (list, tuple)):
                expression = sent_batch[0]
            else:
                expression = str(sent_batch)

            # Start query pipeline timing.
            torch.cuda.synchronize()
            q0 = time.perf_counter()

            text_cpu = tokenize(expression, args.word_len, True)
            clip_token_count = int((text_cpu != 0).sum().item())
            text = text_cpu.cuda(non_blocking=True)

            # Model-only GPU latency.
            ev_start.record()
            pred = model(img, text)
            ev_end.record()

            torch.cuda.synchronize()
            model_ms = ev_start.elapsed_time(ev_end)

            pred = torch.sigmoid(pred)

            if pred.shape[-2:] != img.shape[-2:]:
                pred = F.interpolate(
                    pred,
                    size=img.shape[-2:],
                    mode="bicubic",
                    align_corners=True
                )

            pred = pred.squeeze()
            pred_np = pred.cpu().numpy()

            pred_np = cv2.warpAffine(
                pred_np,
                mat,
                (w, h),
                flags=cv2.INTER_CUBIC,
                borderValue=0.0
            )

            pred_mask = pred_np > 0.35

            torch.cuda.synchronize()
            query_pipeline_ms = (time.perf_counter() - q0) * 1000.0

            inter = np.logical_and(pred_mask, gt)
            union = np.logical_or(pred_mask, gt)

            I = int(inter.sum())
            U = int(union.sum())

            iou = I / (U + 1e-6)
            pred_area = int(pred_mask.sum())

            is_warmup = query_idx < args.warmup_queries

            rows.append({
                "query_idx": query_idx,
                "ref_idx": ref_idx,
                "seg_id": seg_id,
                "img_name": meta["img_name"],
                "category_idx": meta["cat"],
                "sent_idx": sent_idx,
                "expression": expression,
                "word_count": len(expression.split()),
                "clip_token_count": clip_token_count,
                "ori_h": h,
                "ori_w": w,
                "target_area_px": target_area,
                "target_area_ratio": target_area_ratio,
                "bbox_area_ratio": bbox_area_ratio,
                "pred_area_px": pred_area,
                "pred_area_ratio": pred_area / float(h * w),
                "intersection": I,
                "union": U,
                "iou": iou,
                "model_ms": model_ms,
                "query_pipeline_ms": query_pipeline_ms,
                "is_latency_warmup": int(is_warmup),
            })

            cum_I += I
            cum_U += U
            query_idx += 1

        if (ref_idx + 1) % 100 == 0:
            print(
                f"[{ref_idx + 1}/{len(loader)} refs] "
                f"{query_idx} queries"
            )

    wall_s = time.perf_counter() - wall_start

    ious = np.array([r["iou"] for r in rows], dtype=np.float64)

    latency_rows = [
        r for r in rows if not r["is_latency_warmup"]
    ]
    model_lat = np.array(
        [r["model_ms"] for r in latency_rows],
        dtype=np.float64
    )
    pipe_lat = np.array(
        [r["query_pipeline_ms"] for r in latency_rows],
        dtype=np.float64
    )

    summary = {
        "dataset": args.dataset,
        "split": args.test_split,
        "checkpoint": args.path,
        "num_refs": len(loader),
        "num_queries": len(rows),

        "mean_iou": float(ious.mean() * 100.0),
        "overall_iou": float(cum_I * 100.0 / cum_U),

        "pr50": float((ious > 0.50).mean() * 100.0),
        "pr60": float((ious > 0.60).mean() * 100.0),
        "pr70": float((ious > 0.70).mean() * 100.0),
        "pr80": float((ious > 0.80).mean() * 100.0),
        "pr90": float((ious > 0.90).mean() * 100.0),

        "latency_warmup_queries": args.warmup_queries,

        "model_ms_mean": float(model_lat.mean()),
        "model_ms_p50": float(np.percentile(model_lat, 50)),
        "model_ms_p95": float(np.percentile(model_lat, 95)),

        "query_pipeline_ms_mean": float(pipe_lat.mean()),
        "query_pipeline_ms_p50": float(np.percentile(pipe_lat, 50)),
        "query_pipeline_ms_p95": float(np.percentile(pipe_lat, 95)),

        "peak_vram_mb": float(
            torch.cuda.max_memory_allocated() / 1024**2
        ),

        "wall_time_s": wall_s,
        "wall_queries_per_s": len(rows) / wall_s,
    }

    return rows, summary


def main():
    args = get_args()

    out = Path(args.diag_out)
    out.mkdir(parents=True, exist_ok=True)

    ann_file = Path("datasets") / "anns" / args.dataset / f"{args.test_split}.json"

    with open(ann_file, "r") as f:
        anns = json.load(f)

    ann_map = {
        int(x["segment_id"]): x
        for x in anns
    }

    dataset = RefDataset(
        lmdb_dir=args.test_lmdb,
        mask_dir=args.mask_root,
        dataset=args.dataset,
        split=args.test_split,
        mode="test",
        input_size=args.input_size,
        word_length=args.word_len,
    )

    if args.max_refs > 0:
        n = min(args.max_refs, len(dataset))
        dataset = torch.utils.data.Subset(dataset, range(n))

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        pin_memory=True,
    )

    model, _ = build_segmenter(args)
    model = torch.nn.DataParallel(model).cuda()

    print("Loading checkpoint:", args.path)
    checkpoint = torch.load(args.path)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    print("Checkpoint loaded.")

    rows, summary = run_diagnostic(
        loader,
        model,
        args,
        ann_map
    )

    csv_path = out / "per_expression.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(rows[0].keys())
        )
        writer.writeheader()
        writer.writerows(rows)

    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n===== DIAGNOSTIC SUMMARY =====")
    print(json.dumps(summary, indent=2))
    print("\nCSV:", csv_path)


if __name__ == "__main__":
    main()
