#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot only mIoU curves (dataset-aggregated) with shaded uncertainty bands.

Modifications from previous version:
 - Legend moved to upper-left.
 - Baseline horizontal line is drawn at the aggregated baseline value (unchanged),
   but the legend label for the baseline is forced to display "Baseline 80.82%" as requested.
 - Output file name (Chinese) unchanged: 'mIoU曲线.png'.
"""
import os
import glob
import math
from typing import List, Tuple, Optional
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm


plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False
# ======================================================

# ---------------- Paths  ----------------
TEST_IMG_DIR = r"E:\MMseg\BIYELUNWEN\data\test\vis_images"
GT_LABEL_DIR = r"E:\MMseg\BIYELUNWEN\data\test\labels"
CACHE_DIR = r"E:\MMseg\BIYELUNWEN\data\vis_compare_top50_pgf_better\exp_45_panels_guidance_overlay_alpha_v1\cache_npz"
OUT_DIR = r"E:\MMseg\BIYELUNWEN\work_dirs\segformer_landslide\T\quant\miou_final"
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------- Settings ----------------
MAX_BUDGET = 0.50
N_POINTS = 51
R_FIXED = 0.20
N_RANDOM_TRIALS = 5
RNG_SEED = 0

# If you used ignore_index during evaluation, set it here (e.g. 255). Otherwise set None.
IGNORE_INDEX: Optional[int] = None

# Band mode: "sem" (default), "std", "percentile"
BAND_MODE = "sem"

# Methods and styles
METHODS = ["熵", "方差", "随机"]
COLORS = {"熵": "#1f77b4", "方差": "#ff7f0e", "随机": "#7f7f7f"}
LINESTYLES = {"熵": "-", "方差": "-", "随机": "--"}

# Legend baseline text (display only) ✅ 中文基准
BASELINE_LEGEND_TEXT = "基准模型 80.82%"


# ---------------- Helpers ----------------
def list_images(img_dir: str) -> List[str]:
    exts = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")
    return sorted([f for f in os.listdir(img_dir) if f.lower().endswith(exts)])


def stem_of(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def read_mask_as_array(path: str) -> np.ndarray:
    m = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if m is None:
        raise FileNotFoundError(path)
    if m.ndim == 3:
        m = m[..., 0]
    return m


def read_gt01_by_stem(stem: str) -> np.ndarray:
    candidates = glob.glob(os.path.join(GT_LABEL_DIR, stem + ".*"))
    if not candidates:
        raise FileNotFoundError(f"GT for {stem} not found in {GT_LABEL_DIR}")
    arr = read_mask_as_array(candidates[0])
    return np.isin(arr, [1, 255]).astype(np.uint8)


def load_cache_npz(stem: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    p = os.path.join(CACHE_DIR, stem + ".npz")
    if not os.path.exists(p):
        raise FileNotFoundError(p)
    a = np.load(p, allow_pickle=False)
    pred = a["pred"].astype(np.uint8)
    ent = a["ent"].astype(np.float32)
    var = a["var"].astype(np.float32)
    return pred, ent, var


def percentile_rank(arr: np.ndarray) -> np.ndarray:
    flat = arr.ravel()
    if flat.size <= 1:
        return np.zeros_like(arr, dtype=np.float64)
    order = flat.argsort()
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(flat))
    return (ranks / (len(flat) - 1)).reshape(arr.shape)


def select_topk_mask(scores_flat: np.ndarray, k: int) -> np.ndarray:
    n = scores_flat.size
    if k <= 0:
        return np.zeros(n, dtype=bool)
    if k >= n:
        return np.ones(n, dtype=bool)
    idx = np.argpartition(-scores_flat, k - 1)[:k]
    mask = np.zeros(n, dtype=bool);
    mask[idx] = True
    return mask


def mean_iou_from_counts(tp: int, fp: int, fn: int, tn: int, eps: float = 1e-12) -> float:
    iou1 = tp / (tp + fp + fn + eps)
    iou0 = tn / (tn + fn + fp + eps)
    return 0.5 * (iou0 + iou1)


# ---------------- Main ----------------
def main():
    np.random.seed(RNG_SEED)

    if not os.path.exists(CACHE_DIR):
        raise RuntimeError(f"CACHE_DIR not found: {CACHE_DIR}")

    img_files = list_images(TEST_IMG_DIR)
    stems = [stem_of(p) for p in img_files]

    valid = []
    for st in stems:
        if os.path.exists(os.path.join(CACHE_DIR, st + ".npz")) and glob.glob(os.path.join(GT_LABEL_DIR, st + ".*")):
            valid.append(st)
    if not valid:
        raise RuntimeError("No valid stems with cache and GT")

    budgets = np.linspace(0.0, MAX_BUDGET, N_POINTS)
    n_b = len(budgets)

    # totals for dataset-level aggregated mIoU
    totals = {m: {"tp": np.zeros(n_b, dtype=np.int64),
                  "fp": np.zeros(n_b, dtype=np.int64),
                  "fn": np.zeros(n_b, dtype=np.int64),
                  "tn": np.zeros(n_b, dtype=np.int64)} for m in METHODS}

    # per-image corrected mIoU arrays for computing SEM of differences
    per_image_miou = {m: [] for m in METHODS}

    # baseline aggregated totals
    base_tp = base_fp = base_fn = base_tn = 0

    rng = np.random.RandomState(RNG_SEED)

    for st in tqdm(valid, desc="Processing images"):
        pred, ent, var = load_cache_npz(st)
        gt = read_gt01_by_stem(st)

        # possibly remove ignored pixels
        if IGNORE_INDEX is not None:
            valid_mask = (gt != IGNORE_INDEX)
            p_flat = pred.ravel()[valid_mask.ravel()].astype(np.uint8)
            g_flat = gt.ravel()[valid_mask.ravel()].astype(np.uint8)
            ent_flat = ent.ravel()[valid_mask.ravel()]
            var_flat = var.ravel()[valid_mask.ravel()]
        else:
            p_flat = pred.ravel().astype(np.uint8)
            g_flat = gt.ravel().astype(np.uint8)
            ent_flat = ent.ravel()
            var_flat = var.ravel()
        npx = p_flat.size

        # accumulate baseline totals
        base_tp += int(((p_flat == 1) & (g_flat == 1)).sum())
        base_fp += int(((p_flat == 1) & (g_flat == 0)).sum())
        base_fn += int(((p_flat == 0) & (g_flat == 1)).sum())
        base_tn += int(((p_flat == 0) & (g_flat == 0)).sum())

        ent_score = percentile_rank(ent_flat)
        var_score = percentile_rank(var_flat)
        rand_orders = [rng.permutation(npx) for _ in range(N_RANDOM_TRIALS)]

        # per-image corrected mIoU arrays
        miou_ent_img = np.zeros(n_b, dtype=np.float64)
        miou_var_img = np.zeros(n_b, dtype=np.float64)
        miou_rand_img = np.zeros(n_b, dtype=np.float64)

        for i_b, b in enumerate(budgets):
            k = int(round(b * npx))

            # entropy selected pixels -> corrected pred
            sel_ent = select_topk_mask(ent_score, k)
            pred_corr_ent = p_flat.copy()
            pred_corr_ent[sel_ent] = g_flat[sel_ent]
            tp_e = int(((pred_corr_ent == 1) & (g_flat == 1)).sum())
            fp_e = int(((pred_corr_ent == 1) & (g_flat == 0)).sum())
            fn_e = int(((pred_corr_ent == 0) & (g_flat == 1)).sum())
            tn_e = int(((pred_corr_ent == 0) & (g_flat == 0)).sum())
            totals["熵"]["tp"][i_b] += tp_e;
            totals["熵"]["fp"][i_b] += fp_e
            totals["熵"]["fn"][i_b] += fn_e;
            totals["熵"]["tn"][i_b] += tn_e
            miou_ent_img[i_b] = mean_iou_from_counts(tp_e, fp_e, fn_e, tn_e)

            # variance
            sel_var = select_topk_mask(var_score, k)
            pred_corr_var = p_flat.copy()
            pred_corr_var[sel_var] = g_flat[sel_var]
            tp_v = int(((pred_corr_var == 1) & (g_flat == 1)).sum())
            fp_v = int(((pred_corr_var == 1) & (g_flat == 0)).sum())
            fn_v = int(((pred_corr_var == 0) & (g_flat == 1)).sum())
            tn_v = int(((pred_corr_var == 0) & (g_flat == 0)).sum())
            totals["方差"]["tp"][i_b] += tp_v;
            totals["方差"]["fp"][i_b] += fp_v
            totals["方差"]["fn"][i_b] += fn_v;
            totals["方差"]["tn"][i_b] += tn_v
            miou_var_img[i_b] = mean_iou_from_counts(tp_v, fp_v, fn_v, tn_v)

            # random (average across trials)
            if k == 0:
                tp_r = int(((p_flat == 1) & (g_flat == 1)).sum())
                fp_r = int(((p_flat == 1) & (g_flat == 0)).sum())
                fn_r = int(((p_flat == 0) & (g_flat == 1)).sum())
                tn_r = int(((p_flat == 0) & (g_flat == 0)).sum())
                totals["随机"]["tp"][i_b] += tp_r;
                totals["随机"]["fp"][i_b] += fp_r
                totals["随机"]["fn"][i_b] += fn_r;
                totals["随机"]["tn"][i_b] += tn_r
                miou_rand_img[i_b] = mean_iou_from_counts(tp_r, fp_r, fn_r, tn_r)
            else:
                tp_acc = fp_acc = fn_acc = tn_acc = 0
                for t in range(N_RANDOM_TRIALS):
                    order = rand_orders[t]
                    sel_idx = order[:k]
                    sel_mask = np.zeros(npx, dtype=bool);
                    sel_mask[sel_idx] = True
                    pred_corr_r = p_flat.copy();
                    pred_corr_r[sel_mask] = g_flat[sel_mask]
                    tp_acc += int(((pred_corr_r == 1) & (g_flat == 1)).sum())
                    fp_acc += int(((pred_corr_r == 1) & (g_flat == 0)).sum())
                    fn_acc += int(((pred_corr_r == 0) & (g_flat == 1)).sum())
                    tn_acc += int(((pred_corr_r == 0) & (g_flat == 0)).sum())
                tp_avg = int(round(tp_acc / N_RANDOM_TRIALS))
                fp_avg = int(round(fp_acc / N_RANDOM_TRIALS))
                fn_avg = int(round(fn_acc / N_RANDOM_TRIALS))
                tn_avg = int(round(tn_acc / N_RANDOM_TRIALS))
                totals["随机"]["tp"][i_b] += tp_avg;
                totals["随机"]["fp"][i_b] += fp_avg
                totals["随机"]["fn"][i_b] += fn_avg;
                totals["随机"]["tn"][i_b] += tn_avg
                miou_rand_img[i_b] = mean_iou_from_counts(tp_avg, fp_avg, fn_avg, tn_avg)

        per_image_miou["熵"].append(miou_ent_img)
        per_image_miou["方差"].append(miou_var_img)
        per_image_miou["随机"].append(miou_rand_img)

    # aggregated baseline (dataset-level, mmseg-style)
    baseline_agg = mean_iou_from_counts(base_tp, base_fp, base_fn, base_tn)
    print(f"Computed dataset baseline (aggregated) mIoU: {baseline_agg * 100:.2f}%")
    print(f"Note: legend will display: {BASELINE_LEGEND_TEXT}")


    budgets = np.linspace(0.0, MAX_BUDGET, N_POINTS)
    miou_agg = {}
    band_low = {}
    band_high = {}
    for m in METHODS:
        tp_arr = totals[m]["tp"];
        fp_arr = totals[m]["fp"];
        fn_arr = totals[m]["fn"];
        tn_arr = totals[m]["tn"]
        miou_agg[m] = np.array(
            [mean_iou_from_counts(int(tp_arr[i]), int(fp_arr[i]), int(fn_arr[i]), int(tn_arr[i])) for i in
             range(len(budgets))])
        per_img = np.vstack(per_image_miou[m])  # shape (n_images, n_budgets)
        if BAND_MODE == "sem":
            sem_diff = per_img.std(axis=0) / math.sqrt(max(1, per_img.shape[0]))
            low = np.maximum(0.0, miou_agg[m] - sem_diff)
            high = np.minimum(1.0, miou_agg[m] + sem_diff)
        elif BAND_MODE == "std":
            sd = per_img.std(axis=0)
            low = np.maximum(0.0, miou_agg[m] - sd)
            high = np.minimum(1.0, miou_agg[m] + sd)
        elif BAND_MODE == "percentile":
            low = np.percentile(per_img, 5, axis=0)
            high = np.percentile(per_img, 95, axis=0)
        else:
            raise ValueError("Unknown BAND_MODE")
        band_low[m] = low
        band_high[m] = high


    all_low = min((band_low[m].min() for m in METHODS))
    all_high = max((band_high[m].max() for m in METHODS))
    ymin_pct = max(0.0, all_low * 100.0 - 2.0)
    ymax_pct = min(100.0, all_high * 100.0 + 2.0)
    if ymax_pct - ymin_pct < 8.0:
        mid = (ymax_pct + ymin_pct) / 2.0
        ymin_pct = max(0.0, mid - 4.0)
        ymax_pct = min(100.0, mid + 4.0)


    plt.figure(figsize=(8, 5))
    for m in METHODS:
        plt.plot(budgets, miou_agg[m] * 100, label=m, color=COLORS[m], linestyle=LINESTYLES[m], linewidth=2.2)
        plt.fill_between(budgets, band_low[m] * 100, band_high[m] * 100, color=COLORS[m], alpha=0.12)
    # baseline horizontal line drawn at the aggregated baseline but legend text shows requested value
    plt.axhline(baseline_agg * 100, color="k", linestyle="--", linewidth=1.2, label=BASELINE_LEGEND_TEXT)
    plt.xlim(0, MAX_BUDGET);
    plt.ylim(ymin_pct, ymax_pct)


    plt.xlabel("复核比例（图像像素占比）")
    plt.ylabel("平均交并比（mIoU）%")
    plt.legend(loc="upper left")
    plt.grid(alpha=0.2)

    out_png = os.path.join(OUT_DIR, "mIoU曲线.png")
    plt.tight_layout()
    plt.savefig(out_png, dpi=600)
    plt.close()
    print("Saved:", out_png)

    # save CSV summary (aggregated mainline and band)
    csv_path = os.path.join(OUT_DIR, "budget_curve_summary_miou.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        import csv as _csv
        writer = _csv.writer(f)
        header = ["budget"]
        for m in METHODS:
            header += [f"{m}_miou_agg_pct", f"{m}_band_low_pct", f"{m}_band_high_pct"]
        writer.writerow(header)
        for i, b in enumerate(budgets):
            row = [f"{b:.6f}"]
            for m in METHODS:
                row += [f"{miou_agg[m][i] * 100:.4f}", f"{band_low[m][i] * 100:.4f}", f"{band_high[m][i] * 100:.4f}"]
            writer.writerow(row)
    print("Saved CSV:", csv_path)
    print("Done. Outputs in:", OUT_DIR)


if __name__ == "__main__":
    main()