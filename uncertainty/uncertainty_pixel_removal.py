#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v_alpha: 用标准线性 alpha 混合做 overlay，避免加法饱和导致颜色不一致。
生成面板（5 列）：
  1) RGB 原图
  2) Baseline error map（白=TP，品红=FN，蓝=FP）
  3) Entropy-oracle（pred 分组 top-R -> 用 GT 替换后画错误图）
  4) Variance-oracle（pred 分组 top-R -> 用 GT 替换后画错误图）
  5) Guidance-overlay（原图上半透明叠加：ent_region 红，var_secondary 黄；candidate = ent_region ∪ var_region；不做 CC 过滤）

说明：
 - 使用标准 alpha 混合：out = (1-alpha)*orig + alpha*overlay（仅在掩膜处混合）
 - 缓存 MC Dropout 结果到 CACHE_DIR，加速重复运行
 - 修改顶部路径与模型配置后直接运行
"""
import os
import os.path as osp
import glob
import csv
from dataclasses import dataclass, asdict
from typing import List

import numpy as np
import cv2
import torch
import torch.nn.functional as F
# 修复matplotlib 中文乱码 + 负号正常显示
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

from mmseg.apis import init_model, inference_model

# ----------------------------
# 配置
# ----------------------------
TEST_IMG_DIR = r"E:\MMseg\BIYELUNWEN\data\test\vis_images"
TEST_GT_DIR  = r"E:\MMseg\BIYELUNWEN\data\test\labels"
GT_SUFFIX    = ".png"
GT_POS_VALUES = {1, 255}   # 若 GT 为 0/1 改为 {1}

CONFIG_FILE = r"E:\MMseg\mmsegmentation\configs\segformer\segformer_mit-b1_landslide2.py"
CHECKPOINT_FILE = r"E:\MMseg\BIYELUNWEN\work_dirs\segformer_landslide\20260121_000000\segformer_landslide\best_mIoU_iter_93000.pth"
DEVICE = "cuda:0"

# MC Dropout
T = 20
POS_CLASS_ID = 1

# 分组阈值
R_group = 0.20  # pred==1 top-20% ; pred==0 top-20%

# 输出路径
OUT_ROOT = r"E:\MMseg\BIYELUNWEN\data\vis_compare_top50_pgf_better"
OUT_DIR = osp.join(OUT_ROOT, "exp_45_panels_guidance_overlay_alpha_v1")
CACHE_DIR = osp.join(OUT_DIR, "cache_npz")
PANELS_DIR = osp.join(OUT_DIR, "panels")
CSV_PATH = osp.join(OUT_DIR, "per_image_metrics_guidance_overlay_alpha_v1.csv")
THR_PATH = osp.join(OUT_DIR, "thresholds_info_alpha_v1.txt")
TOPK_EXPORT = 260
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(PANELS_DIR, exist_ok=True)

# 颜色（BGR）
BLACK = (0,0,0); WHITE = (255,255,255); RED = (0,0,255); BLUE = (255,0,0); YELLOW = (0,255,255)
MAGENTA = (255,0,255)
OVERLAY_ALPHA = 0.72

# ----------------------------
# 辅助函数
# ----------------------------
def list_images(p):
    files=[]
    for e in ("*.png","*.jpg","*.jpeg","*.tif","*.tiff","*.bmp"):
        files.extend(glob.glob(osp.join(p,e)))
    return sorted(files)

def stem_of(p): return osp.splitext(osp.basename(p))[0]

def find_gt_by_stem(stem):
    p = osp.join(TEST_GT_DIR, stem+GT_SUFFIX)
    if osp.exists(p): return p
    hits = sorted(glob.glob(osp.join(TEST_GT_DIR, stem+".*")))
    if hits: return hits[0]
    raise FileNotFoundError(f"GT for {stem} not found")

def read_gt01(p):
    m = cv2.imread(p, cv2.IMREAD_UNCHANGED)
    if m is None: raise FileNotFoundError(p)
    if m.ndim==3: m = m[...,0]
    return np.isin(m, list(GT_POS_VALUES)).astype(np.uint8)

def enable_dropout_only(model):
    model.eval()
    for mm in model.modules():
        if "dropout" in mm.__class__.__name__.lower():
            mm.train()
    return model

@torch.no_grad()
def inference_prob_map(model, img_path, pos_class_id):
    res = inference_model(model, img_path)
    seg_logits = None
    if hasattr(res, "seg_logits") and res.seg_logits is not None:
        seg_logits = res.seg_logits.data
    elif hasattr(res, "logits") and res.logits is not None:
        seg_logits = res.logits.data
    if seg_logits is None:
        raise RuntimeError("inference_model() did not provide seg_logits/logits")
    if seg_logits.dim()==4: seg_logits = seg_logits[0]
    prob = F.softmax(seg_logits, dim=0)[pos_class_id]
    return prob.cpu().numpy().astype(np.float32)

@torch.no_grad()
def mc_dropout_maps(model, img_path, T, pos_class_id):
    enable_dropout_only(model)
    probs = [torch.from_numpy(inference_prob_map(model, img_path, pos_class_id)) for _ in range(T)]
    probs_t = torch.stack(probs, dim=0)
    p_mean = probs_t.mean(dim=0)
    var = probs_t.var(dim=0, unbiased=False)
    eps = 1e-8
    p = torch.clamp(p_mean, eps, 1-eps)
    ent = -(p * torch.log(p) + (1-p) * torch.log(1-p))
    pred01 = (p_mean >= 0.5).to(torch.uint8)
    return pred01.cpu().numpy(), ent.cpu().numpy().astype(np.float32), var.cpu().numpy().astype(np.float32)

def cache_load_or_calc(model, img_path):
    st = stem_of(img_path)
    cache = osp.join(CACHE_DIR, st + ".npz")
    if osp.exists(cache):
        a = np.load(cache)
        return a["pred"].astype(np.uint8), a["ent"].astype(np.float32), a["var"].astype(np.float32)
    pred, ent, var = mc_dropout_maps(model, img_path, T, POS_CLASS_ID)
    np.savez_compressed(cache, pred=pred, ent=ent, var=var)
    return pred, ent, var

def quantile_threshold(values, r):
    if values.size == 0: return float("inf")
    return float(np.quantile(values, 1.0 - r))

def baseline_error_map(pred, gt):
    h,w = pred.shape
    out = np.zeros((h,w,3), dtype=np.uint8)
    tp = (pred==1)&(gt==1); fp = (pred==1)&(gt==0); fn = (pred==0)&(gt==1)
    out[tp] = WHITE; out[fp] = BLUE; out[fn] = MAGENTA
    return out

def oracle_correct(pred, gt, region01):
    out = pred.copy()
    m = region01.astype(bool)
    out[m] = gt[m]
    return out

def confusion_counts(pred, gt):
    tp = int(((pred==1)&(gt==1)).sum()); fp = int(((pred==1)&(gt==0)).sum())
    fn = int(((pred==0)&(gt==1)).sum()); tn = int(((pred==0)&(gt==0)).sum())
    return {"tp":tp,"fp":fp,"fn":fn,"tn":tn}

def iou_from_counts(tp,fp,fn):
    den = tp+fp+fn
    return float(tp/den) if den>0 else 0.0

# ----------------------------
# 掩膜生成 + 标准 alpha 混合 overlay
# ----------------------------
def make_uncertainty_masks(pred, ent, var, thr_ent_pred1, thr_ent_pred0, thr_var_pred1, thr_var_pred0):
    """
    返回 uint8 (0/1) 掩膜：
      ent_region: 熵候选（pred==1 & ent>=thr_ent_pred1) OR (pred==0 & ent>=thr_ent_pred0)
      var_region: 方差候选（pred==1 & var>=thr_var_pred1) OR (pred==0 & var>=thr_var_pred0)
      var_secondary: var_region 去掉与 ent_region 重合部分
      union_region: ent_region OR var_region
    """
    ent_rej = ((pred == 1) & (ent >= thr_ent_pred1))
    ent_rev = ((pred == 0) & (ent >= thr_ent_pred0))
    ent_region = (ent_rej | ent_rev).astype(np.uint8)

    var_rej = ((pred == 1) & (var >= thr_var_pred1))
    var_rev = ((pred == 0) & (var >= thr_var_pred0))
    var_region = (var_rej | var_rev).astype(np.uint8)

    var_secondary = (var_region.astype(bool) & (~ent_region.astype(bool))).astype(np.uint8)
    union_region = ((ent_region.astype(bool)) | (var_region.astype(bool))).astype(np.uint8)

    return ent_region, var_region, var_secondary, union_region

def overlay_review_on_rgb_alpha(rgb_bgr: np.ndarray, ent_region01: np.ndarray, var_secondary01: np.ndarray, alpha: float = OVERLAY_ALPHA) -> np.ndarray:
    """
    使用线性 alpha 混合（(1-alpha)*orig + alpha*overlay），仅在掩膜处混合。
    Inputs:
      - rgb_bgr: HxWx3 uint8
      - ent_region01: HxW uint8 (0/1)
      - var_secondary01: HxW uint8 (0/1)
      - alpha: 0..1
    """
    out = rgb_bgr.astype(np.float32)
    overlay = np.zeros_like(out, dtype=np.float32)

    # BGR colors as floats
    RED_f = np.array([0.0, 0.0, 255.0], dtype=np.float32)
    YELLOW_f = np.array([0.0, 255.0, 255.0], dtype=np.float32)

    mask_ent = ent_region01.astype(bool)
    mask_var = var_secondary01.astype(bool)
    mask_any = mask_ent | mask_var
    if not mask_any.any():
        return rgb_bgr.copy()

    overlay[mask_ent] = RED_f
    overlay[mask_var] = YELLOW_f

    out_mask = out[mask_any]            # N x 3
    overlay_mask = overlay[mask_any]    # N x 3
    blended = (1.0 - alpha) * out_mask + alpha * overlay_mask
    out[mask_any] = blended

    out_uint8 = np.clip(out, 0, 255).astype(np.uint8)
    return out_uint8

# ----------------------------
# 主流程
# ----------------------------
@dataclass
class Row:
    stem:str; h:int; w:int
    baseline_fp:int; baseline_fn:int; baseline_miou:float
    ent_review_rate:float; ent_fp:int; ent_fn:int; ent_miou:float; ent_fp_reduction:int; ent_fn_reduction:int; ent_miou_delta:float
    var_review_rate:float; var_fp:int; var_fn:int; var_miou:float; var_fp_reduction:int; var_fn_reduction:int; var_miou_delta:float
    guide_review_rate:float; guide_fp:int; guide_fn:int; guide_miou:float; guide_fp_reduction:int; guide_fn_reduction:int; guide_miou_delta:float
    best_score:float; best_method:str

def main():
    print("OUT_DIR:", OUT_DIR)
    os.makedirs(OUT_DIR, exist_ok=True); os.makedirs(PANELS_DIR, exist_ok=True)

    model = init_model(CONFIG_FILE, CHECKPOINT_FILE, device=DEVICE)
    img_files = list_images(TEST_IMG_DIR)
    if not img_files:
        raise RuntimeError("No test images")
    print("Total images:", len(img_files))

    # ---------- 1) 计算分组阈值（pred==1 / pred==0 各取 R_group） ----------
    ent_pred1_all, ent_pred0_all = [], []
    var_pred1_all, var_pred0_all = [], []
    for p in img_files:
        pred, ent, var = cache_load_or_calc(model, p)
        m1 = (pred==1); m0 = (pred==0)
        if ent[m1].size > 0:
            ent_pred1_all.append(ent[m1].reshape(-1))
        if ent[m0].size > 0:
            ent_pred0_all.append(ent[m0].reshape(-1))
        if var[m1].size > 0:
            var_pred1_all.append(var[m1].reshape(-1))
        if var[m0].size > 0:
            var_pred0_all.append(var[m0].reshape(-1))

    ent_pred1_all = np.concatenate(ent_pred1_all) if len(ent_pred1_all)>0 else np.array([])
    ent_pred0_all = np.concatenate(ent_pred0_all) if len(ent_pred0_all)>0 else np.array([])
    var_pred1_all = np.concatenate(var_pred1_all) if len(var_pred1_all)>0 else np.array([])
    var_pred0_all = np.concatenate(var_pred0_all) if len(var_pred0_all)>0 else np.array([])

    thr_ent_pred1 = quantile_threshold(ent_pred1_all, R_group)
    thr_ent_pred0 = quantile_threshold(ent_pred0_all, R_group)
    thr_var_pred1 = quantile_threshold(var_pred1_all, R_group)
    thr_var_pred0 = quantile_threshold(var_pred0_all, R_group)

    with open(THR_PATH, "w", encoding="utf-8") as f:
        f.write(f"R_group={R_group}\n")
        f.write(f"thr_ent_pred1={thr_ent_pred1:.12f}\nthr_ent_pred0={thr_ent_pred0:.12f}\n")
        f.write(f"thr_var_pred1={thr_var_pred1:.12f}\nthr_var_pred0={thr_var_pred0:.12f}\n")
        f.write(f"OVERLAY_ALPHA={OVERLAY_ALPHA:.4f}\n")
    print("Saved thresholds info:", THR_PATH)

    # ---------- 2) 遍历每张图，计算三种 oracle & 导出 panel ----------
    rows = []
    for idx, p in enumerate(img_files, 1):
        st = stem_of(p)
        gt = read_gt01(find_gt_by_stem(st))
        pred, ent, var = cache_load_or_calc(model, p)
        h,w = pred.shape

        base_cnt = confusion_counts(pred, gt)
        base_miou = iou_from_counts(base_cnt["tp"], base_cnt["fp"], base_cnt["fn"])

        # create masks (ent/var/var_secondary/union) per pixel
        ent_region, var_region, var_secondary, union_region = make_uncertainty_masks(
            pred, ent, var, thr_ent_pred1, thr_ent_pred0, thr_var_pred1, thr_var_pred0
        )

        # ent oracle (grouped)
        pred_ent_oracle = oracle_correct(pred, gt, ent_region)
        ent_cnt = confusion_counts(pred_ent_oracle, gt)
        ent_miou = iou_from_counts(ent_cnt["tp"], ent_cnt["fp"], ent_cnt["fn"])
        ent_fp_red = base_cnt["fp"] - ent_cnt["fp"]; ent_fn_red = base_cnt["fn"] - ent_cnt["fn"]
        ent_review_rate = float(ent_region.sum() / (h*w))

        # var oracle (grouped)
        pred_var_oracle = oracle_correct(pred, gt, var_region)
        var_cnt = confusion_counts(pred_var_oracle, gt)
        var_miou = iou_from_counts(var_cnt["tp"], var_cnt["fp"], var_cnt["fn"])
        var_fp_red = base_cnt["fp"] - var_cnt["fp"]; var_fn_red = base_cnt["fn"] - var_cnt["fn"]
        var_review_rate = float(var_region.sum() / (h*w))

        # guidance (union) oracle metrics (for CSV)
        pred_guidance_oracle = oracle_correct(pred, gt, union_region)
        guide_cnt = confusion_counts(pred_guidance_oracle, gt)
        guide_miou = iou_from_counts(guide_cnt["tp"], guide_cnt["fp"], guide_cnt["fn"])
        guide_fp_red = base_cnt["fp"] - guide_cnt["fp"]; guide_fn_red = base_cnt["fn"] - guide_cnt["fn"]
        guide_review_rate = float(union_region.sum() / (h*w))

        # scoring (用于排序)
        ent_score = (1.2*(ent_fp_red/max(base_cnt["fp"],1)) + 1.2*(ent_fn_red/max(base_cnt["fn"],1))) - 2.5*(((ent_region.astype(bool)) & (pred==0)).sum() / max(int((pred==0).sum()),1)) - 0.6*ent_review_rate
        var_score = (1.2*(var_fp_red/max(base_cnt["fp"],1)) + 1.2*(var_fn_red/max(base_cnt["fn"],1))) - 2.5*(((var_region.astype(bool)) & (pred==0)).sum() / max(int((pred==0).sum()),1)) - 0.6*var_review_rate
        guide_score = (1.2*(guide_fp_red/max(base_cnt["fp"],1)) + 1.2*(guide_fn_red/max(base_cnt["fn"],1))) - 2.5*(((union_region.astype(bool)) & (pred==0)).sum() / max(int((pred==0).sum()),1)) - 0.6*guide_review_rate
        best_score, best_method = max( (ent_score,"entropy"), (var_score,"variance"), (guide_score,"guidance") )

        rows.append(Row(
            stem=st, h=h, w=w,
            baseline_fp=base_cnt["fp"], baseline_fn=base_cnt["fn"], baseline_miou=base_miou,
            ent_review_rate=ent_review_rate, ent_fp=ent_cnt["fp"], ent_fn=ent_cnt["fn"], ent_miou=ent_miou,
            ent_fp_reduction=int(ent_fp_red), ent_fn_reduction=int(ent_fn_red), ent_miou_delta=float(ent_miou - base_miou),
            var_review_rate=var_review_rate, var_fp=var_cnt["fp"], var_fn=var_cnt["fn"], var_miou=var_miou,
            var_fp_reduction=int(var_fp_red), var_fn_reduction=int(var_fn_red), var_miou_delta=float(var_miou - base_miou),
            guide_review_rate=guide_review_rate, guide_fp=guide_cnt["fp"], guide_fn=guide_cnt["fn"], guide_miou=guide_miou,
            guide_fp_reduction=int(guide_fp_red), guide_fn_reduction=int(guide_fn_red), guide_miou_delta=float(guide_miou - base_miou),
            best_score=best_score, best_method=best_method
        ))

        if idx % 50 == 0 or idx == len(img_files):
            print(f"[process] {idx}/{len(img_files)} done")

    # ---------- 写 CSV ----------
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        w.writeheader()
        for r in rows:
            w.writerow(asdict(r))
    print("Saved CSV:", CSV_PATH)

    # ---------- 导出面板（按 best_score 排序，5 列：RGB | Baseline | Ent-oracle | Var-oracle | Guidance-overlay） ----------
    rows_sorted = sorted(rows, key=lambda x: x.best_score, reverse=True)
    exported = 0
    for rank_idx, r in enumerate(rows_sorted[:TOPK_EXPORT], 1):
        st = r.stem
        # locate image
        img_path = None
        for ext in (".png",".jpg",".jpeg",".tif",".tiff",".bmp"):
            p = osp.join(TEST_IMG_DIR, st+ext)
            if osp.exists(p): img_path = p; break
        if img_path is None: continue

        rgb = cv2.imread(img_path, cv2.IMREAD_COLOR)
        gt = read_gt01(find_gt_by_stem(st))
        pred, ent, var = cache_load_or_calc(model, img_path)
        h,w = pred.shape

        # recompute masks for this image (consistent with CSV)
        ent_region, var_region, var_secondary, union_region = make_uncertainty_masks(
            pred, ent, var, thr_ent_pred1, thr_ent_pred0, thr_var_pred1, thr_var_pred0
        )

        # baseline col
        col1 = rgb
        col2 = baseline_error_map(pred, gt)

        # ent-oracle col (group-20%)
        pred_ent_oracle = oracle_correct(pred, gt, ent_region)
        col3 = baseline_error_map(pred_ent_oracle, gt)

        # var-oracle col (group-20%)
        pred_var_oracle = oracle_correct(pred, gt, var_region)
        col4 = baseline_error_map(pred_var_oracle, gt)

        # guidance overlay col: union (no CC filtering, no GT restriction), overlay ent red, var_secondary yellow
        # use standard alpha blend
        col5 = overlay_review_on_rgb_alpha(rgb, ent_region, var_secondary, alpha=OVERLAY_ALPHA)

        gap_h = 12
        gap = np.full((gap_h, w, 3), (192, 192, 192), dtype=np.uint8)

        panel = np.vstack([col1, gap, col2, gap, col3, gap, col4, gap, col5])
        out_name = f"{rank_idx:04d}_{st}_best-{r.best_method}_score{r.best_score:.3f}.png"
        out_path = osp.join(PANELS_DIR, out_name)
        cv2.imwrite(out_path, panel)
        if not osp.exists(out_path):
            raise RuntimeError(f"Failed to write {out_path}")
        exported += 1

    print(f"Exported {exported} panels -> {PANELS_DIR}")
    print("Done.")

if __name__ == "__main__":
    main()