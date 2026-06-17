import os
import time
import math
import numpy as np
import torch
import matplotlib.pyplot as plt

# ===================== 修复中文显示（核心！） =====================
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False
# ================================================================

from mmseg.apis import init_model, inference_model

# ---------------- 固定路径（按你的工程） ----------------
CONFIG_FILE = r"E:\MMseg\mmsegmentation\configs\segformer\segformer_mit-b1_landslide2.py"
CHECKPOINT_FILE = r"E:\MMseg\BIYELUNWEN\work_dirs\segformer_landslide\20260121_000000\segformer_landslide\best_mIoU_iter_93000.pth"

TEST_IMG_DIR = r"E:\MMseg\BIYELUNWEN\data\test\vis_images"
GT_LABEL_DIR = r"E:\MMseg\BIYELUNWEN\data\test\labels"

OUT_ROOT = r"E:\MMseg\BIYELUNWEN\work_dirs\segformer_landslide\T"
OUT_DIR = os.path.join(OUT_ROOT, "quant")

# ---------------- 实验设置 ----------------
DEVICE = "cuda:0"
T_MC = 20
LANDSLIDE_IDX = 1
IGNORE_INDEX = 255
BINS = 10

REJECT_RATES = [0.0, 0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50]

BOXPLOT_SAMPLE_PER_GROUP_PER_IMAGE = 3000
BOXPLOT_SEED = 20260217


# ---------------- 基础工具 ----------------
def ensure_dirs():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR, "fig"), exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR, "csv"), exist_ok=True)


def enable_dropout(m):
    if isinstance(m, (torch.nn.Dropout, torch.nn.Dropout2d, torch.nn.Dropout3d)):
        m.train()


def list_images(img_dir: str):
    exts = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")
    return sorted([f for f in os.listdir(img_dir) if f.lower().endswith(exts)])


def read_mask(mask_path: str) -> np.ndarray:
    import cv2
    m = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
    if m is None:
        raise FileNotFoundError(f"Failed to read mask: {mask_path}")
    if m.ndim == 3:
        m = m[..., 0]
    return m.astype(np.int64)


def softmax_prob_from_logits(logits: torch.Tensor, landslide_idx: int) -> np.ndarray:
    C = logits.shape[0]
    if C == 1:
        prob = torch.sigmoid(logits)[0]
    else:
        prob = torch.softmax(logits, dim=0)[landslide_idx]
    return prob.detach().cpu().numpy().astype(np.float32)


def mc_maps(model, img_path: str, T: int, landslide_idx: int = 1, eps: float = 1e-8):
    probs = []
    for _ in range(T):
        result = inference_model(model, img_path)
        logits = result.seg_logits.data
        if logits.ndim == 4:
            logits = logits[0]
        probs.append(softmax_prob_from_logits(logits, landslide_idx))
    probs = np.stack(probs, axis=0)

    p_mean = probs.mean(axis=0).astype(np.float32)
    var_map = probs.var(axis=0).astype(np.float32)
    ent_map = (-(p_mean * np.log(p_mean + eps) + (1 - p_mean) * np.log(1 - p_mean + eps))).astype(np.float32)
    return p_mean, ent_map, var_map


def pred_mask_from_pmean(p_mean: np.ndarray, thr: float = 0.5) -> np.ndarray:
    return (p_mean >= thr).astype(np.uint8)


def apply_ignore(mask: np.ndarray, ignore_index):
    if ignore_index is None:
        return np.ones_like(mask, dtype=bool)
    return (mask != ignore_index)


def confusion_binary(pred: np.ndarray, gt: np.ndarray, valid: np.ndarray):
    p = pred[valid].astype(np.uint8)
    g = gt[valid].astype(np.uint8)
    tp = int(((p == 1) & (g == 1)).sum())
    fp = int(((p == 1) & (g == 0)).sum())
    fn = int(((p == 0) & (g == 1)).sum())
    tn = int(((p == 0) & (g == 0)).sum())
    return tp, fp, fn, tn


def iou_from_conf(tp, fp, fn, eps=1e-12):
    return tp / (tp + fp + fn + eps)


def f1_from_conf(tp, fp, fn, eps=1e-12):
    return (2 * tp) / (2 * tp + fp + fn + eps)


def miou_binary(pred: np.ndarray, gt: np.ndarray, valid: np.ndarray):
    p = pred[valid].astype(np.uint8)
    g = gt[valid].astype(np.uint8)

    tp1 = int(((p == 1) & (g == 1)).sum())
    fp1 = int(((p == 1) & (g == 0)).sum())
    fn1 = int(((p == 0) & (g == 1)).sum())
    iou1 = iou_from_conf(tp1, fp1, fn1)

    tp0 = int(((p == 0) & (g == 0)).sum())
    fp0 = int(((p == 0) & (g == 1)).sum())
    fn0 = int(((p == 1) & (g == 0)).sum())
    iou0 = iou_from_conf(tp0, fp0, fn0)

    return 0.5 * (iou0 + iou1), iou1


def pixel_error_rate(pred: np.ndarray, gt: np.ndarray, valid: np.ndarray) -> float:
    p = pred[valid].astype(np.uint8)
    g = gt[valid].astype(np.uint8)
    return float(np.mean(p != g))


def save_reliability_diagram(probs, labels, bins, out_path):
    probs = probs.astype(np.float64)
    labels = labels.astype(np.int64)

    edges = np.linspace(0.0, 1.0, bins + 1)
    bin_ids = np.digitize(probs, edges[1:-1], right=False)

    acc = np.zeros(bins, dtype=np.float64)
    conf = np.zeros(bins, dtype=np.float64)

    for b in range(bins):
        m = (bin_ids == b)
        if m.any():
            conf[b] = probs[m].mean()
            acc[b] = labels[m].mean()
        else:
            conf[b] = (edges[b] + edges[b + 1]) / 2
            acc[b] = np.nan

    plt.figure(figsize=(5, 5))
    plt.plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
    plt.plot(conf, acc, "-o", color="tab:blue")
    plt.xlabel("置信度")
    plt.ylabel("准确率")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=1000)
    plt.close()


def ece_score(probs, labels, bins):
    probs = probs.astype(np.float64)
    labels = labels.astype(np.int64)
    edges = np.linspace(0.0, 1.0, bins + 1)
    bin_ids = np.digitize(probs, edges[1:-1], right=False)

    ece = 0.0
    n = len(probs)
    for b in range(bins):
        m = (bin_ids == b)
        if not m.any():
            continue
        conf = probs[m].mean()
        acc = labels[m].mean()
        ece += (m.sum() / n) * abs(acc - conf)
    return float(ece)


def brier_score(probs, labels):
    probs = probs.astype(np.float64)
    labels = labels.astype(np.float64)
    return float(np.mean((probs - labels) ** 2))


def _plot_three_methods(x, y_entropy, y_variance, y_random, xlabel, ylabel, out_path):
    plt.figure(figsize=(6, 4))
    plt.plot(x, y_entropy, "-o", label="熵")
    plt.plot(x, y_variance, "-o", label="方差")
    plt.plot(x, y_random, "-o", label="随机")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=1000)
    plt.close()


def plot_error_consistency_boxplot(data_dict, ylabel, out_path, show_mean=False):
    order = ["正确检出", "误检", "漏检", "正确负检"]
    data = [np.asarray(data_dict[k], dtype=np.float32) for k in order]

    colors = {
        "正确检出": "#4C78A8",
        "误检": "#E45756",
        "漏检": "#F58518",
        "正确负检": "#54A24B",
    }

    plt.figure(figsize=(6.2, 4.2))

    bp = plt.boxplot(
        data,
        labels=order,
        showfliers=False,
        whis=(5, 95),
        patch_artist=True,
        widths=0.55,
        medianprops=dict(color="#222222", linewidth=2.0),
        whiskerprops=dict(color="#333333", linewidth=1.2),
        capprops=dict(color="#333333", linewidth=1.2),
        boxprops=dict(color="#333333", linewidth=1.2),
    )

    for patch, label in zip(bp["boxes"], order):
        patch.set_facecolor(colors[label])
        patch.set_alpha(0.28)

    if show_mean:
        means = [float(np.mean(d)) if len(d) else float("nan") for d in data]
        xs = np.arange(1, len(order) + 1)
        plt.scatter(xs, means, s=18, color="#111111", zorder=3, label="均值")

    plt.ylabel(ylabel)
    plt.grid(True, axis="y", alpha=0.25, linewidth=0.8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=1000)
    plt.close()


# ---------------- 主流程 ----------------
def main():
    ensure_dirs()
    fig_dir = os.path.join(OUT_DIR, "fig")
    csv_dir = os.path.join(OUT_DIR, "csv")

    model = init_model(CONFIG_FILE, CHECKPOINT_FILE, device=DEVICE)

    img_files = list_images(TEST_IMG_DIR)
    if len(img_files) == 0:
        raise RuntimeError(f"No images found in {TEST_IMG_DIR}")

    err_stats_rows = []
    rng_box = np.random.default_rng(BOXPLOT_SEED)

    # 字典键改为中文
    box_entropy = {"正确检出": [], "误检": [], "漏检": [], "正确负检": []}
    box_variance = {"正确检出": [], "误检": [], "漏检": [], "正确负检": []}

    reject_fp_rows = []
    fn_capture_rows = []
    all_probs = []
    all_labels = []
    time_per_img = []
    rng_global = np.random.default_rng(12345)

    for i, fn in enumerate(img_files, 1):
        img_path = os.path.join(TEST_IMG_DIR, fn)
        gt_path = os.path.join(GT_LABEL_DIR, fn)
        if not os.path.exists(gt_path):
            base, _ = os.path.splitext(fn)
            cand = os.path.join(GT_LABEL_DIR, base + ".png")
            if os.path.exists(cand):
                gt_path = cand
            else:
                raise FileNotFoundError(f"GT not found for {fn}")

        gt = read_mask(gt_path)
        valid = apply_ignore(gt, IGNORE_INDEX)
        gt01 = (gt == 1).astype(np.uint8)

        model.eval()
        model.apply(enable_dropout)
        t0 = time.perf_counter()
        p_mean, ent, var = mc_maps(model, img_path, T=T_MC, landslide_idx=LANDSLIDE_IDX)
        time_per_img.append(time.perf_counter() - t0)
        pred = pred_mask_from_pmean(p_mean, thr=0.5)

        # 分类命名改为中文
        m_正确检出 = valid & (pred == 1) & (gt01 == 1)
        m_误检 = valid & (pred == 1) & (gt01 == 0)
        m_漏检 = valid & (pred == 0) & (gt01 == 1)
        m_正确负检 = valid & (pred == 0) & (gt01 == 0)

        # 循环名称改为中文
        for name, m in [("正确检出", m_正确检出), ("误检", m_误检), ("漏检", m_漏检), ("正确负检", m_正确负检)]:
            if m.any():
                ent_med = float(np.median(ent[m]))
                ent_q90 = float(np.quantile(ent[m], 0.9))
                var_med = float(np.median(var[m]))
                var_q90 = float(np.quantile(var[m], 0.9))
                npx = int(m.sum())

                idxs = np.flatnonzero(m.ravel())
                k = min(BOXPLOT_SAMPLE_PER_GROUP_PER_IMAGE, len(idxs))
                if k > 0:
                    pick = rng_box.choice(idxs, size=k, replace=False)
                    box_entropy[name].extend(ent.ravel()[pick].tolist())
                    box_variance[name].extend(var.ravel()[pick].tolist())
            else:
                ent_med = ent_q90 = var_med = var_q90 = float("nan")
                npx = 0
            err_stats_rows.append([fn, name, npx, ent_med, ent_q90, var_med, var_q90])

        # B1 误检拒绝
        valid_reject_fp = valid & (pred == 1)
        flat_fp_idx = np.flatnonzero(valid_reject_fp.ravel())
        n_fp = len(flat_fp_idx)
        if n_fp > 0:
            ent_flat = ent.ravel()[flat_fp_idx]
            var_flat = var.ravel()[flat_fp_idx]
            order_ent = np.argsort(-ent_flat)
            order_var = np.argsort(-var_flat)
            order_rand = rng_global.permutation(n_fp)

            for rate in REJECT_RATES:
                k = int(math.floor(rate * n_fp))
                keep_ent = np.ones(n_fp, dtype=bool)
                keep_var = np.ones(n_fp, dtype=bool)
                keep_rnd = np.ones(n_fp, dtype=bool)
                if k > 0:
                    keep_ent[order_ent[:k]] = False
                    keep_var[order_var[:k]] = False
                    keep_rnd[order_rand[:k]] = False

                def risk_under_keep(keep_vec):
                    keep_mask = np.zeros(valid.size, dtype=bool)
                    keep_mask[flat_fp_idx[keep_vec]] = True
                    keep_mask = keep_mask.reshape(valid.shape)
                    return float(pixel_error_rate(pred, gt01, keep_mask))

                reject_fp_rows.append([fn, rate, "entropy", risk_under_keep(keep_ent)])
                reject_fp_rows.append([fn, rate, "variance", risk_under_keep(keep_var)])
                reject_fp_rows.append([fn, rate, "random", risk_under_keep(keep_rnd)])

        # B2 漏检捕获
        valid_review_bg = valid & (pred == 0)
        flat_bg_idx = np.flatnonzero(valid_review_bg.ravel())
        n_bg = len(flat_bg_idx)
        flat_fn_idx = np.flatnonzero((valid_review_bg & (gt01 == 1)).ravel())
        n_fn = len(flat_fn_idx)

        if n_bg > 0 and n_fn > 0:
            ent_bg = ent.ravel()[flat_bg_idx]
            var_bg = var.ravel()[flat_bg_idx]
            order_ent_bg = np.argsort(-ent_bg)
            order_var_bg = np.argsort(-var_bg)
            order_rand_bg = rng_global.permutation(n_bg)
            fn_set = set(flat_fn_idx.tolist())

            def fn_capture(order, k):
                picked_flat = flat_bg_idx[order[:k]]
                hit = sum((int(x) in fn_set) for x in picked_flat)
                return hit / (n_fn + 1e-12)

            for rate in REJECT_RATES:
                k = int(math.floor(rate * n_bg))
                if k <= 0:
                    fn_capture_rows.append([fn, rate, "entropy", 0.0])
                    fn_capture_rows.append([fn, rate, "variance", 0.0])
                    fn_capture_rows.append([fn, rate, "random", 0.0])
                    continue
                fn_capture_rows.append([fn, rate, "entropy", float(fn_capture(order_ent_bg, k))])
                fn_capture_rows.append([fn, rate, "variance", float(fn_capture(order_var_bg, k))])
                fn_capture_rows.append([fn, rate, "random", float(fn_capture(order_rand_bg, k))])

        all_probs.append(p_mean[valid].astype(np.float32))
        all_labels.append(gt01[valid].astype(np.uint8))
        print(f"[{i}/{len(img_files)}] 完成: {fn}")

    # 保存文件
    err_csv = os.path.join(csv_dir, "error_consistency_stats.csv")
    with open(err_csv, "w", encoding="utf-8") as f:
        f.write("image,group,n_pixels,ent_median,ent_q90,var_median,var_q90\n")
        for r in err_stats_rows:
            f.write(",".join(map(str, r)) + "\n")

    # 箱线图
    plot_error_consistency_boxplot(
        box_entropy, ylabel="熵",
        out_path=os.path.join(fig_dir, "error_consistency_boxplot_entropy.png")
    )
    plot_error_consistency_boxplot(
        box_variance, ylabel="方差",
        out_path=os.path.join(fig_dir, "error_consistency_boxplot_variance.png")
    )

    # CSV保存
    rej_fp_csv = os.path.join(csv_dir, "reject_fp_risk_per_image.csv")
    with open(rej_fp_csv, "w", encoding="utf-8") as f:
        f.write("image,reject_rate,method,risk\n")
        for r in reject_fp_rows:
            f.write(",".join(map(str, r)) + "\n")

    fn_cap_csv = os.path.join(csv_dir, "fn_capture_per_image.csv")
    with open(fn_cap_csv, "w", encoding="utf-8") as f:
        f.write("image,review_rate,method,fn_capture\n")
        for r in fn_capture_rows:
            f.write(",".join(map(str, r)) + "\n")

    # 绘图
    rates = REJECT_RATES

    def aggregate(rows, metric_col_idx):
        out = {"entropy": [], "variance": [], "random": []}
        for rate in rates:
            for m in out.keys():
                vals = [row[metric_col_idx] for row in rows if (row[1] == rate and row[2] == m)]
                out[m].append(float(np.mean(vals)) if vals else np.nan)
        return out

    agg_fp = aggregate(reject_fp_rows, 3)
    _plot_three_methods(
        rates, agg_fp["entropy"], agg_fp["variance"], agg_fp["random"],
        xlabel="预测为滑坡像素的拒绝率", ylabel="风险值",
        out_path=os.path.join(fig_dir, "reject_fp_risk.png")
    )

    agg_fn = aggregate(fn_capture_rows, 3)
    _plot_three_methods(
        rates, agg_fn["entropy"], agg_fn["variance"], agg_fn["random"],
        xlabel="预测为背景像素的复核率", ylabel="漏检捕获率",
        out_path=os.path.join(fig_dir, "fn_capture.png")
    )

    # 校准图
    probs = np.concatenate(all_probs, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    rel_png = os.path.join(fig_dir, "reliability_diagram.png")
    save_reliability_diagram(probs, labels, BINS, rel_png)

    ece = ece_score(probs, labels, BINS)
    brier = brier_score(probs, labels)
    cal_csv = os.path.join(csv_dir, "calibration_metrics.csv")
    with open(cal_csv, "w", encoding="utf-8") as f:
        f.write("bins,ece,brier\n")
        f.write(f"{BINS},{ece:.8f},{brier:.8f}\n")

    # 时间
    t = np.array(time_per_img, dtype=np.float64)
    time_csv = os.path.join(csv_dir, "mc_time.csv")
    with open(time_csv, "w", encoding="utf-8") as f:
        f.write("T,mean_sec,std_sec,n\n")
        f.write(f"{T_MC},{t.mean():.6f},{t.std(ddof=1) if len(t) > 1 else 0.0:.6f},{len(t)}\n")

    print("\n[运行完成]")
    print("输出目录:", OUT_DIR)


if __name__ == "__main__":
    main()