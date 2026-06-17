import os
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy import interpolate


plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False
# ====================================================

from mmseg.apis import init_model, inference_model

# ---------------- 固定路径 ----------------
CONFIG_FILE = r"E:\MMseg\mmsegmentation\configs\segformer\segformer_mit-b1_landslide2.py"
CHECKPOINT_FILE = r"E:\MMseg\BIYELUNWEN\work_dirs\segformer_landslide\20260121_000000\segformer_landslide\best_mIoU_iter_93000.pth"

TEST_IMG_DIR = r"E:\MMseg\BIYELUNWEN\data\test\vis_images"
GT_LABEL_DIR = r"E:\MMseg\BIYELUNWEN\data\test\labels"

# 输出目录
OUT_DIR_T = r"E:\MMseg\BIYELUNWEN\work_dirs\segformer_landslide\T"

# ---------------- 实验设置 ----------------
TS = [5, 10, 20, 40]
T_REF = 40
LANDSLIDE_IDX = 1
DEVICE = "cuda:0"

VIS_ONE = ""


# ---------------- 工具函数 ----------------
def ensure_dirs():
    os.makedirs(OUT_DIR_T, exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR_T, "fig"), exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR_T, "qualitative"), exist_ok=True)


def enable_dropout(m):
    if isinstance(m, (torch.nn.Dropout, torch.nn.Dropout2d, torch.nn.Dropout3d)):
        m.train()


def list_images(img_dir: str):
    exts = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")
    return sorted([f for f in os.listdir(img_dir) if f.lower().endswith(exts)])


def pearsonr_flat(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    da = a - a.mean()
    db = b - b.mean()
    denom = (np.sqrt((da * da).sum()) * np.sqrt((db * db).sum())) + eps
    return float((da * db).sum() / denom)


def mc_entropy_variance(model, img_path: str, T: int, landslide_idx: int = 1, eps: float = 1e-8):
    probs = []
    for _ in range(T):
        result = inference_model(model, img_path)
        logits = result.seg_logits.data
        if logits.ndim == 4:
            logits = logits[0]
        if logits.ndim != 3:
            raise ValueError(f"Unexpected seg_logits ndim={logits.ndim}, shape={tuple(logits.shape)}")

        C, H, W = logits.shape
        if C == 1:
            prob = torch.sigmoid(logits).detach().cpu().numpy().astype(np.float32)
            use_idx = 0
        else:
            prob = torch.softmax(logits, dim=0).detach().cpu().numpy().astype(np.float32)
            use_idx = landslide_idx
            if not (0 <= use_idx < C):
                raise ValueError(f"Invalid landslide_idx={use_idx} for C={C}")
        probs.append(prob[use_idx])

    probs = np.stack(probs, axis=0)
    p_mean = probs.mean(axis=0).astype(np.float32)
    var_map = probs.var(axis=0).astype(np.float32)
    entropy_map = (-(p_mean * np.log(p_mean + eps) + (1 - p_mean) * np.log(1 - p_mean + eps))).astype(np.float32)
    return entropy_map, var_map


# 绘制平滑曲线（无误差棒）
def save_smooth_curve(xs, ys, xlabel, ylabel, out_path):
    plt.figure(figsize=(6, 4))
    # 生成平滑插值点
    xs_smooth = np.linspace(min(xs), max(xs), 300)
    f = interpolate.make_interp_spline(xs, ys, k=3)
    ys_smooth = f(xs_smooth)

    # 画平滑曲线 + 数据点
    plt.plot(xs_smooth, ys_smooth, '-', color='#1f77b4', linewidth=2)
    plt.scatter(xs, ys, color='#1f77b4', s=40, zorder=5)

    plt.xlim(0, 40)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=600)
    plt.close()


def save_time_curve_no_title(xs, ys, xlabel, ylabel, out_path):
    plt.figure(figsize=(6, 4))
    # 时间曲线平滑
    xs_smooth = np.linspace(min(xs), max(xs), 300)
    f = interpolate.make_interp_spline(xs, ys, k=3)
    ys_smooth = f(xs_smooth)
    plt.plot(xs_smooth, ys_smooth, '-', color='#ff7f0e', linewidth=2)
    plt.scatter(xs, ys, color='#ff7f0e', s=40, zorder=5)

    plt.xlim(0, 40)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def save_heatmap_no_title(img2d, out_path, cmap="magma", vmin=None, vmax=None):
    plt.figure(figsize=(4, 4))
    plt.axis("off")
    plt.imshow(img2d, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.tight_layout(pad=0)
    plt.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0)
    plt.close()


def var_vmax_q99(v):
    vmax = float(np.quantile(v, 0.99))
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = float(v.max()) if float(v.max()) > 0 else 1e-6
    return vmax


def main():
    ensure_dirs()
    fig_dir = os.path.join(OUT_DIR_T, "fig")
    qual_dir_root = os.path.join(OUT_DIR_T, "qualitative")

    if T_REF not in TS:
        raise ValueError(f"T_REF={T_REF} must be included in TS={TS}")

    model = init_model(CONFIG_FILE, CHECKPOINT_FILE, device=DEVICE)
    img_files = list_images(TEST_IMG_DIR)
    if len(img_files) == 0:
        raise RuntimeError("No images found in TEST_IMG_DIR")

    RH = {T: [] for T in TS if T != T_REF}
    RV = {T: [] for T in TS if T != T_REF}
    time_cost = {T: [] for T in TS}
    vis_target = VIS_ONE.strip().lower() if VIS_ONE else ""
    ENT_VMIN, ENT_VMAX = 0.0, float(np.log(2.0))

    for i, fn in enumerate(img_files, 1):
        img_path = os.path.join(TEST_IMG_DIR, fn)
        model.eval()
        model.apply(enable_dropout)

        # 参考值 T=40
        t0 = time.perf_counter()
        H_ref, V_ref = mc_entropy_variance(model, img_path, T=T_REF, landslide_idx=LANDSLIDE_IDX)
        time_cost[T_REF].append(time.perf_counter() - t0)

        # 其他 T
        for T in TS:
            if T == T_REF:
                continue
            t1 = time.perf_counter()
            H_T, V_T = mc_entropy_variance(model, img_path, T=T, landslide_idx=LANDSLIDE_IDX)
            time_cost[T].append(time.perf_counter() - t1)
            RH[T].append(pearsonr_flat(H_T, H_ref))
            RV[T].append(pearsonr_flat(V_T, H_ref))

        # 定性可视化
        if vis_target and fn.lower() == vis_target:
            base = os.path.splitext(fn)[0]
            qdir = os.path.join(qual_dir_root, base)
            os.makedirs(qdir, exist_ok=True)
            save_heatmap_no_title(H_ref, os.path.join(qdir, f"entropy_T{T_REF}.png"), cmap="magma", vmin=ENT_VMIN,
                                  vmax=ENT_VMAX)
            save_heatmap_no_title(V_ref, os.path.join(qdir, f"variance_T{T_REF}.png"), cmap="viridis", vmin=0.0,
                                  vmax=var_vmax_q99(V_ref))
            for T in TS:
                if T == T_REF:
                    continue
                H_T, V_T = mc_entropy_variance(model, img_path, T=T, landslide_idx=LANDSLIDE_IDX)
                save_heatmap_no_title(H_T, os.path.join(qdir, f"entropy_T{T}.png"), cmap="magma", vmin=ENT_VMIN,
                                      vmax=ENT_VMAX)
                save_heatmap_no_title(V_T, os.path.join(qdir, f"variance_T{T}.png"), cmap="viridis", vmin=0.0,
                                      vmax=var_vmax_q99(V_T))
            print("[INFO] qualitative exported:", qdir)
        print(f"[{i}/{len(img_files)}] done: {fn}")

    # 保存CSV
    csv_path = os.path.join(OUT_DIR_T, "T_sensitivity_summary.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("T,metric,mean,std,n\n")
        for T in TS:
            tm = np.array(time_cost[T], dtype=np.float64)
            f.write(f"{T},time_sec,{tm.mean():.6f},{tm.std(ddof=1) if len(tm) > 1 else 0.0:.6f},{len(tm)}\n")
        for T in TS:
            if T == T_REF:
                continue
            rh = np.array(RH[T], dtype=np.float64)
            rv = np.array(RV[T], dtype=np.float64)
            f.write(f"{T},R_H,{rh.mean():.6f},{rh.std(ddof=1) if len(rh) > 1 else 0.0:.6f},{len(rh)}\n")
            f.write(f"{T},R_V,{rv.mean():.6f},{rv.std(ddof=1) if len(rv) > 1 else 0.0:.6f},{len(rv)}\n")

    # ===================== 平滑曲线绘图（无误差棒） =====================
    xs = TS
    rh_mean = [float(np.mean(RH[T])) if T != T_REF else 1.0 for T in xs]
    rv_mean = [float(np.mean(RV[T])) if T != T_REF else 1.0 for T in xs]

    # 绘制无误差棒的平滑曲线
    save_smooth_curve(xs, rh_mean, xlabel="T", ylabel="R_H(T)", out_path=os.path.join(fig_dir, "RH_curve.png"))
    save_smooth_curve(xs, rv_mean, xlabel="T", ylabel="R_V(T)", out_path=os.path.join(fig_dir, "RV_curve.png"))

    t_mean = [float(np.mean(time_cost[T])) for T in TS]
    save_time_curve_no_title(TS, t_mean, xlabel="T", ylabel="每张影像平均推理耗时（秒）",
                             out_path=os.path.join(fig_dir, "time_curve.png"))

    print("\n[DONE]")
    print("输出目录:", OUT_DIR_T)
    print("CSV文件:", csv_path)
    print("图表目录:", fig_dir)


if __name__ == "__main__":
    main()