"""
更清楚版（不加 colorbar）：
- mean: turbo, vmin=0, vmax=1
- entropy: magma, vmin=0, vmax=ln2（避免一片亮）
- variance: viridis, vmin=0, vmax=quantile(var, 0.99)（避免被极端值压扁）

  MC 只用于不确定性估计。
"""

import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2

from mmseg.apis import init_model, inference_model


CONFIG_FILE = r"E:\MMseg\mmsegmentation\configs\segformer\segformer_mit-b1_landslide2.py"
CHECKPOINT_FILE = r"E:\MMseg\BIYELUNWEN\work_dirs\segformer_landslide\20260121_000000\segformer_landslide\best_mIoU_iter_93000.pth"

TEST_IMG_DIR = r"E:\MMseg\BIYELUNWEN\data\test\vis_images"
GT_LABEL_DIR = r"E:\MMseg\BIYELUNWEN\data\test\labels"

OUT_DIR = r"E:\MMseg\BIYELUNWEN\work_dirs\segformer_landslide\mc_results2_final"

PNG_DIR = os.path.join(OUT_DIR, "png")
NPY_DIR = os.path.join(OUT_DIR, "npy")

MEAN_PNG_DIR = os.path.join(PNG_DIR, "mean")
VAR_PNG_DIR = os.path.join(PNG_DIR, "variance")
ENT_PNG_DIR = os.path.join(PNG_DIR, "entropy")
GATES_PNG_DIR = os.path.join(PNG_DIR, "gates_mean")

MEAN_NPY_DIR = os.path.join(NPY_DIR, "mean")
VAR_NPY_DIR = os.path.join(NPY_DIR, "variance")
ENT_NPY_DIR = os.path.join(NPY_DIR, "entropy")
PRED_NPY_DIR = os.path.join(NPY_DIR, "pred_det")
GT_NPY_DIR = os.path.join(NPY_DIR, "gt")
GATES_NPY_DIR = os.path.join(NPY_DIR, "gates_mean")


def ensure_dirs():
    for d in [
        MEAN_PNG_DIR, VAR_PNG_DIR, ENT_PNG_DIR, GATES_PNG_DIR,
        MEAN_NPY_DIR, VAR_NPY_DIR, ENT_NPY_DIR, PRED_NPY_DIR, GT_NPY_DIR, GATES_NPY_DIR
    ]:
        os.makedirs(d, exist_ok=True)


def enable_dropout(m):
    if isinstance(m, (torch.nn.Dropout, torch.nn.Dropout2d, torch.nn.Dropout3d)):
        m.train()


def save_heatmap_no_colorbar(heatmap, save_path, cmap, vmin=None, vmax=None):
    plt.figure(figsize=(6, 6))
    plt.axis("off")
    plt.imshow(heatmap, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
    plt.close()


def list_images(img_dir: str):
    exts = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")
    return sorted([f for f in os.listdir(img_dir) if f.lower().endswith(exts)])


def find_gt_path(img_name: str) -> str:
    stem = os.path.splitext(img_name)[0]
    for ext in (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"):
        p = os.path.join(GT_LABEL_DIR, stem + ext)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"Cannot find GT label for image {img_name} in {GT_LABEL_DIR}")


def read_label_as_binary(label_path: str) -> np.ndarray:
    lab = cv2.imread(label_path, cv2.IMREAD_UNCHANGED)
    if lab is None:
        raise FileNotFoundError(label_path)
    if lab.ndim == 3:
        lab = cv2.cvtColor(lab, cv2.COLOR_BGR2GRAY)
    return (lab > 0).astype(np.uint8)


def deterministic_pred_from_mmseg(model, img_path: str) -> np.ndarray:
    result = inference_model(model, img_path)
    pred = result.pred_sem_seg.data.squeeze().detach().cpu().numpy().astype(np.int32)
    return (pred == 1).astype(np.uint8)


def mc_uncertainty(model, img_path: str, num_samples: int, landslide_idx: int, debug: bool = False):
    probs = []
    gates_list = []

    for i in range(num_samples):
        result = inference_model(model, img_path)

        logits = result.seg_logits.data
        if logits.ndim == 4:
            logits = logits[0]
        if logits.ndim != 3:
            raise ValueError(f"Unexpected seg_logits ndim={logits.ndim}, shape={tuple(logits.shape)}")

        gates = getattr(model.decode_head, "latest_gates", None)
        if gates is not None:
            if gates.ndim == 4:
                gates = gates[0]
            gates_list.append(gates.detach().cpu().numpy().astype(np.float32))

        if debug and i == 0:
            print("[DEBUG] seg_logits shape:", tuple(result.seg_logits.data.shape))
            if gates is None:
                print("[DEBUG] gates: None")
            else:
                print("[DEBUG] gates shape:", tuple(gates.shape))

        C, H, W = logits.shape
        if C == 1:
            prob = torch.sigmoid(logits).cpu().numpy().astype(np.float32)
            use_idx = 0
        else:
            prob = torch.softmax(logits, dim=0).cpu().numpy().astype(np.float32)
            use_idx = landslide_idx
            if not (0 <= use_idx < C):
                raise ValueError(f"Invalid landslide_idx={use_idx} for C={C}")

        probs.append(prob)

    probs = np.stack(probs, axis=0)
    mean_pred = probs.mean(axis=0)
    var_pred = probs.var(axis=0)

    p_mean = mean_pred[use_idx].astype(np.float32)
    var_map = var_pred[use_idx].astype(np.float32)

    eps = 1e-8
    entropy_map = (-(p_mean * np.log(p_mean + eps) + (1 - p_mean) * np.log(1 - p_mean + eps))).astype(np.float32)

    gates_mean = None
    if len(gates_list) > 0:
        gates_mean = np.stack(gates_list, axis=0).mean(axis=0).astype(np.float32)

    return p_mean, var_map, entropy_map, gates_mean


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--landslide-idx", type=int, default=1)
    parser.add_argument("--var-q", type=float, default=0.99, help="variance 显示上限分位数（默认0.99）")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    ensure_dirs()

    ENT_VMIN, ENT_VMAX = 0.0, float(np.log(2.0))

    model = init_model(CONFIG_FILE, CHECKPOINT_FILE, device="cuda:0")

    img_files = list_images(TEST_IMG_DIR)
    print(f"共发现 {len(img_files)} 张测试图像")

    for idx, img_name in enumerate(img_files, 1):
        img_path = os.path.join(TEST_IMG_DIR, img_name)
        base_name = os.path.splitext(img_name)[0]

        # GT
        gt_path = find_gt_path(img_name)
        gt01 = read_label_as_binary(gt_path)
        np.save(os.path.join(GT_NPY_DIR, f"{base_name}_gt01.npy"), gt01)

        # pred（与第三章一致）
        model.eval()
        pred_det01 = deterministic_pred_from_mmseg(model, img_path)
        np.save(os.path.join(PRED_NPY_DIR, f"{base_name}_pred_det01.npy"), pred_det01)

        # MC 不确定性
        model.eval()
        model.apply(enable_dropout)
        p_mean, var_map, entropy_map, gates_mean = mc_uncertainty(
            model, img_path, num_samples=args.samples, landslide_idx=args.landslide_idx, debug=args.debug
        )

        # NPY
        np.save(os.path.join(MEAN_NPY_DIR, f"{base_name}_p_mean.npy"), p_mean)
        np.save(os.path.join(VAR_NPY_DIR, f"{base_name}_var.npy"), var_map)
        np.save(os.path.join(ENT_NPY_DIR, f"{base_name}_entropy.npy"), entropy_map)

        # PNG：更清楚的可视化规范
        save_heatmap_no_colorbar(p_mean, os.path.join(MEAN_PNG_DIR, f"{base_name}_mean.png"),
                                 cmap="turbo", vmin=0.0, vmax=1.0)

        var_vmax = float(np.quantile(var_map, args.var_q))
        if not np.isfinite(var_vmax) or var_vmax <= 0:
            var_vmax = float(var_map.max()) if float(var_map.max()) > 0 else 1e-6
        save_heatmap_no_colorbar(var_map, os.path.join(VAR_PNG_DIR, f"{base_name}_var.png"),
                                 cmap="viridis", vmin=0.0, vmax=var_vmax)

        save_heatmap_no_colorbar(entropy_map, os.path.join(ENT_PNG_DIR, f"{base_name}_entropy.png"),
                                 cmap="magma", vmin=ENT_VMIN, vmax=ENT_VMAX)

        # gates（若有）
        if gates_mean is not None:
            np.save(os.path.join(GATES_NPY_DIR, f"{base_name}_gates_mean.npy"), gates_mean)
            for s in range(gates_mean.shape[0]):
                save_heatmap_no_colorbar(gates_mean[s], os.path.join(GATES_PNG_DIR, f"{base_name}_gate{s + 1}.png"),
                                         cmap="viridis", vmin=0.0, vmax=1.0)

        print(f"[{idx}/{len(img_files)}] 已处理: {img_name}")

    print("\nDone.")
    print("PNG:", PNG_DIR)
    print("NPY:", NPY_DIR)


if __name__ == "__main__":
    main()