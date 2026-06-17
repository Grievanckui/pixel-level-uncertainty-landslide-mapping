import os
import os.path as osp
import glob
from typing import List

import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

import torch
import torch.nn.functional as F
from mmseg.apis import init_model, inference_model


# =========================
# 1) 选图（rank文件名 -> stem -> test原图）
# =========================
IN_DIR = r"E:\MMseg\BIYELUNWEN\data\vis_compare_top50_pgf_better"
TEST_IMG_DIR = r"E:\MMseg\BIYELUNWEN\data\test\vis_images"

RANKS = [3, 6, 35, 59, 22, 64]
COL_LABELS = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"]

OUT_FILE = osp.join(IN_DIR, "uncertainty_transposed_3_6_35_59_22_64_with_colorbars_tight.png")

# =========================
# 2) 模型
# =========================
CONFIG_FILE = r"E:\MMseg\mmsegmentation\configs\segformer\segformer_mit-b1_landslide2.py"
CHECKPOINT_FILE = r"E:\MMseg\BIYELUNWEN\work_dirs\segformer_landslide\20260121_000000\segformer_landslide\best_mIoU_iter_93000.pth"

T = 20
POS_CLASS_ID = 1


SCALE = 3.0
GAP_PX = 14
GAP_COLOR_BGR = (192, 192, 192)

TOP_HEADER_H = 60
LEFT_LABEL_W = 120
RIGHT_CBAR_W = 95
CBAR_BAR_W = 12
CBAR_LEFT_PAD = 6

HEADER_BG_BGR = (255, 255, 255)

ROW_TITLES = [
    "image",
    "prediction",
    "p_mean",
    "entropy",
    "variance",
]

FONT_SIZE_PX = 32
FONT_CN_PATH = r"C:\Windows\Fonts\simsun.ttc"
FONT_EN_PATH = r"C:\Windows\Fonts\times.ttf"
TEXT_COLOR_RGB = (0, 0, 0)

CMAP_MEAN = cv2.COLORMAP_TURBO
CMAP_ENT = cv2.COLORMAP_MAGMA
CMAP_VAR = cv2.COLORMAP_VIRIDIS

MEAN_VMIN, MEAN_VMAX = 0.0, 1.0
ENT_VMIN, ENT_VMAX = 0.0, float(np.log(2.0))

VAR_VMIN = 0.0
VAR_Q = 0.99


# ---------- 选图 ----------
def find_file_for_rank(rank: int) -> str:
    patterns = [
        osp.join(IN_DIR, f"{rank}_*.*"),
        osp.join(IN_DIR, f"{rank:02d}_*.*"),
        osp.join(IN_DIR, f"*_第{rank}.*"),
    ]
    hits: List[str] = []
    for p in patterns:
        hits.extend(glob.glob(p))
    hits = sorted(hits)
    if not hits:
        raise FileNotFoundError(f"Cannot find file for rank={rank} in {IN_DIR}")
    return hits[0]


def parse_stem_from_rank_file(rank_file: str) -> str:
    base = osp.basename(rank_file)
    stem, _ = osp.splitext(base)
    parts = stem.split("_", 1)
    if len(parts) == 2 and parts[0].isdigit():
        return parts[1]
    raise ValueError(f"Unexpected rank file name: {base}")


def find_image_by_stem(img_dir: str, stem: str) -> str:
    exts = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")
    candidates = [osp.join(img_dir, stem + e) for e in exts if osp.exists(osp.join(img_dir, stem + e))]
    if not candidates:
        hits = []
        for e in exts:
            hits.extend(glob.glob(osp.join(img_dir, stem + e)))
        hits = sorted(hits)
        if not hits:
            raise FileNotFoundError(f"Cannot find image with stem={stem} in {img_dir}")
        candidates = hits
    candidates.sort(key=lambda x: (0 if x.lower().endswith(".png") else 1, x))
    return candidates[0]


# ---------- 工具 ----------
def normalize_to_uint8(x: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    x = np.clip(x, vmin, vmax)
    if vmax <= vmin:
        vmax = vmin + 1e-12
    x = (x - vmin) / (vmax - vmin)
    return (x * 255.0).astype(np.uint8)


def apply_colormap(gray_u8: np.ndarray, cmap: int) -> np.ndarray:
    return cv2.applyColorMap(gray_u8, cmap)


def enable_dropout_only(model):
    model.eval()
    for m in model.modules():
        if "dropout" in m.__class__.__name__.lower():
            m.train()
    return model


@torch.no_grad()
def inference_prob_map(model, img_path: str, pos_class_id: int) -> np.ndarray:
    result = inference_model(model, img_path)
    seg_logits = None
    if hasattr(result, "seg_logits") and result.seg_logits is not None:
        seg_logits = result.seg_logits.data
    elif hasattr(result, "logits") and result.logits is not None:
        seg_logits = result.logits.data

    if seg_logits is None:
        raise RuntimeError("inference_model() did not provide seg_logits/logits; cannot compute probabilities.")

    if seg_logits.dim() == 4:
        seg_logits = seg_logits[0]  # (C,H,W)

    prob = F.softmax(seg_logits, dim=0)[pos_class_id]
    return prob.cpu().numpy().astype(np.float32)


@torch.no_grad()
def mc_dropout_maps(model, img_path: str, T: int, pos_class_id: int):
    enable_dropout_only(model)
    probs = [torch.from_numpy(inference_prob_map(model, img_path, pos_class_id)) for _ in range(T)]
    probs_t = torch.stack(probs, dim=0)  # (T,H,W)

    p_mean = probs_t.mean(dim=0)
    variance = probs_t.var(dim=0, unbiased=False)

    eps = 1e-8
    p = torch.clamp(p_mean, eps, 1.0 - eps)
    entropy = -(p * torch.log(p) + (1 - p) * torch.log(1 - p))

    pred01 = (p_mean >= 0.5).to(torch.uint8)

    return (pred01.cpu().numpy(),
            p_mean.cpu().numpy().astype(np.float32),
            entropy.cpu().numpy().astype(np.float32),
            variance.cpu().numpy().astype(np.float32))


def make_vertical_colorbar(h: int, cmap: int, width: int) -> np.ndarray:
    grad = np.linspace(255, 0, h, dtype=np.uint8).reshape(h, 1)
    grad = np.repeat(grad, width, axis=1)
    return cv2.applyColorMap(grad, cmap)


def pack_cbar(bar_bgr: np.ndarray, right_cbar_w: int, left_pad: int) -> np.ndarray:
    out = np.full((bar_bgr.shape[0], right_cbar_w, 3), 255, dtype=np.uint8)
    x0 = left_pad
    out[:, x0:x0 + bar_bgr.shape[1]] = bar_bgr
    return out


# ---------- PIL 标注 ----------
def bgr_to_pil(img_bgr: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))


def pil_to_bgr(img_pil: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


def draw_centered_multiline_text(draw: ImageDraw.ImageDraw, box, text: str,
                                 font: ImageFont.FreeTypeFont, fill_rgb=(0, 0, 0),
                                 line_spacing_px: int = 4):
    x0, y0, x1, y1 = box
    W = x1 - x0
    H = y1 - y0

    lines = text.split("\n")
    line_sizes = []
    total_h = 0
    for ln in lines:
        bbox = draw.textbbox((0, 0), ln, font=font)
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        line_sizes.append((w, h))
        total_h += h
    total_h += line_spacing_px * (len(lines) - 1) if len(lines) > 1 else 0

    yy = y0 + (H - total_h) // 2
    for (ln, (tw, th)) in zip(lines, line_sizes):
        xx = x0 + (W - tw) // 2
        draw.text((xx, yy), ln, font=font, fill=fill_rgb)
        yy += th + line_spacing_px


def draw_vertical_label(base_img: Image.Image, box, text: str,
                        font: ImageFont.FreeTypeFont, fill_rgb=(0, 0, 0),
                        line_spacing_px: int = 4, rotate_deg: float = 90):
    x0, y0, x1, y1 = box
    bw = x1 - x0
    bh = y1 - y0

    dummy = Image.new("RGB", (10, 10), (255, 255, 255))
    ddraw = ImageDraw.Draw(dummy)

    lines = text.split("\n")
    line_bboxes = [ddraw.textbbox((0, 0), ln, font=font) for ln in lines]
    line_ws = [bb[2] - bb[0] for bb in line_bboxes] if line_bboxes else [1]
    line_hs = [bb[3] - bb[1] for bb in line_bboxes] if line_bboxes else [1]

    text_w = max(line_ws)
    text_h = sum(line_hs) + (line_spacing_px * (len(lines) - 1) if len(lines) > 1 else 0)
    pad = 15  # 原8 -> 4（更紧凑）

    label_img = Image.new("RGB", (text_w + 2 * pad, text_h + 2 * pad), (255, 255, 255))
    ldraw = ImageDraw.Draw(label_img)

    yy = pad
    for ln, h in zip(lines, line_hs):
        bb = ldraw.textbbox((0, 0), ln, font=font)
        lw = bb[2] - bb[0]
        ldraw.text((pad + (text_w - lw) // 2, yy), ln, font=font, fill=fill_rgb)
        yy += h + line_spacing_px

    label_rot = label_img.rotate(rotate_deg, expand=True, fillcolor=(255, 255, 255))
    rw, rh = label_rot.size

    px = x0 + (bw - rw) // 2
    py = y0 + (bh - rh) // 2

    base_img.paste(label_rot, (px, py))


def main():
    os.makedirs(osp.dirname(OUT_FILE), exist_ok=True)

    print("Loading model...")
    model = init_model(CONFIG_FILE, CHECKPOINT_FILE, device="cuda:0")

    rank_files = [find_file_for_rank(r) for r in RANKS]
    stems = [parse_stem_from_rank_file(p) for p in rank_files]
    img_paths = [find_image_by_stem(TEST_IMG_DIR, s) for s in stems]

    print("Selected (rank -> stem -> image):")
    for r, s, p in zip(RANKS, stems, img_paths):
        print(f"  rank={r:<3} stem={s:<20} img={osp.basename(p)}")

    panels_rgb, panels_pred, panels_mean, panels_ent, panels_var = [], [], [], [], []
    for img_path in img_paths:
        img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise FileNotFoundError(img_path)

        pred01, p_mean, ent, var = mc_dropout_maps(model, img_path, T=T, pos_class_id=POS_CLASS_ID)

        pred_vis = np.zeros_like(img_bgr)
        pred_vis[pred01.astype(bool)] = (255, 255, 255)

        mean_hm = apply_colormap(normalize_to_uint8(p_mean, MEAN_VMIN, MEAN_VMAX), CMAP_MEAN)
        ent_hm = apply_colormap(normalize_to_uint8(ent, ENT_VMIN, ENT_VMAX), CMAP_ENT)

        v_upper = float(np.quantile(var, VAR_Q))
        if not np.isfinite(v_upper) or v_upper <= 1e-12:
            v_upper = float(var.max()) if np.isfinite(var.max()) and var.max() > 0 else 1e-6
        var_hm = apply_colormap(normalize_to_uint8(var, VAR_VMIN, v_upper), CMAP_VAR)

        panels_rgb.append(img_bgr)
        panels_pred.append(pred_vis)
        panels_mean.append(mean_hm)
        panels_ent.append(ent_hm)
        panels_var.append(var_hm)

    H, W = panels_rgb[0].shape[:2]

    def resize_list(imgs):
        out = []
        for im in imgs:
            if im.shape[:2] != (H, W):
                im = cv2.resize(im, (W, H), interpolation=cv2.INTER_CUBIC)
            out.append(im)
        return out

    panels_rgb = resize_list(panels_rgb)
    panels_pred = resize_list(panels_pred)
    panels_mean = resize_list(panels_mean)
    panels_ent = resize_list(panels_ent)
    panels_var = resize_list(panels_var)

    gap_v = np.full((H, GAP_PX, 3), GAP_COLOR_BGR, dtype=np.uint8)

    def hconcat6(imgs6: List[np.ndarray]) -> np.ndarray:
        row = imgs6[0]
        for im in imgs6[1:]:
            row = np.hstack([row, gap_v, im])
        return row

    row_rgb = hconcat6(panels_rgb)
    row_pred = hconcat6(panels_pred)
    row_mean = hconcat6(panels_mean)
    row_ent = hconcat6(panels_ent)
    row_var = hconcat6(panels_var)

    right_white = np.full((H, RIGHT_CBAR_W, 3), 255, dtype=np.uint8)
    cbar_mean = pack_cbar(make_vertical_colorbar(H, CMAP_MEAN, CBAR_BAR_W), RIGHT_CBAR_W, CBAR_LEFT_PAD)
    cbar_ent = pack_cbar(make_vertical_colorbar(H, CMAP_ENT, CBAR_BAR_W), RIGHT_CBAR_W, CBAR_LEFT_PAD)
    cbar_var = pack_cbar(make_vertical_colorbar(H, CMAP_VAR, CBAR_BAR_W), RIGHT_CBAR_W, CBAR_LEFT_PAD)

    row_rgb = np.hstack([row_rgb, right_white])
    row_pred = np.hstack([row_pred, right_white])
    row_mean = np.hstack([row_mean, cbar_mean])
    row_ent = np.hstack([row_ent, cbar_ent])
    row_var = np.hstack([row_var, cbar_var])


    content_w_6cols = row_rgb.shape[1] - RIGHT_CBAR_W
    gap_h_left = np.full((GAP_PX, content_w_6cols, 3), GAP_COLOR_BGR, dtype=np.uint8)
    gap_h_right = np.full((GAP_PX, RIGHT_CBAR_W, 3), 255, dtype=np.uint8)
    gap_h = np.hstack([gap_h_left, gap_h_right])

    content = row_rgb
    for r in [row_pred, row_mean, row_ent, row_var]:
        content = np.vstack([content, gap_h, r])

    content_h, content_w = content.shape[:2]

    total_w = LEFT_LABEL_W + content_w
    total_h = TOP_HEADER_H + content_h

    canvas = np.full((total_h, total_w, 3), HEADER_BG_BGR, dtype=np.uint8)
    canvas[TOP_HEADER_H:TOP_HEADER_H + content_h, LEFT_LABEL_W:LEFT_LABEL_W + content_w] = content

    canvas_pil = bgr_to_pil(canvas)
    draw = ImageDraw.Draw(canvas_pil)

    font_cn = ImageFont.truetype(FONT_CN_PATH, FONT_SIZE_PX)
    font_en = ImageFont.truetype(FONT_EN_PATH, FONT_SIZE_PX)


    for j, lab in enumerate(COL_LABELS):
        x0 = LEFT_LABEL_W + j * W + j * GAP_PX
        x1 = x0 + W
        draw_centered_multiline_text(draw, (x0, 0, x1, TOP_HEADER_H), lab, font_en, fill_rgb=TEXT_COLOR_RGB)


    for i, title in enumerate(ROW_TITLES):
        yy0 = TOP_HEADER_H + i * H + i * GAP_PX
        yy1 = yy0 + H
        draw_vertical_label(canvas_pil, (0, yy0, LEFT_LABEL_W, yy1), title, font_en, fill_rgb=TEXT_COLOR_RGB)


    x_cbar0 = LEFT_LABEL_W + (6 * W + 5 * GAP_PX)
    y_mean0 = TOP_HEADER_H + 2 * H + 2 * GAP_PX
    y_ent0 = TOP_HEADER_H + 3 * H + 3 * GAP_PX
    y_var0 = TOP_HEADER_H + 4 * H + 4 * GAP_PX

    text_x = x_cbar0 + CBAR_LEFT_PAD + CBAR_BAR_W + 6

    draw.text((text_x, y_mean0 + 4), "1.0", font=font_en, fill=TEXT_COLOR_RGB)
    draw.text((text_x, y_mean0 + H - 34), "0.0", font=font_en, fill=TEXT_COLOR_RGB)

    draw.text((text_x, y_ent0 + 4), "ln2", font=font_en, fill=TEXT_COLOR_RGB)
    draw.text((text_x, y_ent0 + H - 34), "0", font=font_en, fill=TEXT_COLOR_RGB)

    draw.text((text_x, y_var0 + 4), "Q99", font=font_en, fill=TEXT_COLOR_RGB)
    draw.text((text_x, y_var0 + H - 34), "0", font=font_en, fill=TEXT_COLOR_RGB)

    canvas_pil.save(OUT_FILE, dpi=(1000, 1000))
    print("Saved:", OUT_FILE)

    # out_bgr = pil_to_bgr(canvas_pil)
    # cv2.imwrite(OUT_FILE, out_bgr)
    # print("Saved:", OUT_FILE)


if __name__ == "__main__":
    main()