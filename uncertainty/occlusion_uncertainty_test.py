import os
import os.path as osp
import glob
import tempfile
from typing import List, Tuple

import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

import torch
import torch.nn.functional as F
from mmseg.apis import init_model, inference_model


# =========================
# 🔧 配置区
# =========================
IN_DIR = r"E:\MMseg\BIYELUNWEN\data\vis_compare_top50_pgf_better"
TEST_IMG_DIR = r"E:\MMseg\BIYELUNWEN\data\test\vis_images"
CONFIG_FILE = r"E:\MMseg\mmsegmentation\configs\segformer\segformer_mit-b1_landslide2.py"
CHECKPOINT_FILE = r"E:\MMseg\BIYELUNWEN\work_dirs\segformer_landslide\20260121_000000\segformer_landslide\best_mIoU_iter_93000.pth"

TARGET_RANKS = [3, 6, 35, 59, 22, 64]
OUTPUT_ROOT = osp.join(osp.dirname(__file__), "noise_experiments")
T = 20
CLOUD_DENSITY = 0.4
CLOUD_TRANSPARENCY = 0.7
SHADOW_AREA_RATIO = 0.4
SHADOW_INTENSITY = 0.6

GAP_PX = 14
GAP_COLOR_BGR = (192, 192, 192)
TOP_HEADER_H = 60
LEFT_LABEL_W = 120
HEADER_BG_BGR = (255, 255, 255)
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

NOISY_IMAGES_DIR = osp.join(OUTPUT_ROOT, "noisy_images")
INDIVIDUAL_RESULTS_DIR = osp.join(OUTPUT_ROOT, "individual_results")
os.makedirs(NOISY_IMAGES_DIR, exist_ok=True)
os.makedirs(INDIVIDUAL_RESULTS_DIR, exist_ok=True)


# =========================
# 🔍 文件查找函数
# =========================
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


# =========================
# 🌧️ 噪声模拟函数
# =========================
def generate_perlin_noise_2d(shape: Tuple[int, int], res: Tuple[int, int] = (4, 4)) -> np.ndarray:
    def f(t):
        return 6 * t**5 - 15 * t**4 + 10 * t**3
    delta = (res[0] / shape[0], res[1] / shape[1])
    d = (shape[0] // res[0], shape[1] // res[1])
    grid = np.mgrid[0:res[0]:delta[0], 0:res[1]:delta[1]].transpose(1, 2, 0) % 1
    angles = 2 * np.pi * np.random.rand(res[0] + 1, res[1] + 1)
    gradients = np.dstack((np.cos(angles), np.sin(angles)))
    g00 = gradients[0:-1, 0:-1].repeat(d[0], 0).repeat(d[1], 1)
    g10 = gradients[1:, 0:-1].repeat(d[0], 0).repeat(d[1], 1)
    g01 = gradients[0:-1, 1:].repeat(d[0], 0).repeat(d[1], 1)
    g11 = gradients[1:, 1:].repeat(d[0], 0).repeat(d[1], 1)
    n00 = np.sum(grid * g00, 2)
    n10 = np.sum(np.dstack((grid[:, :, 0] - 1, grid[:, :, 1])) * g10, 2)
    n01 = np.sum(np.dstack((grid[:, :, 0], grid[:, :, 1] - 1)) * g01, 2)
    n11 = np.sum(np.dstack((grid[:, :, 0], grid[:, :, 1] - 1)) * g11, 2)
    t_ = f(grid)
    n0 = n00 * (1 - t_[:, :, 0]) + t_[:, :, 0] * n10
    n1 = n01 * (1 - t_[:, :, 0]) + t_[:, :, 0] * n11
    return np.sqrt(2) * ((1 - t_[:, :, 1]) * n0 + t_[:, :, 1] * n1)

def generate_cloud_mask(shape: Tuple[int, int], density: float = 0.6) -> np.ndarray:
    h, w = shape
    noise1 = generate_perlin_noise_2d((h, w), (4, 4))
    noise2 = generate_perlin_noise_2d((h, w), (8, 8))
    noise3 = generate_perlin_noise_2d((h, w), (16, 16))
    cloud = (noise1 + 0.5 * noise2 + 0.25 * noise3) / 1.75
    cloud = (cloud - cloud.min()) / (cloud.max() - cloud.min())
    cloud = np.where(cloud > (1 - density), cloud, 0)
    cloud = cv2.GaussianBlur(cloud, (21, 21), 5)
    return cloud

def generate_shadow_mask(shape: Tuple[int, int], area_ratio: float = 0.2) -> np.ndarray:
    h, w = shape
    mask = np.zeros((h, w), dtype=np.float32)
    num_vertices = np.random.randint(5, 10)
    center_x = np.random.randint(w // 4, 3 * w // 4)
    center_y = np.random.randint(h // 4, 3 * h // 4)
    vertices = []
    for _ in range(num_vertices):
        angle = np.random.uniform(0, 2 * np.pi)
        radius = np.random.uniform(min(h, w) * 0.2, min(h, w) * 0.4)
        x = int(center_x + radius * np.cos(angle))
        y = int(center_y + radius * np.sin(angle))
        vertices.append((x, y))
    vertices = np.array(vertices, dtype=np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(mask, [vertices], 1.0)
    dist_transform = cv2.distanceTransform((1 - mask).astype(np.uint8), cv2.DIST_L2, 5)
    dist_transform = dist_transform / dist_transform.max()
    shadow = 1 - dist_transform
    shadow = np.where(mask > 0, shadow, 0)
    shadow = cv2.GaussianBlur(shadow, (15, 15), 3)
    return shadow

def add_clouds(img_bgr: np.ndarray, density: float = 0.6, transparency: float = 0.7) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    cloud_mask = generate_cloud_mask((h, w), density)
    cloud_color = np.array([255, 255, 245], dtype=np.float32)
    result = img_bgr.astype(np.float32)
    for c in range(3):
        result[:, :, c] = result[:, :, c] * (1 - cloud_mask * transparency) + cloud_color[c] * cloud_mask * transparency
    return result.astype(np.uint8)

def add_shadow(img_bgr: np.ndarray, area_ratio: float = 0.2, intensity: float = 0.6) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    shadow_mask = generate_shadow_mask((h, w), area_ratio)
    result = img_bgr.astype(np.float32)
    for c in range(3):
        result[:, :, c] = result[:, :, c] * (1 - shadow_mask * intensity)
    return result.astype(np.uint8)


# =========================
# 🤖 模型推理函数
# =========================
def enable_dropout_only(model):
    model.eval()
    for m in model.modules():
        if "dropout" in m.__class__.__name__.lower():
            m.train()
    return model

@torch.no_grad()
def mc_dropout_maps(model, img_path: str, T: int):
    enable_dropout_only(model)
    preds = []
    for _ in range(T):
        result = inference_model(model, img_path)
        seg_logits = result.seg_logits.data
        if seg_logits.dim() == 4:
            seg_logits = seg_logits[0]
        prob = F.softmax(seg_logits, dim=0)
        preds.append(prob.cpu().numpy())
    preds = np.stack(preds, axis=0)
    mean_pred = preds.mean(axis=0)
    var_pred = preds.var(axis=0)
    target_class = 0
    mean_map = mean_pred[target_class]
    var_map = var_pred[target_class]
    eps = 1e-8
    p = mean_map
    entropy_map = -(p * np.log(p + eps) + (1 - p) * np.log(1 - p + eps))
    pred01 = (mean_map >= 0.5).astype(np.uint8)
    return pred01, mean_map, entropy_map, var_map

def normalize_to_uint8(x: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    x = np.clip(x, vmin, vmax)
    if vmax <= vmin:
        vmax = vmin + 1e-12
    x = (x - vmin) / (vmax - vmin)
    return (x * 255.0).astype(np.uint8)

def apply_colormap(gray_u8: np.ndarray, cmap: int) -> np.ndarray:
    return cv2.applyColorMap(gray_u8, cmap)

def process_single_image(model, img_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
    pred01, p_mean, ent, var = mc_dropout_maps(model, img_path, T=T)
    pred_vis = np.zeros_like(img_bgr)
    pred_vis[pred01.astype(bool)] = (255, 255, 255)
    mean_gray = normalize_to_uint8(p_mean, MEAN_VMIN, MEAN_VMAX)
    mean_hm = apply_colormap(255 - mean_gray, CMAP_MEAN)
    ent_hm = apply_colormap(normalize_to_uint8(ent, ENT_VMIN, ENT_VMAX), CMAP_ENT)
    v_upper = float(np.quantile(var, VAR_Q))
    if not np.isfinite(v_upper) or v_upper <= 1e-12:
        v_upper = float(var.max()) if np.isfinite(var.max()) and var.max() > 0 else 1e-6
    var_hm = apply_colormap(normalize_to_uint8(var, VAR_VMIN, v_upper), CMAP_VAR)
    return img_bgr, pred_vis, mean_hm, ent_hm, var_hm


# =========================
# 🎨 辅助绘图函数
# =========================
def bgr_to_pil(img_bgr: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))

def draw_centered_multiline_text(draw, box, text, font, fill_rgb=(0,0,0), line_spacing_px=4):
    x0, y0, x1, y1 = box
    W = x1 - x0
    H = y1 - y0
    lines = text.split("\n")
    line_sizes = []
    total_h = 0
    for ln in lines:
        bbox = draw.textbbox((0,0), ln, font=font)
        w = bbox[2]-bbox[0]
        h = bbox[3]-bbox[1]
        line_sizes.append((w,h))
        total_h += h
    total_h += line_spacing_px*(len(lines)-1) if len(lines)>1 else 0
    yy = y0 + (H-total_h)//2
    for ln,(tw,th) in zip(lines, line_sizes):
        xx = x0 + (W-tw)//2
        draw.text((xx,yy), ln, font=font, fill=fill_rgb)
        yy += th + line_spacing_px

def draw_vertical_label(base_img, box, text, font, fill_rgb=(0,0,0), line_spacing_px=4, rotate_deg=90):
    x0,y0,x1,y1 = box
    bw = x1-x0
    bh = y1-y0
    dummy = Image.new("RGB", (10,10), (255,255,255))
    ddraw = ImageDraw.Draw(dummy)
    lines = text.split("\n")
    line_bboxes = [ddraw.textbbox((0,0), ln, font=font) for ln in lines]
    line_ws = [bb[2]-bb[0] for bb in line_bboxes] if line_bboxes else [1]
    line_hs = [bb[3]-bb[1] for bb in line_bboxes] if line_bboxes else [1]
    text_w = max(line_ws)
    text_h = sum(line_hs) + (line_spacing_px*(len(lines)-1) if len(lines)>1 else 0)
    pad = 15
    label_img = Image.new("RGB", (text_w+2*pad, text_h+2*pad), (255,255,255))
    ldraw = ImageDraw.Draw(label_img)
    yy = pad
    for ln,h in zip(lines, line_hs):
        bb = ldraw.textbbox((0,0), ln, font=font)
        lw = bb[2]-bb[0]
        ldraw.text((pad+(text_w-lw)//2, yy), ln, font=font, fill=fill_rgb)
        yy += h + line_spacing_px
    label_rot = label_img.rotate(rotate_deg, expand=True, fillcolor=(255,255,255))
    rw, rh = label_rot.size
    px = x0 + (bw - rw)//2
    py = y0 + (bh - rh)//2
    base_img.paste(label_rot, (px, py))


# =========================
# 🎨 原始布局函数（用于汇总图）
# =========================
def make_vertical_colorbar(h: int, cmap: int, width: int) -> np.ndarray:
    grad = np.linspace(255, 0, h, dtype=np.uint8).reshape(h, 1)
    grad = np.repeat(grad, width, axis=1)
    return cv2.applyColorMap(grad, cmap)

def pack_cbar(bar_bgr: np.ndarray, right_cbar_w: int, left_pad: int) -> np.ndarray:
    out = np.full((bar_bgr.shape[0], right_cbar_w, 3), 255, dtype=np.uint8)
    x0 = left_pad
    out[:, x0:x0+bar_bgr.shape[1]] = bar_bgr
    return out

def generate_comparison_image(results_list: List[Tuple], col_labels: List[str],
                              row_titles: List[str], save_path: str):
    H, W = results_list[0][0].shape[:2]
    num_cols = len(results_list)
    num_rows = len(row_titles)
    def resize_list(imgs):
        out = []
        for im in imgs:
            if im.shape[:2] != (H, W):
                im = cv2.resize(im, (W, H), interpolation=cv2.INTER_CUBIC)
            out.append(im)
        return out
    rows = []
    for i in range(num_rows):
        row_imgs = [resize_list([res[i] for res in results_list])[j] for j in range(num_cols)]
        rows.append(row_imgs)
    gap_v = np.full((H, GAP_PX, 3), GAP_COLOR_BGR, dtype=np.uint8)
    def hconcat(imgs):
        row = imgs[0]
        for im in imgs[1:]:
            row = np.hstack([row, gap_v, im])
        return row
    row_rgb = hconcat(rows[0])
    row_pred = hconcat(rows[1])
    row_mean = hconcat(rows[2])
    row_ent = hconcat(rows[3])
    row_var = hconcat(rows[4])
    RIGHT_CBAR_W = 95
    CBAR_BAR_W = 12
    CBAR_LEFT_PAD = 6
    right_white = np.full((H, RIGHT_CBAR_W, 3), 255, dtype=np.uint8)
    cbar_mean = pack_cbar(make_vertical_colorbar(H, CMAP_MEAN, CBAR_BAR_W), RIGHT_CBAR_W, CBAR_LEFT_PAD)
    cbar_ent = pack_cbar(make_vertical_colorbar(H, CMAP_ENT, CBAR_BAR_W), RIGHT_CBAR_W, CBAR_LEFT_PAD)
    cbar_var = pack_cbar(make_vertical_colorbar(H, CMAP_VAR, CBAR_BAR_W), RIGHT_CBAR_W, CBAR_LEFT_PAD)
    row_rgb = np.hstack([row_rgb, right_white])
    row_pred = np.hstack([row_pred, right_white])
    row_mean = np.hstack([row_mean, cbar_mean])
    row_ent = np.hstack([row_ent, cbar_ent])
    row_var = np.hstack([row_var, cbar_var])
    content_w_cols = row_rgb.shape[1] - RIGHT_CBAR_W
    gap_h_left = np.full((GAP_PX, content_w_cols, 3), GAP_COLOR_BGR, dtype=np.uint8)
    gap_h_right = np.full((GAP_PX, RIGHT_CBAR_W, 3), 255, dtype=np.uint8)
    gap_h = np.hstack([gap_h_left, gap_h_right])
    content = row_rgb
    for r in [row_pred, row_mean, row_ent, row_var]:
        content = np.vstack([content, gap_h, r])
    content_h, content_w = content.shape[:2]
    total_w = LEFT_LABEL_W + content_w
    total_h = TOP_HEADER_H + content_h
    canvas = np.full((total_h, total_w, 3), HEADER_BG_BGR, dtype=np.uint8)
    canvas[TOP_HEADER_H:TOP_HEADER_H+content_h, LEFT_LABEL_W:LEFT_LABEL_W+content_w] = content
    canvas_pil = bgr_to_pil(canvas)
    draw = ImageDraw.Draw(canvas_pil)
    font_cn = ImageFont.truetype(FONT_CN_PATH, FONT_SIZE_PX)
    font_en = ImageFont.truetype(FONT_EN_PATH, FONT_SIZE_PX)
    for j, lab in enumerate(col_labels):
        x0 = LEFT_LABEL_W + j*W + j*GAP_PX
        x1 = x0 + W
        draw_centered_multiline_text(draw, (x0,0,x1,TOP_HEADER_H), lab, font_en, fill_rgb=TEXT_COLOR_RGB)
    for i, title in enumerate(row_titles):
        yy0 = TOP_HEADER_H + i*H + i*GAP_PX
        yy1 = yy0 + H
        draw_vertical_label(canvas_pil, (0,yy0,LEFT_LABEL_W,yy1), title, font_cn, fill_rgb=TEXT_COLOR_RGB)
    x_cbar0 = LEFT_LABEL_W + (num_cols*W + (num_cols-1)*GAP_PX)
    y_mean0 = TOP_HEADER_H + 2*H + 2*GAP_PX
    y_ent0 = TOP_HEADER_H + 3*H + 3*GAP_PX
    y_var0 = TOP_HEADER_H + 4*H + 4*GAP_PX
    text_x = x_cbar0 + CBAR_LEFT_PAD + CBAR_BAR_W + 6
    draw.text((text_x, y_mean0+4), "1.0", font=font_en, fill=TEXT_COLOR_RGB)
    draw.text((text_x, y_mean0+H-34), "0.0", font=font_en, fill=TEXT_COLOR_RGB)
    draw.text((text_x, y_ent0+4), "ln2", font=font_en, fill=TEXT_COLOR_RGB)
    draw.text((text_x, y_ent0+H-34), "0", font=font_en, fill=TEXT_COLOR_RGB)
    draw.text((text_x, y_var0+4), "Q99", font=font_en, fill=TEXT_COLOR_RGB)
    draw.text((text_x, y_var0+H-34), "0", font=font_en, fill=TEXT_COLOR_RGB)
    canvas_pil.save(save_path, dpi=(1000,1000))
    print(f"Saved comparison image (original layout) to: {save_path}")


# =========================
# 🎨 新布局函数（三行五列，底部水平色带，尺寸与原始一致）
# =========================
def generate_comparison_image_new(results_list: List[Tuple],
                                  row_titles: List[str],
                                  col_labels: List[str],
                                  save_path: str):
    H, W = results_list[0][0].shape[:2]
    num_rows = len(results_list)
    num_cols = len(results_list[0])

    def resize_img(img):
        if img.shape[:2] != (H, W):
            return cv2.resize(img, (W, H), interpolation=cv2.INTER_CUBIC)
        return img

    # 构建网格
    rows = []
    for r in range(num_rows):
        row_imgs = [resize_img(results_list[r][c]) for c in range(num_cols)]
        rows.append(row_imgs)

    gap_v = np.full((H, GAP_PX, 3), GAP_COLOR_BGR, dtype=np.uint8)
    def hconcat_row(imgs):
        row = imgs[0]
        for img in imgs[1:]:
            row = np.hstack([row, gap_v, img])
        return row

    content_rows = [hconcat_row(r) for r in rows]
    content_w = content_rows[0].shape[1]
    gap_h_full = np.full((GAP_PX, content_w, 3), GAP_COLOR_BGR, dtype=np.uint8)
    content = content_rows[0]
    for r in range(1, num_rows):
        content = np.vstack([content, gap_h_full, content_rows[r]])
    content_h, content_w = content.shape[:2]

    RIGHT_PAD = 30
    total_w = LEFT_LABEL_W + content_w + RIGHT_PAD

    # 色带参数与原始竖直色带一致（高度12像素，无边框）
    CBAR_BAR_H = 12
    BOTTOM_EXTRA_H = CBAR_BAR_H + 35   # 为标注文字留出空间
    total_h = TOP_HEADER_H + content_h + BOTTOM_EXTRA_H + 10

    canvas = np.full((total_h, total_w, 3), HEADER_BG_BGR, dtype=np.uint8)
    canvas[TOP_HEADER_H:TOP_HEADER_H+content_h, LEFT_LABEL_W:LEFT_LABEL_W+content_w] = content

    # 在 NumPy 上绘制水平色带（仅第3,4,5列）
    cbar_col_indices = [2, 3, 4]
    cmap_list = [CMAP_MEAN, CMAP_ENT, CMAP_VAR]
    y_bar_start = TOP_HEADER_H + content_h + 5

    for idx, col_idx in enumerate(cbar_col_indices):
        x0_bar = LEFT_LABEL_W + col_idx * (W + GAP_PX)
        x1_bar = x0_bar + W
        grad = np.linspace(0, 255, W, dtype=np.uint8).reshape(1, W)
        grad = np.repeat(grad, CBAR_BAR_H, axis=0)
        cbar_bgr = cv2.applyColorMap(grad, cmap_list[idx])
        if y_bar_start + CBAR_BAR_H <= canvas.shape[0]:
            canvas[y_bar_start:y_bar_start+CBAR_BAR_H, x0_bar:x1_bar] = cbar_bgr
        else:
            # 自动扩展画布（极少发生）
            new_h = y_bar_start + CBAR_BAR_H + 10
            canvas = np.vstack([canvas, np.full((new_h - canvas.shape[0], total_w, 3), HEADER_BG_BGR, dtype=np.uint8)])
            canvas[y_bar_start:y_bar_start+CBAR_BAR_H, x0_bar:x1_bar] = cbar_bgr

    canvas_pil = bgr_to_pil(canvas)
    draw = ImageDraw.Draw(canvas_pil)
    font_cn = ImageFont.truetype(FONT_CN_PATH, FONT_SIZE_PX)

    # 顶部列标题
    for c, lab in enumerate(col_labels):
        x0 = LEFT_LABEL_W + c * (W + GAP_PX)
        x1 = x0 + W
        draw_centered_multiline_text(draw, (x0, 0, x1, TOP_HEADER_H), lab, font_cn, fill_rgb=TEXT_COLOR_RGB)

    # 左侧行标题
    for r, title in enumerate(row_titles):
        y0 = TOP_HEADER_H + r * (H + GAP_PX)
        y1 = y0 + H
        draw_vertical_label(canvas_pil, (0, y0, LEFT_LABEL_W, y1), title, font_cn, fill_rgb=TEXT_COLOR_RGB)

    # 色带标注文字
    for idx, col_idx in enumerate(cbar_col_indices):
        x0_bar = LEFT_LABEL_W + col_idx * (W + GAP_PX)
        x1_bar = x0_bar + W
        text_y = y_bar_start + CBAR_BAR_H + 2
        if col_idx == 2:
            min_text, max_text = "0.0", "1.0"
        elif col_idx == 3:
            min_text, max_text = "0", "ln2"
        else:
            min_text, max_text = "0", "Q99"
        draw.text((x0_bar + 5, text_y), min_text, font=font_cn, fill=TEXT_COLOR_RGB)
        bbox = draw.textbbox((0, 0), max_text, font=font_cn)
        tw = bbox[2] - bbox[0]
        draw.text((x1_bar - tw - 5, text_y), max_text, font=font_cn, fill=TEXT_COLOR_RGB)

    canvas_pil.save(save_path, dpi=(1000, 1000))
    print(f"Saved comparison image (new layout) to: {save_path}")


# =========================
# 🚀 主程序
# =========================
def main():
    print("=" * 60)
    print("噪声模拟与对比实验脚本（已调低云密度）")
    print(f"输出目录: {OUTPUT_ROOT}")
    print(f"处理图片: rank={TARGET_RANKS}")
    print(f"云密度: {CLOUD_DENSITY}")
    print("=" * 60)

    print("\nLoading model...")
    model = init_model(CONFIG_FILE, CHECKPOINT_FILE, device="cuda:0")

    row_titles_new = ["原始", "云遮挡", "地形阴影"]
    col_labels_new = ["影像", "预测结果", "概率均值", "熵", "方差"]

    all_image_results = []

    with tempfile.TemporaryDirectory() as tmpdir:
        print(f"\n临时目录: {tmpdir}")

        for rank in TARGET_RANKS:
            print(f"\nProcessing rank={rank}...")
            rank_file = find_file_for_rank(rank)
            stem = parse_stem_from_rank_file(rank_file)
            img_path = find_image_by_stem(TEST_IMG_DIR, stem)
            print(f"  Found image: {osp.basename(img_path)}")

            original_img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if original_img is None:
                print(f"Warning: Cannot read image {img_path}, skipping...")
                continue

            print("  Generating noisy images...")
            cloud_img = add_clouds(original_img, CLOUD_DENSITY, CLOUD_TRANSPARENCY)
            shadow_img = add_shadow(original_img, SHADOW_AREA_RATIO, SHADOW_INTENSITY)

            tmp_cloud_path = osp.join(tmpdir, f"tmp_{rank}_cloud.png")
            tmp_shadow_path = osp.join(tmpdir, f"tmp_{rank}_shadow.png")
            cv2.imwrite(tmp_cloud_path, cloud_img)
            cv2.imwrite(tmp_shadow_path, shadow_img)

            cv2.imwrite(osp.join(NOISY_IMAGES_DIR, f"rank_{rank}_cloud.png"), cloud_img)
            cv2.imwrite(osp.join(NOISY_IMAGES_DIR, f"rank_{rank}_shadow.png"), shadow_img)
            print(f"  Saved noisy images to {NOISY_IMAGES_DIR}")

            print("  Running inference...")
            results_original = process_single_image(model, img_path)
            results_cloud = process_single_image(model, tmp_cloud_path)
            results_shadow = process_single_image(model, tmp_shadow_path)

            individual_save_path = osp.join(INDIVIDUAL_RESULTS_DIR, f"rank_{rank}_comparison.png")
            results_list_new = [results_original, results_cloud, results_shadow]
            generate_comparison_image_new(results_list_new, row_titles_new, col_labels_new, individual_save_path)

            all_image_results.append((rank, results_original, results_cloud, results_shadow))

    if len(all_image_results) > 0:
        print("\nGenerating summary comparison image (original layout)...")
        summary_results = []
        summary_col_labels = []
        for rank, res_orig, res_cloud, res_shadow in all_image_results:
            summary_results.extend([res_orig, res_cloud, res_shadow])
            summary_col_labels.extend([
                f"Rank {rank}\n原始",
                f"Rank {rank}\n云遮挡",
                f"Rank {rank}\n地形阴影"
            ])
        row_titles_old = ["影像", "预测结果", "概率均值", "熵", "方差"]
        summary_save_path = osp.join(OUTPUT_ROOT, "all_6_images_summary.png")
        generate_comparison_image(summary_results, summary_col_labels, row_titles_old, summary_save_path)

    print("\n" + "=" * 60)
    print("所有处理完成！")
    print(f"加噪声后的原始图片: {NOISY_IMAGES_DIR}")
    print(f"单张图片对比结果: {INDIVIDUAL_RESULTS_DIR}")
    print(f"6张图片汇总对比结果: {osp.join(OUTPUT_ROOT, 'all_6_images_summary.png')}")
    print("=" * 60)


if __name__ == "__main__":
    main()