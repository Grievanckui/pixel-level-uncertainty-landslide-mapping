import os
import os.path as osp
import glob
from typing import List

import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

# =========================
# 1) 核心配置（按需修改）
# =========================
IN_DIR = r"E:\MMseg\BIYELUNWEN\data\vis_compare_top50_pgf_better"  # 用于通过rank找原图stem
TEST_IMG_DIR = r"E:\MMseg\BIYELUNWEN\data\test\vis_images"  # 测试原图目录
GT_LABEL_DIR = r"E:\MMseg\BIYELUNWEN\data\test\labels"  # GT标签目录（你要求的）

RANKS = [3, 6, 35, 59, 22, 64]  # 你要展示的图片序号

OUT_FILE = osp.join(IN_DIR, "rgb_gt_compare_3_6_35_59_22_64.png")  # 输出文件名

# =========================
# 2) 版式配置
# =========================
SCALE = 3.0
GAP_PX = 14
GAP_COLOR_BGR = (192, 192, 192)

TOP_HEADER_H = 0  # 去除顶部列标签，高度设为0
LEFT_LABEL_W = 60  # 左侧行标签宽度

HEADER_BG_BGR = (255, 255, 255)

ROW_TITLES = [
    "Image",
    "Annotation",  # 第二行改成GT
]

# =========================
# 3) 字体配置
# =========================
FONT_SIZE_PX = 32
FONT_CN_PATH = r"C:\Windows\Fonts\simsun.ttc"
FONT_EN_PATH = r"C:\Windows\Fonts\times.ttf"
TEXT_COLOR_RGB = (0, 0, 0)


# ---------- 选图与找GT工具 ----------
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


def find_gt_by_stem(gt_dir: str, stem: str) -> str:
    """专门找GT标签，适配img_000.png格式"""
    # 优先尝试直接匹配 stem.png
    gt_path = osp.join(gt_dir, f"{stem}.png")
    if osp.exists(gt_path):
        return gt_path

    # 如果没找到，尝试模糊匹配
    exts = (".png", ".tif", ".bmp")
    for e in exts:
        hits = glob.glob(osp.join(gt_dir, f"{stem}*{e}"))
        if hits:
            return sorted(hits)[0]

    raise FileNotFoundError(f"Cannot find GT label for stem={stem} in {gt_dir}")


def load_and_convert_gt(gt_path: str, target_shape: tuple) -> np.ndarray:
    """
    加载GT并转换为0-255的3通道BGR图
    target_shape: (H, W)，用于统一尺寸
    """
    # 读取GT（通常是单通道0/1图）
    gt_gray = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
    if gt_gray is None:
        raise FileNotFoundError(f"Cannot read GT: {gt_path}")

    # 统一尺寸
    if gt_gray.shape[:2] != target_shape:
        gt_gray = cv2.resize(gt_gray, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_NEAREST)

    # 核心转换：0->0, 1->255（防止全黑）
    gt_0255 = (gt_gray > 0).astype(np.uint8) * 255

    # 转成3通道BGR，方便和RGB拼接
    gt_bgr = cv2.cvtColor(gt_0255, cv2.COLOR_GRAY2BGR)
    return gt_bgr


# ---------- PIL 标注工具（保留行标签） ----------
def bgr_to_pil(img_bgr: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))


def pil_to_bgr(img_pil: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


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
    pad = 15

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

    # 1. 找图
    rank_files = [find_file_for_rank(r) for r in RANKS]
    stems = [parse_stem_from_rank_file(p) for p in rank_files]
    img_paths = [find_image_by_stem(TEST_IMG_DIR, s) for s in stems]
    gt_paths = [find_gt_by_stem(GT_LABEL_DIR, s) for s in stems]

    print("Selected (rank -> stem -> RGB -> GT):")
    for r, s, p, g in zip(RANKS, stems, img_paths, gt_paths):
        print(f"  rank={r:<3} stem={s:<20} rgb={osp.basename(p)} gt={osp.basename(g)}")

    # 2. 加载图片和GT
    panels_rgb = []
    panels_gt = []

    # 先读第一张图确定统一尺寸
    first_img = cv2.imread(img_paths[0], cv2.IMREAD_COLOR)
    if first_img is None:
        raise FileNotFoundError(img_paths[0])
    H, W = first_img.shape[:2]

    for img_path, gt_path in zip(img_paths, gt_paths):
        # 加载RGB
        img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise FileNotFoundError(img_path)
        if img_bgr.shape[:2] != (H, W):
            img_bgr = cv2.resize(img_bgr, (W, H), interpolation=cv2.INTER_AREA)

        # 加载并转换GT
        gt_bgr = load_and_convert_gt(gt_path, target_shape=(H, W))

        panels_rgb.append(img_bgr)
        panels_gt.append(gt_bgr)

    # 3. 拼接图片
    gap_v = np.full((H, GAP_PX, 3), GAP_COLOR_BGR, dtype=np.uint8)

    def hconcat_list(imgs: List[np.ndarray]) -> np.ndarray:
        row = imgs[0]
        for im in imgs[1:]:
            row = np.hstack([row, gap_v, im])
        return row

    row_rgb = hconcat_list(panels_rgb)
    row_gt = hconcat_list(panels_gt)

    # 行间灰线
    gap_h = np.full((GAP_PX, row_rgb.shape[1], 3), GAP_COLOR_BGR, dtype=np.uint8)

    # 垂直拼接两行
    content = np.vstack([row_rgb, gap_h, row_gt])
    content_h, content_w = content.shape[:2]

    # 4. 加左侧行标签
    total_w = LEFT_LABEL_W + content_w
    total_h = TOP_HEADER_H + content_h

    canvas = np.full((total_h, total_w, 3), HEADER_BG_BGR, dtype=np.uint8)
    canvas[TOP_HEADER_H:TOP_HEADER_H + content_h, LEFT_LABEL_W:LEFT_LABEL_W + content_w] = content

    canvas_pil = bgr_to_pil(canvas)
    draw = ImageDraw.Draw(canvas_pil)

    font_en = ImageFont.truetype(FONT_EN_PATH, FONT_SIZE_PX)
    font_cn = ImageFont.truetype(FONT_CN_PATH, FONT_SIZE_PX)

    # 绘制左侧行标题（RGB和GT）
    for i, title in enumerate(ROW_TITLES):
        yy0 = TOP_HEADER_H + i * H + i * GAP_PX
        yy1 = yy0 + H
        draw_vertical_label(canvas_pil, (0, yy0, LEFT_LABEL_W, yy1), title, font_en, fill_rgb=TEXT_COLOR_RGB)

    # 5. 保存
    out_bgr = pil_to_bgr(canvas_pil)
    cv2.imwrite(OUT_FILE, out_bgr)
    print("Saved:", OUT_FILE)


if __name__ == "__main__":
    main()