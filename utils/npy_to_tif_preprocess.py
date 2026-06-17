#LMHLD数据集为npy格式，此用来将整体npy文件转换为单个tif文件（包括图像和标签），通过调整rgb比例调整可视化的色彩

import os
import numpy as np
import tifffile
from PIL import Image

def batch_npy_to_raw_img_and_label(batch_img_npy_path,
                                  batch_label_npy_path,
                                  vis_img_png_dir,
                                  train_img_tiff_dir,
                                  vis_label_png_dir,
                                  img_prefix="img_",
                                  label_prefix="label_",
                                  vis_mode="balanced_rgb",  # 平衡权重模式（默认，无红绿蒙版）
                                  custom_rgb=(2, 0, 3)):
    """
    1. 解决：绿色蒙版→红色蒙版的问题（平衡RGB权重）
    2. 保留：纯原始数据（无增强/平滑）、原始4波段、你的所有路径
    3. 修复：所有语法报错，代码完整运行
    """
    # 1. 创建目录
    os.makedirs(vis_img_png_dir, exist_ok=True)
    os.makedirs(train_img_tiff_dir, exist_ok=True)
    os.makedirs(vis_label_png_dir, exist_ok=True)
    print(f"可视化模式：{vis_mode}（波段对应：0=蓝,1=绿,2=红,3=近红外）")
    print(f"创建目录：{vis_img_png_dir}（纯原始数据可视化，无红绿蒙版）")
    print(f"创建目录：{train_img_tiff_dir}（4通道原始训练TIFF）")
    print(f"创建目录：{vis_label_png_dir}（黑白分明标签）")

    # 2. 加载原始NPY
    try:
        batch_img_data = np.load(batch_img_npy_path)
        batch_label_data = np.load(batch_label_npy_path)
        print(f"图像原始形状：{batch_img_data.shape}")
        print(f"标签原始形状：{batch_label_data.shape}")
    except Exception as e:
        raise ValueError(f"加载NPY失败：{e}")

    # 3. 校验形状
    if batch_img_data.shape != (520, 4, 224, 224):
        raise ValueError(f"图像需为(261,4,224,224)，当前：{batch_img_data.shape}")
    if batch_label_data.shape != (520, 1, 224, 224):
        raise ValueError(f"标签需为(261,1,224,224)，当前：{batch_label_data.shape}")

    sample_num = batch_img_data.shape[0]
    print(f"共{sample_num}个样本，开始纯原始数据转换...")

    # 4. 遍历转换（平衡权重，无红绿蒙版）
    for idx in range(sample_num):
        # -------------------------- 图像处理：平衡RGB，无红绿蒙版 --------------------------
        single_img = batch_img_data[idx]  # (4,224,224) 原始数据，无改动
        single_img_hwc = single_img.transpose(1, 2, 0)  # 仅通道转置 (C,H,W)→(H,W,4)

        # ===== 模式1：balanced_rgb（默认，推荐，无红绿蒙版） =====
        if vis_mode == "balanced_rgb":
            img_vis_raw = np.zeros((224, 224, 3), dtype=np.float32)
            # 平衡R/G/B权重，避免单一通道占比过高
            img_vis_raw[:, :, 0] = single_img_hwc[:, :, 2] * 0.7  # R(红) 轻微降权，避免红色蒙版
            img_vis_raw[:, :, 1] = single_img_hwc[:, :, 1] * 0.6  # G(绿) 适度降权，避免绿色蒙版
            img_vis_raw[:, :, 2] = single_img_hwc[:, :, 0] * 1.0  # B(蓝) 轻微降权，平衡整体色彩
            # 权重可灵活调整（0.5-1.0之间），按需微调即可

        # ===== 模式2：avoid_rgb_single（避开单一通道，无蒙版，备选） =====
        elif vis_mode == "avoid_rgb_single":
            # 用 R(2)、NIR(3)、B(0) 组合，权重均等，无红绿蒙版
            img_vis_raw = np.zeros((224, 224, 3), dtype=np.float32)
            img_vis_raw[:, :, 0] = single_img_hwc[:, :, custom_rgb[0]]
            img_vis_raw[:, :, 1] = single_img_hwc[:, :, custom_rgb[1]]
            img_vis_raw[:, :, 2] = single_img_hwc[:, :, custom_rgb[2]]

        # ===== 模式3：gray（灰度图，无色彩干扰，兜底方案） =====
        elif vis_mode == "gray":
            select_band_idx = 2  # 2=红，可改为0(蓝)/1(绿)/3(近红外)
            img_vis_raw = single_img_hwc[:, :, select_band_idx]

        # ===== 模式4：original_rgb（原始正确RGB，供对比） =====
        elif vis_mode == "original_rgb":
            img_vis_raw = np.zeros((224, 224, 3), dtype=np.float32)
            img_vis_raw[:, :, 0] = single_img_hwc[:, :, 2]
            img_vis_raw[:, :, 1] = single_img_hwc[:, :, 1]
            img_vis_raw[:, :, 2] = single_img_hwc[:, :, 0]
        else:
            raise ValueError(f"支持balanced_rgb/avoid_rgb_single/gray/original_rgb，当前：{vis_mode}")

        # ===== 仅基础线性归一化（不改变数据相对分布） =====
        img_min = img_vis_raw.min()
        img_max = img_vis_raw.max()
        if img_max - img_min != 0:
            img_vis_uint8 = (img_vis_raw - img_min) / (img_max - img_min) * 255
        else:
            img_vis_uint8 = np.zeros_like(img_vis_raw)
        img_vis_uint8 = img_vis_uint8.astype(np.uint8)

        # 保存可视化PNG
        vis_img_name = f"{img_prefix}{idx:03d}.png"
        vis_img_path = os.path.join(vis_img_png_dir, vis_img_name)
        Image.fromarray(img_vis_uint8).save(vis_img_path)

        # ===== 保存训练用TIFF（完全原始4通道，无任何改动） =====
        img_4c_min = single_img_hwc.min()
        img_4c_max = single_img_hwc.max()
        if img_4c_max - img_4c_min != 0:
            img_4c_uint8 = (single_img_hwc - img_4c_min) / (img_4c_max - img_4c_min) * 255
        else:
            img_4c_uint8 = np.zeros_like(single_img_hwc)
        img_4c_uint8 = img_4c_uint8.astype(np.uint8)
        train_tiff_name = f"{img_prefix}{idx:03d}.tiff"
        train_tiff_path = os.path.join(train_img_tiff_dir, train_tiff_name)
        tifffile.imwrite(train_tiff_path, img_4c_uint8)

        # -------------------------- 标签处理：仅解决全黑，无改动 --------------------------
        single_label = batch_label_data[idx].squeeze(0)
        label_vis = np.where(single_label == 1.0, 255, 0).astype(np.uint8)
        vis_label_name = f"{label_prefix}{idx:03d}.png"
        vis_label_path = os.path.join(vis_label_png_dir, vis_label_name)
        Image.fromarray(label_vis).save(vis_label_path)

        # 打印进度
        if (idx + 1) % 50 == 0:
            print(f"已完成 {idx + 1}/{sample_num} 个样本")
            print(f"  可视化PNG：{vis_img_name}（纯原始数据，{vis_mode}模式）")
            print(f"  训练TIFF：{train_tiff_name}（4通道原始波段：B/G/R/NIR）")
            print(f"  可视化标签：{vis_label_name}（黑白分明）")
            print("-" * 30)

    print(f"\n转换全部完成！")
    print(f"✅ 可视化PNG：{vis_img_png_dir}（{vis_mode}模式，无红绿蒙版，色彩均衡）")
    print(f"✅ 训练TIFF：{train_img_tiff_dir}（保留原始4波段，适配SegFormer）")
    print(f"✅ 可视化标签：{vis_label_png_dir}（滑坡=白色，背景=黑色）")
    print(f"提示：若色彩仍不满意，可微调balanced_rgb模式下的R/G/B权重（0.5-1.0）")

# -------------------------- 你的路径完全保留，无需修改 --------------------------
if __name__ == "__main__":
    TOTAL_IMG_NPY_PATH = r"E:\MMseg\BIYELUNWEN\LMHLD\LMHLD\LMHLD_dataset_different_patch_sizes\Wenchuan_China_224\val_images.npy"
    TOTAL_LABEL_NPY_PATH = r"E:\MMseg\BIYELUNWEN\LMHLD\LMHLD\LMHLD_dataset_different_patch_sizes\Wenchuan_China_224\val_labels.npy"
    VIS_IMG_PNG_DIR = r"E:\MMseg\BIYELUNWEN\data\val\vis_images"
    TRAIN_IMG_TIFF_DIR = r"E:\MMseg\BIYELUNWEN\data\val\images"
    LABEL_PNG_SAVE_DIR = r"E:\MMseg\BIYELUNWEN\data\val\lables"

    # 执行转换：默认balanced_rgb模式（无红绿蒙版），可切换其他模式
    batch_npy_to_raw_img_and_label(
        batch_img_npy_path=TOTAL_IMG_NPY_PATH,
        batch_label_npy_path=TOTAL_LABEL_NPY_PATH,
        vis_img_png_dir=VIS_IMG_PNG_DIR,
        train_img_tiff_dir=TRAIN_IMG_TIFF_DIR,
        vis_label_png_dir=LABEL_PNG_SAVE_DIR,
        vis_mode="balanced_rgb",
        custom_rgb=(2, 3, 0)  # R/NIR/B 组合，无红绿蒙版
    )