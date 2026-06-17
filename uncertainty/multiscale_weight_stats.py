import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from scipy import ndimage
from scipy.ndimage import distance_transform_edt, label, generate_binary_structure
from tqdm import tqdm


plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

# ========== 配置路径 ==========
GT_MASK_DIR    = r"E:/MMseg/BIYELUNWEN/work_dirs/segformer_landslide/mc_results2_final/npy/gt/"
PRED_MASK_DIR  = r"E:/MMseg/BIYELUNWEN/work_dirs/segformer_landslide/mc_results2_final/npy/pred_det/"
GATES_MEAN_DIR = r"E:/MMseg/BIYELUNWEN/work_dirs/segformer_landslide/mc_results2_final/npy/gates_mean/"

OUTPUT_DIR = r"./analysis_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

BORDER_DIST_THRESH = 2
SMALL_AREA_THRESH = 50

# ========== 提取核心ID ==========
def extract_core_id(filename, suffix='.npy'):
    base = filename.replace(suffix, '')
    parts = base.split('_')
    if len(parts) >= 2 and parts[0] == 'img':
        return f"{parts[0]}_{parts[1]}"
    return base

def get_core_ids_from_dir(dir_path, suffix='.npy'):
    files = os.listdir(dir_path)
    core_ids = set()
    for f in files:
        if f.endswith(suffix):
            core_ids.add(extract_core_id(f, suffix))
    return core_ids

gt_cores = get_core_ids_from_dir(GT_MASK_DIR)
pred_cores = get_core_ids_from_dir(PRED_MASK_DIR)
gates_cores = get_core_ids_from_dir(GATES_MEAN_DIR)

print(f"GT目录核心ID数: {len(gt_cores)}，示例: {list(gt_cores)[:5]}")
print(f"PRED目录核心ID数: {len(pred_cores)}，示例: {list(pred_cores)[:5]}")
print(f"GATES目录核心ID数: {len(gates_cores)}，示例: {list(gates_cores)[:5]}")

common_cores = sorted(gt_cores & pred_cores & gates_cores)
print(f"共有核心ID数量: {len(common_cores)}")

if len(common_cores) == 0:
    raise RuntimeError("没有找到共有图像，请检查路径和文件。")
else:
    image_ids = common_cores

print(f"最终参与分析的图像数量: {len(image_ids)}")

def find_file(dir_path, core_id):
    for f in os.listdir(dir_path):
        if core_id in f and f.endswith('.npy'):
            return os.path.join(dir_path, f)
    raise FileNotFoundError(f"在 {dir_path} 中找不到包含 {core_id} 的 .npy 文件")

# ========== 初始化存储容器 ==========
region_names = ['interior', 'border', 'small_slide', 'confused_bg']
scale_names = ['Scale1', 'Scale2', 'Scale3', 'Scale4']
weight_data = {region: {scale: [] for scale in scale_names} for region in region_names}

def get_boundary_distance_mask(binary_mask):
    return distance_transform_edt(binary_mask)

def get_connected_components(binary_mask):
    s = generate_binary_structure(2, 2)
    labeled, num = label(binary_mask, structure=s)
    sizes = ndimage.sum(binary_mask, labeled, range(1, num+1))
    return labeled, sizes

# ========== 逐图像处理 ==========
for core_id in tqdm(image_ids, desc="处理图像"):
    gt_path = find_file(GT_MASK_DIR, core_id)
    pred_path = find_file(PRED_MASK_DIR, core_id)
    gates_path = find_file(GATES_MEAN_DIR, core_id)

    gt = np.load(gt_path).astype(np.uint8)
    pred = np.load(pred_path).astype(np.uint8)
    gates = np.load(gates_path)

    if gates.ndim == 3 and gates.shape[0] == 4:
        gates = gates.transpose(1, 2, 0)
    else:
        raise ValueError(f"gates格式错误，应为4×H×W，实际{gates.shape}")

    if gates.shape[:2] != gt.shape:
        gates = cv2.resize(gates, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_LINEAR)
        gates = gates.astype(np.float32)

    tp_mask = (pred == 1) & (gt == 1)
    dist_to_border = get_boundary_distance_mask(tp_mask)

    interior_mask_temp = (tp_mask) & (dist_to_border > BORDER_DIST_THRESH)
    border_mask_temp   = (tp_mask) & (dist_to_border <= BORDER_DIST_THRESH)

    labeled_tp, sizes_tp = get_connected_components(tp_mask)
    small_comp_ids = np.where(sizes_tp <= SMALL_AREA_THRESH)[0] + 1
    small_slide_mask = (tp_mask) & (np.isin(labeled_tp, small_comp_ids))

    interior_mask = interior_mask_temp & (~small_slide_mask)
    border_mask   = border_mask_temp   & (~small_slide_mask)

    fp_mask = (pred == 1) & (gt == 0)
    confused_bg_mask = fp_mask

    masks = {
        'interior': interior_mask,
        'border': border_mask,
        'small_slide': small_slide_mask,
        'confused_bg': confused_bg_mask
    }

    for region_name, mask in masks.items():
        if np.any(mask):
            region_weights = gates[mask, :]
            for s_idx, scale_name in enumerate(scale_names):
                weight_data[region_name][scale_name].extend(region_weights[:, s_idx].tolist())

# ========== 转换为DataFrame ==========
rows = []
for region in region_names:
    for scale in scale_names:
        values = weight_data[region][scale]
        if values:
            for v in values:
                rows.append({'Region': region, 'Scale': scale, 'Weight': v})

df = pd.DataFrame(rows)

if df.empty:
    raise RuntimeError("没有收集到任何权重数据，请检查区域掩膜是否正确生成。")


plt.figure(figsize=(12, 6))


region_labels = ['滑坡内部', '滑坡边界', '细小滑坡', '混淆背景']
df['Region_en'] = df['Region'].map(dict(zip(region_names, region_labels)))


palette = sns.color_palette("Set2", n_colors=4)


ax = sns.violinplot(x='Region_en', y='Weight', hue='Scale', data=df,
                    palette=palette, split=False, inner='quartile',
                    linewidth=1.2, cut=0)

ax.set_xlabel('')
ax.set_ylabel('尺度分配权重', fontsize=12)
ax.tick_params(axis='both', labelsize=11)


handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, ['尺度 1', '尺度 2', '尺度 3', '尺度 4'],
          title='尺度', loc='upper right', fontsize=10, title_fontsize=11)

plt.tight_layout()

# 保存图片
fig_path = os.path.join(OUTPUT_DIR, 'fig5_2_region_scale_violin.png')
plt.savefig(fig_path, dpi=300, bbox_inches='tight')
plt.show()


with open(os.path.join(OUTPUT_DIR, 'fig5_2_caption.txt'), 'w', encoding='utf-8') as f:
    f.write(caption)
print(f"图注说明已保存至 {os.path.join(OUTPUT_DIR, 'fig5_2_caption.txt')}")

# ========== 计算统计表 ==========
stats = []
for region in region_names:
    for scale in scale_names:
        values = weight_data[region][scale]
        if values:
            mean = np.mean(values)
            std = np.std(values)
            p25 = np.percentile(values, 25)
            p50 = np.median(values)
            p75 = np.percentile(values, 75)
        else:
            mean = std = p25 = p50 = p75 = np.nan
        stats.append({
            'Region': region,
            'Scale': scale,
            'Mean': mean,
            'Std': std,
            'Q1': p25,
            'Median': p50,
            'Q3': p75
        })

df_stats = pd.DataFrame(stats)
print("\n表 各区域尺度权重统计")
print(df_stats.to_string(index=False))

df_stats.to_csv(os.path.join(OUTPUT_DIR, 'table5_1_region_scale_stats.csv'), index=False)
print(f"\n结果已保存至 {OUTPUT_DIR}")