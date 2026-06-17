import os
import numpy as np
import pandas as pd
import cv2
from scipy import ndimage, stats
from scipy.ndimage import distance_transform_edt, label, generate_binary_structure
from tqdm import tqdm

# ========== 路径配置（请根据实际修改） ==========
GT_MASK_DIR    = r"E:/MMseg/BIYELUNWEN/work_dirs/segformer_landslide/mc_results2_final/npy/gt/"
PRED_MASK_DIR  = r"E:/MMseg/BIYELUNWEN/work_dirs/segformer_landslide/mc_results2_final/npy/pred_det/"
GATES_MEAN_DIR = r"E:/MMseg/BIYELUNWEN/work_dirs/segformer_landslide/mc_results2_final/npy/gates_mean/"
ENTROPY_DIR    = r"E:/MMseg/BIYELUNWEN/work_dirs/segformer_landslide/mc_results2_final/npy/entropy/"
VARIANCE_DIR   = r"E:/MMseg/BIYELUNWEN/work_dirs/segformer_landslide/mc_results2_final/npy/variance/"

OUTPUT_DIR = r"./analysis_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ========== 参数设置 ==========
BORDER_DIST_THRESH = 2
SMALL_AREA_THRESH = 100

# 区域名称（四个区域）
region_names = ['Interior', 'Border', 'Small', 'Confused']
region_map_int = {'interior':0, 'border':1, 'small_slide':2, 'confused_bg':3}

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
entropy_cores = get_core_ids_from_dir(ENTROPY_DIR)
variance_cores = get_core_ids_from_dir(VARIANCE_DIR)

common_cores = sorted(gt_cores & pred_cores & gates_cores & entropy_cores & variance_cores)
print(f"共有核心ID数量: {len(common_cores)}")
if len(common_cores) == 0:
    raise RuntimeError("没有找到共有图像，请检查路径和文件。")
image_ids = common_cores

def find_file(dir_path, core_id):
    for f in os.listdir(dir_path):
        if core_id in f and f.endswith('.npy'):
            return os.path.join(dir_path, f)
    raise FileNotFoundError(f"在 {dir_path} 中找不到包含 {core_id} 的 .npy 文件")

# ========== 辅助函数 ==========
def get_boundary_distance_mask(binary_mask):
    return distance_transform_edt(binary_mask)

def get_connected_components(binary_mask):
    s = generate_binary_structure(2, 2)
    labeled, num = label(binary_mask, structure=s)
    sizes = ndimage.sum(binary_mask, labeled, range(1, num+1))
    return labeled, sizes

# ========== 初始化数据收集 ==========
# 我们将收集所有像素的：区域标签 (0-3)，四个尺度权重，预测熵，预测方差
region_labels = []
weights_all = []   # 每个像素的4个权重，形状 (N,4)
entropy_all = []
variance_all = []

# ========== 逐图像处理 ==========
for core_id in tqdm(image_ids, desc="处理图像"):
    # 加载数据
    gt = np.load(find_file(GT_MASK_DIR, core_id)).astype(np.uint8)
    pred = np.load(find_file(PRED_MASK_DIR, core_id)).astype(np.uint8)
    gates = np.load(find_file(GATES_MEAN_DIR, core_id))
    entropy = np.load(find_file(ENTROPY_DIR, core_id)).astype(np.float32)
    variance = np.load(find_file(VARIANCE_DIR, core_id)).astype(np.float32)

    # 处理gates：转置并上采样
    if gates.ndim == 3 and gates.shape[0] == 4:
        gates = gates.transpose(1, 2, 0)  # (H, W, 4)
    else:
        raise ValueError(f"gates格式错误，应为4×H×W，实际{gates.shape}")
    if gates.shape[:2] != gt.shape:
        gates = cv2.resize(gates, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_LINEAR)
        gates = gates.astype(np.float32)

    # 生成区域掩膜
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

    region_map = np.full(gt.shape, -1, dtype=np.int8)
    region_map[interior_mask] = region_map_int['interior']
    region_map[border_mask]   = region_map_int['border']
    region_map[small_slide_mask] = region_map_int['small_slide']
    region_map[confused_bg_mask] = region_map_int['confused_bg']

    valid_mask = region_map >= 0
    if not np.any(valid_mask):
        continue

    region_labels.append(region_map[valid_mask])
    weights_all.append(gates[valid_mask, :])   # (N,4)
    entropy_all.append(entropy[valid_mask])
    variance_all.append(variance[valid_mask])

# 拼接
region_labels = np.concatenate(region_labels)
weights_all = np.concatenate(weights_all, axis=0)   # (总像素数, 4)
entropy_all = np.concatenate(entropy_all)
variance_all = np.concatenate(variance_all)

print(f"总有效像素数: {len(region_labels)}")
print(f"各区域像素数:")
for rid, rname in enumerate(region_names):
    cnt = np.sum(region_labels == rid)
    print(f"  {rname}: {cnt}")

# ========== 计算每个区域每个尺度权重与熵、方差的相关系数 ==========
results = []
scale_names = ['Scale1', 'Scale2', 'Scale3', 'Scale4']

for rid, rname in enumerate(region_names):
    mask = region_labels == rid
    if np.sum(mask) == 0:
        continue
    w = weights_all[mask]          # (n,4)
    e = entropy_all[mask]
    v = variance_all[mask]
    for s_idx, sname in enumerate(scale_names):
        r_e, p_e = stats.pearsonr(w[:, s_idx], e)
        r_v, p_v = stats.pearsonr(w[:, s_idx], v)
        results.append({
            'Region': rname,
            'Scale': sname,
            'Corr_Entropy': r_e,
            'P_Entropy': p_e,
            'Corr_Variance': r_v,
            'P_Variance': p_v
        })

df = pd.DataFrame(results)
# 按区域和尺度排序
df = df.sort_values(by=['Region', 'Scale']).reset_index(drop=True)

# 保存结果
df.to_csv(os.path.join(OUTPUT_DIR, 'scale_weight_correlations.csv'), index=False)
print(df.to_string())

# ========== 修改绘图：改为1行2列横向并排热力图 ==========
import matplotlib.pyplot as plt
import seaborn as sns


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# 绘制预测熵的热力图（a）
pivot_entropy = df.pivot(index='Region', columns='Scale', values='Corr_Entropy')
sns.heatmap(pivot_entropy, annot=True, cmap='coolwarm', center=0, vmin=-0.5, vmax=0.5,
            cbar_kws={'label': 'Pearson r with Prediction Entropy'}, ax=ax1)
ax1.set_title('(a)', fontsize=12, pad=10)
ax1.set_xlabel('')
ax1.set_ylabel('')

# 绘制预测方差的热力图（b）
pivot_variance = df.pivot(index='Region', columns='Scale', values='Corr_Variance')
sns.heatmap(pivot_variance, annot=True, cmap='coolwarm', center=0, vmin=-0.5, vmax=0.5,
            cbar_kws={'label': 'Pearson r with Prediction Variance'}, ax=ax2)
ax2.set_title('(b)', fontsize=12, pad=10)
ax2.set_xlabel('')
ax2.set_ylabel('')


plt.tight_layout()

# 保存横向拼接图
plt.savefig(os.path.join(OUTPUT_DIR, 'fig5_4_heatmap_horizontal.png'), dpi=300, bbox_inches='tight')
plt.show()

print(f"\n结果已保存至 {OUTPUT_DIR}")