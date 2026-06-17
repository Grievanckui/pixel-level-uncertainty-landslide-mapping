from hooks.auto_curve_hook import AutoCurveSaveHook
from hooks.best_metric_print_hook import BestMetricPrintHook
from hooks.mmseg_custom_vis_hook import MyOverlayVisHook


# 硬编码绝对工作目录
WORK_DIR = r'E:\MMseg\BIYELUNWEN\work_dirs\segformer_landslide'
VIS_SAVE_DIR = f'{WORK_DIR}/vis_results'
DATA_ROOT = r'E:\MMseg\BIYELUNWEN\data'


CROP_SIZE = (224, 224)
IMG_SCALE = (224, 224)
BATCH_SIZE_TRAIN = 2
BATCH_SIZE_VAL_TEST = 1
MAX_ITERS = 100000
LR = 6e-4
WARMUP_ITERS = 500
CLASS_WEIGHTS = [0.4, 0.6]


# ================================= 数据预处理配置 =================================
data_preprocessor = dict(
    bgr_to_rgb=True,
    mean=[123.675, 116.28, 103.53],
    pad_val=0,
    seg_pad_val=0,
    size=CROP_SIZE,
    std=[58.395, 57.12, 57.375],
    test_cfg=dict(size_divisor=32),
    type='SegDataPreProcessor')

# ================================= 模型配置 =================================
model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    init_cfg=dict(
        type='Pretrained',
        checkpoint='https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b1_20220624-02e5a6a1.pth'
    ),
    backbone=dict(
        type='MixVisionTransformer',
        in_channels=3,
        embed_dims=64,
        num_stages=4,
        num_layers=[2, 2, 2, 2],
        num_heads=[1, 2, 5, 8],
        patch_sizes=[7, 3, 3, 3],
        sr_ratios=[8, 4, 2, 1],
        out_indices=(0, 1, 2, 3),
        mlp_ratio=4,
        qkv_bias=True,
        drop_rate=0,
        attn_drop_rate=0,
        drop_path_rate=0.1),
    decode_head=dict(
        type='SegformerHead',
        in_channels=[64, 128, 320, 512],
        in_index=[0, 1, 2, 3],
        channels=256,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=dict(type='BN', requires_grad=True),
        align_corners=True,
        # threshold=0.5,
        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=1.0,
            class_weight=[1.0, 2.0],
            avg_non_ignore=True)),
    auxiliary_head=None,
    train_cfg=dict(),
    test_cfg=dict(
        mode='whole',
        # threshold=0.5
    ))

norm_cfg = dict(type='BN', requires_grad=True)

# ================================= 优化器配置 =================================
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=LR,
        betas=(0.9, 0.999),
        weight_decay=5e-4),
    clip_grad=dict(max_norm=35, norm_type=2),
)

# ================================= 学习率调度器 =================================
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1e-6,
        by_epoch=False,
        begin=0,
        end=WARMUP_ITERS),
    dict(
        type='PolyLR',
        eta_min=LR * 0.01,
        power=0.9,
        begin=WARMUP_ITERS,
        end=MAX_ITERS,
        by_epoch=False)
]

# ================================= 默认钩子 =================================
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=False,
        interval=1000,
        max_keep_ckpts=2,
        save_best='mIoU',
        save_last=False),
    logger=dict(
        type='LoggerHook',
        interval=500,
        log_metric_by_epoch=False,
        ignore_last=True),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
)

# ================================= 自定义钩子（曲线保存） =================================
custom_hooks = [
    dict(type=MyOverlayVisHook,
         vis_enabled=True,
         num_samples=3),
    dict(
        type=AutoCurveSaveHook,
        # vis_save_root=VIS_SAVE_DIR,
        interval=500,
        metric_names=['Loss', 'mIoU', 'Recall', 'Precision', 'F1']
    ),
# 最优指标打印Hook
    dict(
        type=BestMetricPrintHook,
        metric_names=['mIoU', 'mPrecision', 'mRecall', 'mFscore']  # 与AutoCurveSaveHook保持一致
    )
]

# ================================= 数据管道 =================================
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(
        type='RandomResize',
        scale=IMG_SCALE,
        ratio_range=(0.75, 1.25),
        keep_ratio=True),
    dict(
        type='RandomCrop',
        crop_size=CROP_SIZE,
        cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='RandomRotate',
        prob=0.5,
        degree=10,
        pad_val=0,
        seg_pad_val=0,
    ),
    dict(
        type='PhotoMetricDistortion',
        brightness_delta=16,
        contrast_range=(0.8, 1.2),
        saturation_range=(0.8, 1.2),
        hue_delta=9),
    dict(type='PackSegInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=IMG_SCALE, keep_ratio=True),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs', meta_keys=('img_path', 'seg_map_path', 'ori_shape', 'img_shape'))
]

# ================================= 数据集配置 =================================
dataset_type = 'MyLandslideDataset'

train_dataloader = dict(
    batch_size=BATCH_SIZE_TRAIN,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=DATA_ROOT,
        data_prefix=dict(
            img_path='train/vis_images',
            seg_map_path='train/labels'),
        pipeline=train_pipeline,
        label_map={0: 0, 255: 1},
        serialize_data=True),
    pin_memory=False)

val_dataloader = dict(
    batch_size=BATCH_SIZE_VAL_TEST,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=DATA_ROOT,
        data_prefix=dict(
            img_path='val/vis_images',
            seg_map_path='val/labels'),
        pipeline=test_pipeline,
        label_map={0: 0, 255: 1}),
    pin_memory=False)

# test_dataloader = val_dataloader
# ========== 测试集 独立配置  ==========
test_dataloader = dict(
    batch_size=BATCH_SIZE_VAL_TEST,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=DATA_ROOT,
        data_prefix=dict(
            img_path='test/vis_images',
            seg_map_path='test/labels'),
        pipeline=test_pipeline,
        label_map={0: 0, 255: 1}),
    pin_memory=False)


# ================================= 评估器配置 =================================
val_evaluator = dict(
    type='IoUMetric',
    iou_metrics=['mIoU', 'mDice', 'mFscore'])

test_evaluator = val_evaluator
# test_evaluator = dict(
#     type='IoUMetric',
#     metric=['mIoU', 'mAcc', 'aAcc', 'mDice', 'mFscore', 'mPrecision', 'mRecall'],
#     output_dir=r'E:\MMseg\BIYELUNWEN\work_dirs\segformer_landslide\test_results',
#     keep_results=True
# )
# ================================= 训练/验证配置 =================================
train_cfg = dict(
    type='IterBasedTrainLoop',
    max_iters=MAX_ITERS,
    val_interval=500
)

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# ================================= TTA配置 =================================
img_ratios = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
tta_model = dict(type='SegTTAModel')
tta_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='TestTimeAug',
        transforms=[
            [
                dict(type='Resize', scale_factor=r, keep_ratio=True)
                for r in img_ratios
            ],
            [
                dict(type='RandomFlip', prob=0.0, direction='horizontal'),
                dict(type='RandomFlip', prob=1.0, direction='horizontal'),
            ],
            [dict(type='LoadAnnotations')],
            [dict(type='PackSegInputs')],
        ])
]

# ================================= 可视化配置 =================================
PALETTE = [[0, 0, 0], [200, 0, 0]]
CLASS_NAMES = ['background', 'landslide']

vis_backends = [
    dict(
        type='LocalVisBackend',
        save_dir=VIS_SAVE_DIR
    ),
]
visualizer = dict(
    type='SegLocalVisualizer',
    vis_backends=vis_backends,
    name='visualizer',
    palette=PALETTE,
    alpha=1.0,
    classes=CLASS_NAMES
)

# ================================= 环境配置 =================================
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))

# ================================= 其他配置 =================================
randomness = dict(seed=42, deterministic=False)
log_processor = dict(by_epoch=False)
log_level = 'INFO'
default_scope = 'mmseg'
load_from = None
resume = False



