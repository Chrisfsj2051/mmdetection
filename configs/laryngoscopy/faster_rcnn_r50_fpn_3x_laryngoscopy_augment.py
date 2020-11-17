_base_ = 'faster_rcnn_r50_fpn_1x_laryngoscopy.py'


work_dir = f'./laryngoscopy_output/faster_rcnn_r50_fpn_3x_laryngoscopy_augment/fold1'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
replace = (104, 116, 124)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Rotate', prob=0.5, level=5),
    dict(type='Shear', prob=0.5, level=5),
    dict(type='Resize', img_scale=(640, 640), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]

data = dict(
    train=dict(
        pipeline=train_pipeline
    ),
)

lr_config = dict(step=[24, 33])
total_epochs = 36