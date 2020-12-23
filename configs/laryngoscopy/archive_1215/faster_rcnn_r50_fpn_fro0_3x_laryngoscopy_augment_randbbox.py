_base_ = 'faster_rcnn_r50_fpn_1x_laryngoscopy.py'

work_dir = f'./laryngoscopy_output/faster_rcnn_r50_fro0_fpn_3x_laryngoscopy_augment_randbbox/fold1'

model = dict(
    type='LaryngoscopyFasterRCNN',
    backbone=dict(
        norm_eval=False,
        frozen_stages=0,
        plugins=[
            dict(
                cfg=dict(
                    type='MMDETDropBlock',
                    drop_prob=0.1,
                    block_size=7,
                    warmup_iters=200,
                    postfix='_1'),
                stages=(False, False, True, True),
                position='after_conv1'),
            dict(
                cfg=dict(
                    type='MMDETDropBlock',
                    drop_prob=0.1,
                    block_size=7,
                    warmup_iters=200,
                    postfix='_2'),
                stages=(False, False, True, True),
                position='after_conv2'),
            dict(
                cfg=dict(
                    type='MMDETDropBlock',
                    drop_prob=0.1,
                    block_size=7,
                    warmup_iters=200,
                    postfix='_3'),
                stages=(False, False, True, True),
                position='after_conv3')
        ]),
)

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
replace = (104, 116, 124)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='MMDETRandomAugmentBBox'),
    dict(type='Resize', img_scale=(640, 640), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5, direction='horizontal'),
    dict(type='RandomFlip', flip_ratio=0.5, direction='vertical'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]

data = dict(train=dict(pipeline=train_pipeline), )

lr_config = dict(step=[24, 33])
total_epochs = 36
