_base_ = '../yolo/yolov3_d53_mstrain-608_273e_coco.py'

dataset_type = 'Laryngoscopy'
data_root = 'data/laryngoscopy/'
fold_idx = 1

work_dir = 'yolov3_d53_mstrain-416_273e_laryngoscopy'
train_anns = data_root + f'medical_train_fold{fold_idx}.json'
test_anns = data_root + f'medical_test_fold{fold_idx}.json'

model = dict(bbox_head=dict(num_classes=4))

img_norm_cfg = dict(mean=[0, 0, 0], std=[255., 255., 255.], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='PhotoMetricDistortion'),
    dict(
        type='Expand',
        mean=img_norm_cfg['mean'],
        to_rgb=img_norm_cfg['to_rgb'],
        ratio_range=(1, 2)),
    dict(
        type='MinIoURandomCrop',
        min_ious=(0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
        min_crop_size=0.3),
    dict(type='Resize', img_scale=[(320, 320), (416, 416)], keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(416, 416),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type='RepeatDataset',
        times=10,
        dataset=dict(
            type='Laryngoscopy',
            ann_file=train_anns,
            class_equal=True,
            img_prefix=data_root + '/image',
            pipeline=train_pipeline)),
    val=dict(
        type=dataset_type,
        ann_file=test_anns,
        save_roc_path=work_dir,
        img_prefix=data_root + '/image',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=test_anns,
        img_prefix=data_root + '/image',
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric='recall')
