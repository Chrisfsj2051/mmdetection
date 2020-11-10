_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn_laryngoscopy.py',
    '../_base_/datasets/laryngoscopy.py',
    '../_base_/schedules/schedule_laryngoscopy.py',
    '../_base_/default_runtime.py'
]
data_root = 'data/laryngoscopy/'

fold_idx = 1

work_dir = f'./laryngoscopy_output/faster_rcnn_r50_fpn_1x_laryngoscopy/fold{fold_idx}'
train_anns = data_root + f'medical_train_fold{fold_idx}.json'
test_anns = data_root + f'medical_test_fold{fold_idx}.json'

model = dict(
    backbone=dict(norm_eval=False),
    roi_head=dict(bbox_head=dict(num_classes=4))
)

test_cfg = dict(rcnn=dict(score_thr=0.00))

data=dict(
    workers_per_gpu=2,
    samples_per_gpu=2,
    train=dict(
        ann_file=train_anns
    ),
    val=dict(
        ann_file=test_anns,
        save_roc_path = work_dir
    ),
    test=dict(
        ann_file=test_anns
    )
)
