_base_ = [
    # '../_base_/models/faster_rcnn_r50_fpn.py',
    '../_base_/models/faster_rcnn_r50_fpn_laryngoscopy.py',
    '../_base_/datasets/laryngoscopy.py',
    '../_base_/schedules/schedule_laryngoscopy.py',
    '../_base_/default_runtime.py'
]

model = dict(
    backbone=dict(norm_eval=False),
    roi_head=dict(bbox_head=dict(num_classes=5))
)
# data=dict(workers_per_gpu=0)
lr_config = dict(step=[16, 22])
total_epochs = 24

test_cfg = dict(rcnn=dict(score_thr=0.3))
