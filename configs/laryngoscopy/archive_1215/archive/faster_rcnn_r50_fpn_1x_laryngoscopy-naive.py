_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn.py',
    '../_base_/datasets/laryngoscopy-naive.py',
    '../_base_/schedules/schedule_laryngoscopy.py',
    '../_base_/default_runtime.py'
]

model = dict(
    backbone=dict(norm_eval=False),
    roi_head=dict(bbox_head=dict(num_classes=5)))
