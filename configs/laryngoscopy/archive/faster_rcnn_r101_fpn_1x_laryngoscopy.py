_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn.py',
    '../_base_/datasets/laryngoscopy.py',
    '../_base_/schedules/schedule_laryngoscopy.py',
    '../_base_/default_runtime.py'
]

model = dict(
    pretrained='torchvision://resnet101',
    backbone=dict(depth=101, norm_eval=False),
    roi_head=dict(bbox_head=dict(num_classes=5))
)
