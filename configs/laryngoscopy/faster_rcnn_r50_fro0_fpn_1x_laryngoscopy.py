_base_ = 'faster_rcnn_r50_fpn_1x_laryngoscopy.py'

lr_config = dict(step=[16, 22])
total_epochs = 24

model = dict(
    backbone=dict(
        frozen_stages=0,
        norm_eval=False
    ),
    roi_head=dict(bbox_head=dict(num_classes=4))
)
