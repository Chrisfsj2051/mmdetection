_base_ = 'faster_rcnn_r50_fpn_1x_laryngoscopy.py'

work_dir = f'./laryngoscopy_output/faster_rcnn_r50_fro0_fpn_1x_laryngoscopy/fold1'

model = dict(
    backbone=dict(frozen_stages=0, norm_eval=False),
    roi_head=dict(bbox_head=dict(num_classes=4)))
