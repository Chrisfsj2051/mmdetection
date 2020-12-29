_base_ = 'faster_rcnn_r50_fpn_2x_100image_laryngoscopy.py'

work_dir = f'./laryngoscopy_output/faster_rcnn_r50_fro0_fpn_3x_100image_laryngoscopy/fold1'

model = dict(
    backbone=dict(frozen_stages=0, norm_eval=False),
    roi_head=dict(bbox_head=dict(num_classes=4)))

lr_config = dict(step=[24, 33])
total_epochs = 36
