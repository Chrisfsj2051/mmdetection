_base_ = 'faster_rcnn_r50_fpn_2x_100image_laryngoscopy.py'

work_dir = f'./laryngoscopy_output/faster_rcnn_r50_fro-1_fpn_2x_100image_laryngoscopy/fold1'

model = dict(
    backbone=dict(frozen_stages=1, norm_eval=False),
    roi_head=dict(bbox_head=dict(num_classes=4)))

lr_config = dict(step=[16, 22])
total_epochs = 24
