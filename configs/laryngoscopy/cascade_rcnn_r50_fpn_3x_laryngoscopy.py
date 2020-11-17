_base_ = 'cascade_rcnn_r50_fpn_1x_laryngoscopy.py'

work_dir = f'./laryngoscopy_output/cascade_rcnn_r50_fpn_3x_laryngoscopy/fold1'

lr_config = dict(step=[24, 33])
total_epochs = 36

