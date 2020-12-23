_base_ = 'faster_rcnn_r50_fpn_1x_laryngoscopy.py'

work_dir = f'./laryngoscopy_output/faster_rcnn_r50_fpn_2x_laryngoscopy/fold1'

lr_config = dict(step=[16, 22])
total_epochs = 24
