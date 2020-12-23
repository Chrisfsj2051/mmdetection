_base_ = 'faster_rcnn_r50_fpn_1x_laryngoscopy.py'

work_dir = f'./laryngoscopy_output/faster_rcnn_r50_fpn_2x_100image_laryngoscopy/fold1'

fold_idx = 2
data_root = 'data/laryngoscopy/'
train_anns = data_root + f'100_test_image_train_fold{fold_idx}.json'
test_anns = data_root + f'100_test_image_test_fold{fold_idx}.json'

data = dict(
    train=dict(ann_file=train_anns),
    val=dict(ann_file=test_anns, save_roc_path=work_dir),
    test=dict(ann_file=test_anns))

lr_config = dict(step=[16, 22])
total_epochs = 24
