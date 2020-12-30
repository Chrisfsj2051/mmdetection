_base_ = 'faster_rcnn_r50_fpn_1x_laryngoscopy.py'

work_dir = f'./laryngoscopy_output/faster_rcnn_r50_fpn_2x_100image_laryngoscopy/fold1'

fold_idx = 1
data_root = 'data/laryngoscopy/'
train_anns = data_root + f'100image_train_fold{fold_idx}.json'
test_anns = data_root + f'100image_test_fold{fold_idx}.json'

data = dict(
    samples_per_gpu=2,
    train=dict(normal_thr=0.05, class_equal=True, ann_file=train_anns),
    val=dict(ann_file=test_anns, save_roc_path=work_dir),
    test=dict(ann_file=test_anns))

optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
lr_config = dict(step=[16, 22])
total_epochs = 24
