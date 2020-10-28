_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn.py',
    '../_base_/datasets/laryngoscopy.py',
    '../_base_/schedules/schedule_laryngoscopy.py',
    '../_base_/default_runtime.py'
]

model = dict(
    backbone=dict(
        depth=50,
        norm_eval=False,
        frozen_stages=0,
        plugins=[
            dict(
                cfg=dict(
                    type='DropBlock', drop_prob=0.05, block_size=3, postfix='_1'),
                stages=(False, False, True, True),
                position='after_conv1'),
            dict(
                cfg=dict(
                    type='DropBlock', drop_prob=0.05, block_size=3, postfix='_2'),
                stages=(False, False, True, True),
                position='after_conv2'),
            dict(
                cfg=dict(
                    type='DropBlock', drop_prob=0.05, block_size=3, postfix='_3'),
                stages=(False, False, True, True),
                position='after_conv3')
        ]),
    roi_head = dict(bbox_head=dict(num_classes=5))
)
