_base_ = 'faster_rcnn_r50_fpn_1x_laryngoscopy.py'

model = dict(
    backbone=dict(
        norm_eval=False,
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
        ]
    )
)
