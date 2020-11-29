_base_ = 'faster_rcnn_r50_fpn_1x_laryngoscopy.py'

work_dir = f'./laryngoscopy_output/faster_rcnn_regnet3.2GF_fpn_1x_laryngoscopy/fold1'

model = dict(
    pretrained='open-mmlab://regnetx_3.2gf',
    backbone=dict(
        _delete_=True,
        type='RegNet',
        arch='regnetx_3.2gf',
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_eval=False,
        with_cp=True,
        norm_cfg=dict(type='BN', requires_grad=True),
        style='pytorch',
        plugins=[
            dict(
                cfg=dict(
                    type='MMDETDropBlock',
                    drop_prob=0.05,
                    block_size=7,
                    warmup_iters=200,
                    postfix='_1'),
                stages=(False, False, True, True),
                position='after_conv1'),
            dict(
                cfg=dict(
                    type='MMDETDropBlock',
                    drop_prob=0.05,
                    block_size=7,
                    warmup_iters=200,
                    postfix='_2'),
                stages=(False, False, True, True),
                position='after_conv2'),
            dict(
                cfg=dict(
                    type='MMDETDropBlock',
                    drop_prob=0.05,
                    block_size=7,
                    warmup_iters=200,
                    postfix='_3'),
                stages=(False, False, True, True),
                position='after_conv3')
        ]
    ),
    neck=dict(
        type='FPN',
        in_channels=[96, 192, 432, 1008],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_lateral',
        upsample_cfg=dict(mode='bilinear', align_corners=True),
        num_outs=5),
)

optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.00005)

lr_config = dict(step=[16, 22])
total_epochs = 24
