_base_ = 'faster_rcnn_r50_fpn_1x_laryngoscopy.py'

work_dir = f'./laryngoscopy_output/faster_rcnn_r50_dropblock_fpn_2x_laryngoscopy/fold1'

model = dict(
    type='LaryngoscopyFasterRCNN',
    backbone=dict(
        norm_eval=False,
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
        ]),
)

lr_config = dict(step=[16, 22])
total_epochs = 24
