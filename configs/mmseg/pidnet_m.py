"""PIDNet-M, Cityscapes — an mmseg **0.x** config for this project's build."""
norm_cfg = dict(type='BN', requires_grad=True)

model = dict(
    type='EncoderDecoder',
    backbone=dict(
        type='PIDNet',
        in_channels=3,
        channels=64,
        ppm_channels=96,
        num_stem_blocks=2,
        num_branch_blocks=3,
        align_corners=False,
        norm_cfg=norm_cfg,
        act_cfg=dict(type='ReLU', inplace=True)),
    decode_head=dict(
        type='PIDHead',
        in_channels=256,
        channels=128,
        num_classes=19,
        norm_cfg=norm_cfg,
        act_cfg=dict(type='ReLU', inplace=True),
        align_corners=True,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    test_cfg=dict(mode='whole'))
