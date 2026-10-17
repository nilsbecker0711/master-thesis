r"""
PIDNet-M, Cityscapes — an mmseg **0.x** config for this project's build.

STANDALONE, not `_base_ = './pidnet_s.py'`. The registry's rule is one config
per checkpoint, and a self-contained file can be copied next to the weights on
the cluster without dragging a base along. See configs/mmseg/pidnet_s.py for
why a hand-written 0.x config is needed at all, and for the three
inference-neutral deviations from the zoo config (BN instead of SyncBN, a
single CrossEntropyLoss, no ImageNet init_cfg) — they apply here unchanged.

PIDNet-M IS PIDNet-S WIDENED, nothing more: channels 32 -> 64, and so the
backbone's fused output and the head's in_channels 128 -> 256. num_stem_blocks
is still 2, so it keeps S's PAPPM + LightBag; only PIDNet-L switches to
DAPPM + Bag. Head `channels` stays 128, as in the zoo config — it is NOT
widened with the backbone, and setting it to 256 here would build a head whose
i_head/p_head weights do not match the checkpoint.

WEIGHTS:
  https://download.openmmlab.com/mmsegmentation/v0.5/pidnet/pidnet-m_2xb6-120k_1024x1024-cityscapes/pidnet-m_2xb6-120k_1024x1024-cityscapes_20230301_143452-f9bcdbf3.pth
Published Cityscapes val mIoU 80.22 (82.05 ms+flip), trained at crop 1024x1024.
"""
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
