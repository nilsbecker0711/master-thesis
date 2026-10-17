r"""
PIDNet-L, Cityscapes — an mmseg **0.x** config for this project's build.

THE VARIANT THE LITERATURE COMPARISON NEEDS. Maag & Fischer (arXiv:2408.10021)
Table 1 reports "PIDNet 80.89" on Cityscapes val, which is PIDNet-**L**'s zoo
number exactly (S is 78.74, M is 80.22). docs/experiment_log.md leans on that
paper for the 512x1024 protocol, so `pidnet` aliases to this entry, not to S.

STANDALONE, not `_base_ = './pidnet_s.py'` — see configs/mmseg/pidnet_m.py.
configs/mmseg/pidnet_s.py carries the full rationale for a hand-written 0.x
config and the three inference-neutral deviations from the zoo config (BN
instead of SyncBN, a single CrossEntropyLoss, no ImageNet init_cfg).

L IS A DIFFERENT GRAPH, NOT JUST WIDER. num_stem_blocks=3 flips the backbone
onto the heavy D branch: DAPPM instead of PAPPM for the pyramid pooling, Bag
instead of LightBag for the boundary fusion, and two BasicBlock single-layers
instead of BasicBlock + Bottleneck. ppm_channels also rises to 112 and
num_branch_blocks to 4. Getting num_stem_blocks wrong here does not mis-size
the weights, it builds the wrong modules — see the D-branch comment in
patchreach/models/pidnet.py. Head channels is 256 for L, where S and M use 128.

WEIGHTS:
  https://download.openmmlab.com/mmsegmentation/v0.5/pidnet/pidnet-l_2xb6-120k_1024x1024-cityscapes/pidnet-l_2xb6-120k_1024x1024-cityscapes_20230303_114514-0783ca6b.pth
Published Cityscapes val mIoU 80.89 (82.37 ms+flip), trained at crop 1024x1024.
"""
norm_cfg = dict(type='BN', requires_grad=True)

model = dict(
    type='EncoderDecoder',
    backbone=dict(
        type='PIDNet',
        in_channels=3,
        channels=64,
        ppm_channels=112,
        num_stem_blocks=3,
        num_branch_blocks=4,
        align_corners=False,
        norm_cfg=norm_cfg,
        act_cfg=dict(type='ReLU', inplace=True)),
    decode_head=dict(
        type='PIDHead',
        in_channels=256,
        channels=256,
        num_classes=19,
        norm_cfg=norm_cfg,
        act_cfg=dict(type='ReLU', inplace=True),
        align_corners=True,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    test_cfg=dict(mode='whole'))
