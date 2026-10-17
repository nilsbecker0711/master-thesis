r"""
PIDNet-S, Cityscapes — an mmseg **0.x** config for this project's build.

NOT the zoo config. configs/pidnet/pidnet-s_2xb6-120k_1024x1024-cityscapes.py
in mmsegmentation is a 1.x config: `data_preprocessor`, `optim_wrapper`,
`param_scheduler`, `train_dataloader`, `PackSegInputs`, `GenerateEdge` — none
of which mmcv 1.x's Config + mmseg 0.x's build_segmentor understand. This file
declares the same ARCHITECTURE in 0.x form and nothing else, which is all
patchreach.models.registry.load_segmentor reads.

ARCHITECTURE IS IDENTICAL TO THE ZOO CONFIG, so the published checkpoint loads
with no key remapping:
    backbone channels=32  ppm_channels=96  num_stem_blocks=2  num_branch_blocks=3
    head     in_channels=128  channels=128  num_classes=19  align_corners=True
`PIDNet` and `PIDHead` are registered by patchreach/models/pidnet.py, which
load_segmentor imports when it sees backbone type='PIDNet' — the same gate
InternImage goes through.

WEIGHTS (download next to this config's registry entry, _M in registry.py):
  https://download.openmmlab.com/mmsegmentation/v0.5/pidnet/pidnet-s_2xb6-120k_1024x1024-cityscapes/pidnet-s_2xb6-120k_1024x1024-cityscapes_20230302_191700-bb8e3bcc.pth
Published Cityscapes val mIoU 78.74 (80.87 ms+flip), trained at crop 1024x1024.

THREE DELIBERATE DEVIATIONS FROM THE ZOO CONFIG, all inference-neutral:

  * norm_cfg is BN, not SyncBN. Identical state_dict keys and identical
    numerics in eval (both read running stats), and it removes any dependence
    on a process group for the single-GPU runs this project does.
  * loss_decode is one plain CrossEntropyLoss, not the zoo's four-term
    [CE, OhemCrossEntropy, BoundaryLoss, OhemCrossEntropy]. OhemCrossEntropy
    and BoundaryLoss do not exist in mmseg 0.x, and build_segmentor builds
    loss_decode eagerly, so an unported type would fail at BUILD time even for
    pure inference. Nothing trains through this head — PIDHead.forward_train
    raises — so the choice cannot affect a number.
  * no `init_cfg` ImageNet pretrain on the backbone. load_segmentor calls
    load_checkpoint with the full Cityscapes weights straight after building,
    which would overwrite it.

RESOLUTION IS NOT PART OF THIS FILE, as for every other entry in the registry:
1024x1024 is the zoo TRAINING crop; input size comes from the dataloader via
--img_h/--img_w. PIDNet's backbone hard-wires its output grid to input//8, so
both axes must be multiples of 8 — 512x1024 and the native 1024x2048 are.

test_cfg mode='whole', which is the zoo's own choice, so `--inference auto`
gives one forward here (unlike segformer/setr, which declare 'slide').

Input normalisation is the ImageNet mean/std the zoo config uses
(mean 123.675/116.28/103.53, std 58.395/57.12/57.375 on 0-255 RGB), which is
patchreach.data.cityscapes.IMG_MEAN/IMG_STD * 255 — already what the
dataloader applies, so there is nothing to declare here.
"""
norm_cfg = dict(type='BN', requires_grad=True)

model = dict(
    type='EncoderDecoder',
    backbone=dict(
        type='PIDNet',
        in_channels=3,
        channels=32,
        ppm_channels=96,
        num_stem_blocks=2,
        num_branch_blocks=3,
        align_corners=False,
        norm_cfg=norm_cfg,
        act_cfg=dict(type='ReLU', inplace=True)),
    decode_head=dict(
        type='PIDHead',
        in_channels=128,
        channels=128,
        num_classes=19,
        norm_cfg=norm_cfg,
        act_cfg=dict(type='ReLU', inplace=True),
        align_corners=True,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    test_cfg=dict(mode='whole'))
