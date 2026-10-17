r"""
PIDNet for mmseg 0.x — backbone, decode head and the blocks they need.

WHY THIS FILE EXISTS AT ALL. PIDNet landed in mmsegmentation **1.0.0**
("Support PIDNet (#2609)", 04/06/2023). This project is pinned to torch 1.11 +
mmcv 1.x + mmseg 0.x (see pyproject.toml), where `mmseg.models.backbones` has
no PIDNet and `mmseg.models.decode_heads` has no PIDHead. Pointing the registry
at the zoo config would raise "PIDNet is not in the models registry" three
frames inside build_segmentor. Upgrading mmseg is not an option: 1.x moves to
mmengine, replaces `Config.fromfile`/`build_segmentor`, and drops the
`encode_decode(img, img_metas)` entry point that patchreach.models.wrapper and
~30 call sites are built on. InternImage is already handled by vendoring a
backbone (mmseg_custom), so vendoring is the established answer here.

PORTED FROM mmsegmentation v1.2.2, MIT licence:
    mmseg/models/backbones/pidnet.py      (PagFM, Bag, LightBag, PIDNet)
    mmseg/models/decode_heads/pid_head.py (BasePIDHead, PIDHead)
    mmseg/models/utils/basic_block.py     (BasicBlock, Bottleneck)
    mmseg/models/utils/ppm.py             (DAPPM, PAPPM)
which is itself ported from https://github.com/XuJiacong/PIDNet (MIT).

THE PORT IS DELIBERATELY MECHANICAL. Module names, attribute names and nesting
are the upstream ones, because that is what makes the state_dict keys of the
published zoo checkpoint load with no remapping. Only the imports and the
registration change:

    mmengine.model.BaseModule/ModuleList/Sequential -> mmcv.runner.*
    MODELS.register_module()                        -> BACKBONES / HEADS
    MODELS.build(act_cfg)                           -> build_activation_layer
    mmengine.runner.CheckpointLoader                -> dropped, see init_weights

TWO THINGS ARE *NOT* PORTED, both training-only:

  * `PIDHead.loss_by_feat` and `_stack_batch_gt`. They need a boundary GT map,
    produced by the 1.x-only `GenerateEdge` pipeline transform, plus the
    1.x-only `OhemCrossEntropy` and `BoundaryLoss`. `forward_train` therefore
    raises instead of silently optimising the wrong objective — this project
    never trains a segmentor (scripts/train.py trains a PATCH; load_segmentor
    sets `cfg.model.train_cfg = None` and the wrapper calls `.eval()`).
  * the ImageNet `init_cfg=dict(type='Pretrained', ...)` download. The registry
    loads the full Cityscapes checkpoint through `load_checkpoint`, so an
    ImageNet backbone init would be overwritten a line later anyway.

INFERENCE PATH, which is the only path this project uses. In eval mode the
backbone returns ONE tensor (the fused P/I/D feature at stride 8) rather than
the training triple, and PIDHead.forward routes it through the integral head
alone. Both are plain `self.training` branches, inherited unchanged from
upstream, so `EncoderDecoder.encode_decode` -> `extract_feat` ->
`_decode_head_forward_test` works with no special-casing and
`WrappedSegModel`/`SlidingWindowSegModel` need no per-arch branch.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, build_activation_layer, build_norm_layer
from mmcv.runner import BaseModule, ModuleList, Sequential
from torch import Tensor

from mmseg.models.builder import BACKBONES, HEADS
from mmseg.models.decode_heads.decode_head import BaseDecodeHead


# ─── residual blocks ────────────────────────────────────────────────────────
# Upstream mmseg/models/utils/basic_block.py. NOTE Bottleneck.expansion is 2
# here, not ResNet's 4 — PIDNet's channel arithmetic depends on it, and a 4
# would build a model whose state_dict silently does not match the checkpoint.

class BasicBlock(BaseModule):
    """ResNet basic block. `act_cfg_out=None` leaves the block unactivated."""

    expansion = 1

    def __init__(self,
                 in_channels: int,
                 channels: int,
                 stride: int = 1,
                 downsample: Optional[nn.Module] = None,
                 norm_cfg: Optional[Dict] = dict(type='BN'),
                 act_cfg: Optional[Dict] = dict(type='ReLU', inplace=True),
                 act_cfg_out: Optional[Dict] = dict(type='ReLU', inplace=True),
                 init_cfg: Optional[Dict] = None):
        super().__init__(init_cfg)
        self.conv1 = ConvModule(
            in_channels,
            channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.conv2 = ConvModule(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            norm_cfg=norm_cfg,
            act_cfg=None)
        self.downsample = downsample
        if act_cfg_out:
            self.act = build_activation_layer(act_cfg_out)

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)

        if self.downsample:
            residual = self.downsample(x)

        # Out-of-place, where upstream writes `out += residual`. The in-place
        # form is autograd-legal (add's backward needs neither operand, and
        # BatchNorm's needs its input, not its output) but this is the one
        # codebase where every forward exists to be back-propagated through,
        # often under torch.utils.checkpoint recomputation, so the version
        # counter is not worth the one saved buffer.
        out = out + residual

        if hasattr(self, 'act'):
            out = self.act(out)

        return out


class Bottleneck(BaseModule):
    """ResNet bottleneck block. expansion 2, as in PIDNet — not ResNet's 4."""

    expansion = 2

    def __init__(self,
                 in_channels: int,
                 channels: int,
                 stride: int = 1,
                 downsample: Optional[nn.Module] = None,
                 norm_cfg: Optional[Dict] = dict(type='BN'),
                 act_cfg: Optional[Dict] = dict(type='ReLU', inplace=True),
                 act_cfg_out: Optional[Dict] = None,
                 init_cfg: Optional[Dict] = None):
        super().__init__(init_cfg)
        self.conv1 = ConvModule(
            in_channels, channels, 1, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.conv2 = ConvModule(
            channels,
            channels,
            3,
            stride,
            1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.conv3 = ConvModule(
            channels,
            channels * self.expansion,
            1,
            norm_cfg=norm_cfg,
            act_cfg=None)
        if act_cfg_out:
            self.act = build_activation_layer(act_cfg_out)
        self.downsample = downsample

    def forward(self, x: Tensor) -> Tensor:
        residual = x

        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)

        if self.downsample:
            residual = self.downsample(x)

        out = out + residual

        if hasattr(self, 'act'):
            out = self.act(out)

        return out


# ─── pyramid pooling ────────────────────────────────────────────────────────
# Upstream mmseg/models/utils/ppm.py. DAPPM is DDRNet's; PAPPM is PIDNet's
# parallel variant and REPLACES the inherited `processes` ModuleList with one
# grouped ConvModule, which is why the two have different state_dict shapes
# under `spp.processes.*`. PIDNet-S/M use PAPPM, PIDNet-L uses DAPPM.

class DAPPM(BaseModule):
    """Deep Aggregation Pyramid Pooling Module (DDRNet)."""

    def __init__(self,
                 in_channels: int,
                 branch_channels: int,
                 out_channels: int,
                 num_scales: int,
                 kernel_sizes: List[int] = [5, 9, 17],
                 strides: List[int] = [2, 4, 8],
                 paddings: List[int] = [2, 4, 8],
                 norm_cfg: Dict = dict(type='BN', momentum=0.1),
                 act_cfg: Dict = dict(type='ReLU', inplace=True),
                 conv_cfg: Dict = dict(
                     order=('norm', 'act', 'conv'), bias=False),
                 upsample_mode: str = 'bilinear'):
        super().__init__()

        self.num_scales = num_scales
        self.unsample_mode = upsample_mode     # upstream's spelling; kept
        self.in_channels = in_channels
        self.branch_channels = branch_channels
        self.out_channels = out_channels
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.conv_cfg = conv_cfg

        self.scales = ModuleList([
            ConvModule(
                in_channels,
                branch_channels,
                kernel_size=1,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                **conv_cfg)
        ])
        for i in range(1, num_scales - 1):
            self.scales.append(
                Sequential(*[
                    nn.AvgPool2d(
                        kernel_size=kernel_sizes[i - 1],
                        stride=strides[i - 1],
                        padding=paddings[i - 1]),
                    ConvModule(
                        in_channels,
                        branch_channels,
                        kernel_size=1,
                        norm_cfg=norm_cfg,
                        act_cfg=act_cfg,
                        **conv_cfg)
                ]))
        self.scales.append(
            Sequential(*[
                nn.AdaptiveAvgPool2d((1, 1)),
                ConvModule(
                    in_channels,
                    branch_channels,
                    kernel_size=1,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    **conv_cfg)
            ]))
        self.processes = ModuleList()
        for i in range(num_scales - 1):
            self.processes.append(
                ConvModule(
                    branch_channels,
                    branch_channels,
                    kernel_size=3,
                    padding=1,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    **conv_cfg))

        self.compression = ConvModule(
            branch_channels * num_scales,
            out_channels,
            kernel_size=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            **conv_cfg)

        self.shortcut = ConvModule(
            in_channels,
            out_channels,
            kernel_size=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            **conv_cfg)

    def forward(self, inputs: Tensor) -> Tensor:
        feats = [self.scales[0](inputs)]

        for i in range(1, self.num_scales):
            feat_up = F.interpolate(
                self.scales[i](inputs),
                size=inputs.shape[2:],
                mode=self.unsample_mode)
            feats.append(self.processes[i - 1](feat_up + feats[i - 1]))

        return self.compression(torch.cat(feats,
                                          dim=1)) + self.shortcut(inputs)


class PAPPM(DAPPM):
    """Parallel Aggregation Pyramid Pooling Module (PIDNet)."""

    def __init__(self,
                 in_channels: int,
                 branch_channels: int,
                 out_channels: int,
                 num_scales: int,
                 kernel_sizes: List[int] = [5, 9, 17],
                 strides: List[int] = [2, 4, 8],
                 paddings: List[int] = [2, 4, 8],
                 norm_cfg: Dict = dict(type='BN', momentum=0.1),
                 act_cfg: Dict = dict(type='ReLU', inplace=True),
                 conv_cfg: Dict = dict(
                     order=('norm', 'act', 'conv'), bias=False),
                 upsample_mode: str = 'bilinear'):
        super().__init__(in_channels, branch_channels, out_channels,
                         num_scales, kernel_sizes, strides, paddings, norm_cfg,
                         act_cfg, conv_cfg, upsample_mode)

        # Overwrites DAPPM's ModuleList with a single grouped conv: all scales
        # are processed in parallel instead of cascaded. Reassignment (not a
        # separate name) is upstream's, and it is what keeps the checkpoint key
        # `spp.processes.conv.weight` — no `.0.` index — matching.
        self.processes = ConvModule(
            self.branch_channels * (self.num_scales - 1),
            self.branch_channels * (self.num_scales - 1),
            kernel_size=3,
            padding=1,
            groups=self.num_scales - 1,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
            **self.conv_cfg)

    def forward(self, inputs: Tensor) -> Tensor:
        x_ = self.scales[0](inputs)
        feats = []
        for i in range(1, self.num_scales):
            feat_up = F.interpolate(
                self.scales[i](inputs),
                size=inputs.shape[2:],
                mode=self.unsample_mode,
                align_corners=False)
            feats.append(feat_up + x_)
        scale_out = self.processes(torch.cat(feats, dim=1))
        return self.compression(torch.cat([x_, scale_out],
                                          dim=1)) + self.shortcut(inputs)


# ─── fusion modules ─────────────────────────────────────────────────────────

class PagFM(BaseModule):
    """Pixel-attention-guided fusion: gate the I branch into the P branch."""

    def __init__(self,
                 in_channels: int,
                 channels: int,
                 after_relu: bool = False,
                 with_channel: bool = False,
                 upsample_mode: str = 'bilinear',
                 norm_cfg: Optional[Dict] = dict(type='BN'),
                 act_cfg: Optional[Dict] = dict(type='ReLU', inplace=True),
                 init_cfg: Optional[Dict] = None):
        super().__init__(init_cfg)
        self.after_relu = after_relu
        self.with_channel = with_channel
        self.upsample_mode = upsample_mode
        self.f_i = ConvModule(
            in_channels, channels, 1, norm_cfg=norm_cfg, act_cfg=None)
        self.f_p = ConvModule(
            in_channels, channels, 1, norm_cfg=norm_cfg, act_cfg=None)
        if with_channel:
            self.up = ConvModule(
                channels, in_channels, 1, norm_cfg=norm_cfg, act_cfg=None)
        if after_relu:
            self.relu = build_activation_layer(act_cfg)

    def forward(self, x_p: Tensor, x_i: Tensor) -> Tensor:
        if self.after_relu:
            x_p = self.relu(x_p)
            x_i = self.relu(x_i)

        f_i = self.f_i(x_i)
        f_i = F.interpolate(
            f_i,
            size=x_p.shape[2:],
            mode=self.upsample_mode,
            align_corners=False)

        f_p = self.f_p(x_p)

        if self.with_channel:
            sigma = torch.sigmoid(self.up(f_p * f_i))
        else:
            sigma = torch.sigmoid(torch.sum(f_p * f_i, dim=1).unsqueeze(1))

        x_i = F.interpolate(
            x_i,
            size=x_p.shape[2:],
            mode=self.upsample_mode,
            align_corners=False)

        return sigma * x_i + (1 - sigma) * x_p


class Bag(BaseModule):
    """Boundary-attention-guided fusion. PIDNet-L."""

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 3,
                 padding: int = 1,
                 norm_cfg: Optional[Dict] = dict(type='BN'),
                 act_cfg: Optional[Dict] = dict(type='ReLU', inplace=True),
                 conv_cfg: Optional[Dict] = dict(
                     order=('norm', 'act', 'conv')),
                 init_cfg: Optional[Dict] = None):
        super().__init__(init_cfg)

        self.conv = ConvModule(
            in_channels,
            out_channels,
            kernel_size,
            padding=padding,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            **conv_cfg)

    def forward(self, x_p: Tensor, x_i: Tensor, x_d: Tensor) -> Tensor:
        sigma = torch.sigmoid(x_d)
        return self.conv(sigma * x_p + (1 - sigma) * x_i)


class LightBag(BaseModule):
    """Light boundary-attention-guided fusion. PIDNet-S/M."""

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 norm_cfg: Optional[Dict] = dict(type='BN'),
                 act_cfg: Optional[Dict] = None,
                 init_cfg: Optional[Dict] = None):
        super().__init__(init_cfg)
        self.f_p = ConvModule(
            in_channels,
            out_channels,
            kernel_size=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.f_i = ConvModule(
            in_channels,
            out_channels,
            kernel_size=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

    def forward(self, x_p: Tensor, x_i: Tensor, x_d: Tensor) -> Tensor:
        sigma = torch.sigmoid(x_d)

        f_p = self.f_p((1 - sigma) * x_i + x_p)
        f_i = self.f_i(x_i + sigma * x_p)

        return f_p + f_i


# ─── backbone ───────────────────────────────────────────────────────────────

@BACKBONES.register_module()
class PIDNet(BaseModule):
    r"""
    PIDNet backbone (arXiv:2206.02066), ported to mmseg 0.x. See module header.

    THREE BRANCHES, fused at stride 8. P (detail, held at stride 8 throughout),
    I (context, strided down to 1/64 and pooled by DAPPM/PAPPM), D (boundary,
    the "derivative" term). PagFM gates I into P twice; Bag/LightBag then fuses
    all three under the boundary gate.

    WHAT THIS MEANS FOR THE ERF BRACKET. There is no self-attention anywhere —
    the receptive field grows only by convolution, striding and the pyramid
    pooling in PAPPM/DAPPM, much as DeepLab's ASPP does. PIDNet therefore
    belongs in the "none" bracket of patchreach.models.registry, beside
    deeplabv3plus and internimage, not with segformer or setr.

    ENTRY STRIDE 2, OUTPUT STRIDE 8. The stem is two stride-2 ConvModules, so
    the P branch runs the network at 1/8 resolution. `w_out`/`h_out` below are
    hard-wired to `// 8`, which is why the input must be a multiple of 8 in
    both axes — 512x1024 and the native 1024x2048 both are.
    """

    def __init__(self,
                 in_channels: int = 3,
                 channels: int = 64,
                 ppm_channels: int = 96,
                 num_stem_blocks: int = 2,
                 num_branch_blocks: int = 3,
                 align_corners: bool = False,
                 norm_cfg: Optional[Dict] = dict(type='BN'),
                 act_cfg: Optional[Dict] = dict(type='ReLU', inplace=True),
                 init_cfg: Optional[Dict] = None,
                 **kwargs):
        super().__init__(init_cfg)
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.align_corners = align_corners

        # stem layer
        self.stem = self._make_stem_layer(in_channels, channels,
                                          num_stem_blocks)
        self.relu = nn.ReLU()

        # I Branch
        self.i_branch_layers = nn.ModuleList()
        for i in range(3):
            self.i_branch_layers.append(
                self._make_layer(
                    block=BasicBlock if i < 2 else Bottleneck,
                    in_channels=channels * 2**(i + 1),
                    channels=channels * 8 if i > 0 else channels * 4,
                    num_blocks=num_branch_blocks if i < 2 else 2,
                    stride=2))

        # P Branch
        self.p_branch_layers = nn.ModuleList()
        for i in range(3):
            self.p_branch_layers.append(
                self._make_layer(
                    block=BasicBlock if i < 2 else Bottleneck,
                    in_channels=channels * 2,
                    channels=channels * 2,
                    num_blocks=num_stem_blocks if i < 2 else 1))
        self.compression_1 = ConvModule(
            channels * 4,
            channels * 2,
            kernel_size=1,
            bias=False,
            norm_cfg=norm_cfg,
            act_cfg=None)
        self.compression_2 = ConvModule(
            channels * 8,
            channels * 2,
            kernel_size=1,
            bias=False,
            norm_cfg=norm_cfg,
            act_cfg=None)
        self.pag_1 = PagFM(channels * 2, channels)
        self.pag_2 = PagFM(channels * 2, channels)

        # D Branch. num_stem_blocks == 2 is PIDNet-S/M, 3 is PIDNet-L, and the
        # two take DIFFERENT module types (PAPPM + LightBag vs DAPPM + Bag), so
        # a config that gets num_stem_blocks wrong does not merely mis-size the
        # weights, it builds a different graph.
        if num_stem_blocks == 2:
            self.d_branch_layers = nn.ModuleList([
                self._make_single_layer(BasicBlock, channels * 2, channels),
                self._make_layer(Bottleneck, channels, channels, 1)
            ])
            channel_expand = 1
            spp_module = PAPPM
            dfm_module = LightBag
            act_cfg_dfm = None
        else:
            self.d_branch_layers = nn.ModuleList([
                self._make_single_layer(BasicBlock, channels * 2,
                                        channels * 2),
                self._make_single_layer(BasicBlock, channels * 2, channels * 2)
            ])
            channel_expand = 2
            spp_module = DAPPM
            dfm_module = Bag
            act_cfg_dfm = act_cfg

        self.diff_1 = ConvModule(
            channels * 4,
            channels * channel_expand,
            kernel_size=3,
            padding=1,
            bias=False,
            norm_cfg=norm_cfg,
            act_cfg=None)
        self.diff_2 = ConvModule(
            channels * 8,
            channels * 2,
            kernel_size=3,
            padding=1,
            bias=False,
            norm_cfg=norm_cfg,
            act_cfg=None)

        self.spp = spp_module(
            channels * 16, ppm_channels, channels * 4, num_scales=5)
        self.dfm = dfm_module(
            channels * 4, channels * 4, norm_cfg=norm_cfg, act_cfg=act_cfg_dfm)

        self.d_branch_layers.append(
            self._make_layer(Bottleneck, channels * 2, channels * 2, 1))

    def _make_stem_layer(self, in_channels: int, channels: int,
                         num_blocks: int) -> nn.Sequential:
        layers = [
            ConvModule(
                in_channels,
                channels,
                kernel_size=3,
                stride=2,
                padding=1,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg),
            ConvModule(
                channels,
                channels,
                kernel_size=3,
                stride=2,
                padding=1,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)
        ]

        layers.append(
            self._make_layer(BasicBlock, channels, channels, num_blocks))
        layers.append(nn.ReLU())
        layers.append(
            self._make_layer(
                BasicBlock, channels, channels * 2, num_blocks, stride=2))
        layers.append(nn.ReLU())

        return nn.Sequential(*layers)

    def _make_layer(self,
                    block,
                    in_channels: int,
                    channels: int,
                    num_blocks: int,
                    stride: int = 1) -> nn.Sequential:
        downsample = None
        if stride != 1 or in_channels != channels * block.expansion:
            downsample = ConvModule(
                in_channels,
                channels * block.expansion,
                kernel_size=1,
                stride=stride,
                norm_cfg=self.norm_cfg,
                act_cfg=None)

        layers = [block(in_channels, channels, stride, downsample)]
        in_channels = channels * block.expansion
        for i in range(1, num_blocks):
            layers.append(
                block(
                    in_channels,
                    channels,
                    stride=1,
                    act_cfg_out=None if i == num_blocks - 1 else self.act_cfg))
        return nn.Sequential(*layers)

    def _make_single_layer(self,
                           block,
                           in_channels: int,
                           channels: int,
                           stride: int = 1) -> nn.Module:
        downsample = None
        if stride != 1 or in_channels != channels * block.expansion:
            downsample = ConvModule(
                in_channels,
                channels * block.expansion,
                kernel_size=1,
                stride=stride,
                norm_cfg=self.norm_cfg,
                act_cfg=None)
        return block(
            in_channels, channels, stride, downsample, act_cfg_out=None)

    def init_weights(self):
        """
        Kaiming init. Upstream also downloads the ImageNet backbone named in
        `init_cfg`; that is dropped here because load_segmentor loads the full
        Cityscapes checkpoint immediately afterwards and would overwrite it.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: Tensor) -> Union[Tensor, Tuple[Tensor, ...]]:
        """
        Eval: ONE tensor, the fused feature at stride 8 — which is what makes
        `EncoderDecoder.extract_feat` -> `PIDHead.forward` work unchanged.
        Train: (P, fused, D) for the three-way loss, kept for completeness
        even though PIDHead.forward_train refuses to use it here.
        """
        w_out = x.shape[-1] // 8
        h_out = x.shape[-2] // 8

        # stage 0-2
        x = self.stem(x)

        # stage 3
        x_i = self.relu(self.i_branch_layers[0](x))
        x_p = self.p_branch_layers[0](x)
        x_d = self.d_branch_layers[0](x)

        comp_i = self.compression_1(x_i)
        x_p = self.pag_1(x_p, comp_i)
        diff_i = self.diff_1(x_i)
        # Out-of-place, where upstream writes `x_d += ...`; see BasicBlock.
        x_d = x_d + F.interpolate(
            diff_i,
            size=[h_out, w_out],
            mode='bilinear',
            align_corners=self.align_corners)
        if self.training:
            temp_p = x_p.clone()

        # stage 4
        x_i = self.relu(self.i_branch_layers[1](x_i))
        x_p = self.p_branch_layers[1](self.relu(x_p))
        x_d = self.d_branch_layers[1](self.relu(x_d))

        comp_i = self.compression_2(x_i)
        x_p = self.pag_2(x_p, comp_i)
        diff_i = self.diff_2(x_i)
        x_d = x_d + F.interpolate(
            diff_i,
            size=[h_out, w_out],
            mode='bilinear',
            align_corners=self.align_corners)
        if self.training:
            temp_d = x_d.clone()

        # stage 5
        x_i = self.i_branch_layers[2](x_i)
        x_p = self.p_branch_layers[2](self.relu(x_p))
        x_d = self.d_branch_layers[2](self.relu(x_d))

        x_i = self.spp(x_i)
        x_i = F.interpolate(
            x_i,
            size=[h_out, w_out],
            mode='bilinear',
            align_corners=self.align_corners)
        out = self.dfm(x_p, x_i, x_d)
        return (temp_p, out, temp_d) if self.training else out


# ─── decode head ────────────────────────────────────────────────────────────

class BasePIDHead(BaseModule):
    """One branch head: (norm, act, conv), then norm, act, then optional cls."""

    def __init__(self,
                 in_channels: int,
                 channels: int,
                 norm_cfg: Optional[Dict] = dict(type='BN'),
                 act_cfg: Optional[Dict] = dict(type='ReLU', inplace=True),
                 init_cfg: Optional[Dict] = None):
        super().__init__(init_cfg)
        self.conv = ConvModule(
            in_channels,
            channels,
            kernel_size=3,
            padding=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            order=('norm', 'act', 'conv'))
        _, self.norm = build_norm_layer(norm_cfg, num_features=channels)
        self.act = build_activation_layer(act_cfg)

    def forward(self, x: Tensor, cls_seg: Optional[nn.Module]) -> Tensor:
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        if cls_seg is not None:
            x = cls_seg(x)
        return x


@HEADS.register_module()
class PIDHead(BaseDecodeHead):
    r"""
    PIDNet decode head, ported to mmseg 0.x.

    THREE HEADS, ONE USED AT TEST TIME. `i_head` + `cls_seg` is the whole
    inference path; `p_head`/`p_cls_seg` and `d_head`/`d_cls_seg` exist only for
    the auxiliary P and boundary losses. They are still CONSTRUCTED, because the
    published checkpoint carries their weights and `load_checkpoint` would
    otherwise report them as unexpected keys.

    `self.conv_seg` and `self.cls_seg` come from BaseDecodeHead, so the
    19-channel output and utils.channel_probe behave exactly as for every other
    arch in the registry. `d_cls_seg` emits 1 channel (a boundary logit), not
    num_classes — it never reaches the segmentation output.
    """

    def __init__(self,
                 in_channels: int,
                 channels: int,
                 num_classes: int,
                 norm_cfg: Optional[Dict] = dict(type='BN'),
                 act_cfg: Optional[Dict] = dict(type='ReLU', inplace=True),
                 **kwargs):
        super().__init__(
            in_channels,
            channels,
            num_classes=num_classes,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            **kwargs)
        # out_channels only exists on BaseDecodeHead from mmseg 0.21; on older
        # 0.x it is num_classes by construction.
        out_channels = getattr(self, 'out_channels', None) or num_classes
        self.i_head = BasePIDHead(in_channels, channels, norm_cfg, act_cfg)
        self.p_head = BasePIDHead(in_channels // 2, channels, norm_cfg,
                                  act_cfg)
        self.d_head = BasePIDHead(
            in_channels // 2,
            in_channels // 4,
            norm_cfg,
        )
        self.p_cls_seg = nn.Conv2d(channels, out_channels, kernel_size=1)
        self.d_cls_seg = nn.Conv2d(in_channels // 4, 1, kernel_size=1)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(
        self, inputs: Union[Tensor, Tuple[Tensor, ...]]
    ) -> Union[Tensor, Tuple[Tensor, ...]]:
        """
        Eval: integral branch only, `inputs` being the backbone's single
        tensor. Note it does NOT call `_transform_inputs` — there is no feature
        pyramid to index into, which is why `in_index` / `input_transform` are
        absent from the config.
        """
        if self.training:
            x_p, x_i, x_d = inputs
            x_p = self.p_head(x_p, self.p_cls_seg)
            x_i = self.i_head(x_i, self.cls_seg)
            x_d = self.d_head(x_d, self.d_cls_seg)
            return x_p, x_i, x_d
        return self.i_head(inputs, self.cls_seg)

    def forward_train(self, inputs, img_metas, gt_semantic_seg, train_cfg):
        """
        REFUSES, rather than training the wrong objective. PIDNet's loss is
        four terms over (P, I, D) and needs a boundary GT map from the 1.x-only
        `GenerateEdge` transform plus the 1.x-only `OhemCrossEntropy` and
        `BoundaryLoss`. None of that is ported, because this project trains
        patches and never segmentors. Falling through to BaseDecodeHead.losses
        would quietly optimise the integral branch under a plain CE and look
        like it had worked.
        """
        raise NotImplementedError(
            "PIDHead here is INFERENCE-ONLY. Training PIDNet needs the "
            "boundary GT map from mmseg 1.x's GenerateEdge transform and its "
            "OhemCrossEntropy/BoundaryLoss, neither of which is ported into "
            "this mmseg 0.x build — see patchreach/models/pidnet.py. Use the "
            "published Cityscapes checkpoint; patch optimisation never needs "
            "a segmentor gradient step.")
