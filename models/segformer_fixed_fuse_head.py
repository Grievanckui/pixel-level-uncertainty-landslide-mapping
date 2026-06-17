from __future__ import annotations

from typing import List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.registry import MODELS
from mmseg.models.decode_heads.decode_head import BaseDecodeHead


class _GateNet(nn.Module):
    """Pixel-wise gating network producing 4 weight maps."""

    def __init__(self, in_channels: int, hidden_channels: int = 32, dropout_ratio: float = 0.0):
        super().__init__()
        # GN is stable for small dataset / small batch; groups must divide channels.
        if hidden_channels % 8 == 0:
            groups = 8
        elif hidden_channels % 4 == 0:
            groups = 4
        else:
            groups = 1

        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=groups, num_channels=hidden_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_ratio) if dropout_ratio and dropout_ratio > 0 else nn.Identity(),
            nn.Conv2d(hidden_channels, 4, kernel_size=1, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)  # [B, 4, H, W]


@MODELS.register_module()
class SegformerFixedFuseHead(BaseDecodeHead):
    """SegFormer-like decode head with pixel-wise gating and true auxiliary losses.

    Pipeline:
    1) project each backbone feature to `channels` and upsample to highest resolution
    2) compute pixel-wise weights w_s(x) via a tiny gate net (softmax over 4 scales)
    3) gate features: f'_s = w_s * f_s
    4) fuse:
       - concat: concat([f'_1..f'_4]) -> 1x1 conv(+GN+ReLU) -> fused_feat
       - (optional) sum: sum_s f'_s
    5) final logits = cls_seg(fused_feat)

    Training:
    - main loss on fused logits (BaseDecodeHead.loss_by_feat)
    - optional auxiliary CE losses on each scale logits (small weight)

    Notes about gates export:
    - mmseg's predict/resize pipeline expects decode_head.forward() to return a Tensor (seg_logits).
      If forward() returns a tuple (seg_logits, gates), inference_model() will crash.
    - Therefore, when return_gates=True we cache gates to `self.latest_gates` but still return seg_logits only.
      You can read gates after inference via: `model.decode_head.latest_gates`.
    """

    def __init__(
        self,
        in_channels: Sequence[int],
        channels: int,
        num_classes: int,
        fuse_weights: Sequence[float] = (0.1, 0.2, 0.3, 0.4),
        dropout_ratio: float = 0.1,
        align_corners: bool = False,
        use_gating: bool = True,
        gate_channels: int = 32,
        gate_dropout: float = 0.0,
        fusion_mode: str = 'concat',  # 'concat' (recommended) or 'sum'
        aux_loss: bool = True,
        aux_loss_weight: float = 0.1,  # total aux weight (distributed equally across 4 scales)
        aux_loss_use_main_loss_decode: bool = True,
        return_gates: bool = False,
        **kwargs,
    ):
        super().__init__(
            in_channels=in_channels,
            channels=channels,
            num_classes=num_classes,
            dropout_ratio=dropout_ratio,
            align_corners=align_corners,
            input_transform='multiple_select',
            **kwargs)

        assert isinstance(in_channels, (list, tuple)) and len(in_channels) == 4, \
            f"SegformerFixedFuseHead expects 4 feature maps, got in_channels={in_channels}"
        assert fusion_mode in ('sum', 'concat'), f"fusion_mode must be 'sum' or 'concat', got {fusion_mode}"

        self.use_gating = use_gating
        self.return_gates = return_gates
        self.fusion_mode = fusion_mode

        # NEW: cache slot for gates from latest forward
        self.latest_gates: Optional[torch.Tensor] = None  # [B,4,H,W] on same device as forward

        self.aux_loss = aux_loss
        self.aux_loss_weight = float(aux_loss_weight)
        self.aux_loss_use_main_loss_decode = aux_loss_use_main_loss_decode

        # scalar weight prior (used only if use_gating=False)
        w = torch.tensor(list(fuse_weights), dtype=torch.float32)
        if w.numel() != 4:
            raise ValueError(f"fuse_weights must have length 4, got {len(fuse_weights)}")
        w = w / (w.sum() + 1e-12)
        self.register_buffer('fuse_weights', w)

        # projections
        self.proj = nn.ModuleList([
            nn.Conv2d(in_channels[0], channels, kernel_size=1, bias=False),
            nn.Conv2d(in_channels[1], channels, kernel_size=1, bias=False),
            nn.Conv2d(in_channels[2], channels, kernel_size=1, bias=False),
            nn.Conv2d(in_channels[3], channels, kernel_size=1, bias=False),
        ])

        self.gate = _GateNet(in_channels=4 * channels, hidden_channels=gate_channels, dropout_ratio=gate_dropout)

        if self.fusion_mode == 'concat':
            # 4C -> C
            if channels % 32 == 0:
                gn_groups = 32
            elif channels % 16 == 0:
                gn_groups = 16
            elif channels % 8 == 0:
                gn_groups = 8
            else:
                gn_groups = 1

            self.fuse_conv = nn.Sequential(
                nn.Conv2d(4 * channels, channels, kernel_size=1, bias=False),
                nn.GroupNorm(num_groups=gn_groups, num_channels=channels),
                nn.ReLU(inplace=True),
            )
        else:
            self.fuse_conv = None

        self.dropout = nn.Dropout2d(dropout_ratio) if dropout_ratio and dropout_ratio > 0 else nn.Identity()
        self.cls_seg = nn.Conv2d(channels, num_classes, kernel_size=1)

        # per-scale logits for aux losses
        if self.aux_loss:
            self.scale_cls = nn.ModuleList([
                nn.Conv2d(channels, num_classes, kernel_size=1),
                nn.Conv2d(channels, num_classes, kernel_size=1),
                nn.Conv2d(channels, num_classes, kernel_size=1),
                nn.Conv2d(channels, num_classes, kernel_size=1),
            ])

    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        """Forward used by mmseg in both training (via loss()) and inference predict.

        IMPORTANT:
        - Must return a Tensor seg_logits for mmseg's predict/resize pipeline.
        - If return_gates=True, gates are cached to self.latest_gates for external access.
        """
        fused_logits, weights, _ = self._forward_impl(inputs, return_scale_logits=False)

        # cache gates for later read (e.g., MC sampling script)
        if self.return_gates and weights is not None:
            self.latest_gates = weights.detach()
        else:
            self.latest_gates = None

        # Always return logits only (avoid tuple that breaks inference_model)
        return fused_logits

    def _forward_impl(
        self,
        inputs: List[torch.Tensor],
        return_scale_logits: bool,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[List[torch.Tensor]]]:
        feats = self._transform_inputs(inputs)
        x1, x2, x3, x4 = feats
        target_size = x1.shape[2:]
        xs = [x1, x2, x3, x4]

        proj_feats: List[torch.Tensor] = []
        for i in range(4):
            z = self.proj[i](xs[i])
            z = F.interpolate(z, size=target_size, mode='bilinear', align_corners=self.align_corners)
            proj_feats.append(z)

        if self.use_gating:
            gate_in = torch.cat(proj_feats, dim=1)       # [B, 4C, H, W]
            gate_logits = self.gate(gate_in)             # [B, 4, H, W]
            weights = torch.softmax(gate_logits, dim=1)  # [B, 4, H, W]
            gated = [
                weights[:, 0:1] * proj_feats[0],
                weights[:, 1:2] * proj_feats[1],
                weights[:, 2:3] * proj_feats[2],
                weights[:, 3:4] * proj_feats[3],
            ]
        else:
            weights = None
            w = self.fuse_weights
            gated = [w[i] * proj_feats[i] for i in range(4)]

        if self.fusion_mode == 'sum':
            fused_feat = gated[0] + gated[1] + gated[2] + gated[3]
        else:
            fused_feat = self.fuse_conv(torch.cat(gated, dim=1))

        fused_feat = self.dropout(fused_feat)
        fused_logits = self.cls_seg(fused_feat)

        if return_scale_logits and self.aux_loss:
            scale_logits = [self.scale_cls[i](self.dropout(proj_feats[i])) for i in range(4)]
        else:
            scale_logits = None

        return fused_logits, weights, scale_logits

    def loss(self, inputs: List[torch.Tensor], batch_data_samples, train_cfg: Optional[dict] = None):
        """Training-time loss.

        mmseg's segmentors call decode_head.loss() during training.
        We override it so we can compute:
        - main loss on fused_logits
        - aux losses on scale_logits
        """
        seg_logits, _, scale_logits = self._forward_impl(inputs, return_scale_logits=True)

        # main loss (uses config loss_decode)
        losses = super().loss_by_feat(seg_logits, batch_data_samples)

        # aux losses
        if self.aux_loss and scale_logits is not None and self.aux_loss_weight > 0:
            per_scale_w = self.aux_loss_weight / 4.0

            # If True: reuse the SAME loss_decode settings (class weights etc.) for aux losses
            # If False: use plain CE without extra weighting (sometimes more stable).
            if self.aux_loss_use_main_loss_decode:
                for i, s_logit in enumerate(scale_logits):
                    aux_i = super().loss_by_feat(s_logit, batch_data_samples)
                    # scale all keys to keep naming clean
                    for k, v in aux_i.items():
                        losses[f'aux{i}.{k}'] = v * per_scale_w
            else:
                # plain CE aux (no class_weight); robust fallback
                seg_label = self._stack_batch_gt(batch_data_samples)
                if seg_label.dim() == 4:
                    seg_label = seg_label.squeeze(1)
                for i, s_logit in enumerate(scale_logits):
                    # resize logits to gt
                    if s_logit.shape[-2:] != seg_label.shape[-2:]:
                        s_logit = F.interpolate(
                            s_logit, size=seg_label.shape[-2:], mode='bilinear', align_corners=self.align_corners)
                    losses[f'aux{i}.loss_ce'] = F.cross_entropy(
                        s_logit, seg_label.long(), ignore_index=self.ignore_index) * per_scale_w

        return losses