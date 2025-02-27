from functools import partial
import torch
import torch.nn as nn
import spconv.pytorch as spconv
from timm.models.layers import DropPath
from collections import OrderedDict
import functools

from Modules.Loss import point_wise_loss
from Modules.Utils import cuda_cast

try:
    import flash_attn
except ImportError:
    flash_attn = None

from .blocks import *

# The adapted Transformer with heads
class PointTransformerWithHeads(nn.Module):
    def __init__(
        self, 
        dim_feat=4, 
        use_feats=False, 
        voxel_size=0.02,
        loss_multiplier_semantic=1,
        loss_multiplier_offset=1,
        enable_flash=False,
        **kwargs
    ):
        super().__init__()

        self.voxel_size = voxel_size
        self.loss_multiplier_semantic = loss_multiplier_semantic
        self.loss_multiplier_offset = loss_multiplier_offset
        self.use_feats = use_feats

        self.backbone = PointTransformerV3(
            in_channels = dim_feat,
            enable_flash = enable_flash
        )

        norm_fn = functools.partial(nn.BatchNorm1d, eps=1e-4, momentum=0.1)
        
        # head
        self.semantic_linear = MLP_Head(32, 2, norm_fn=norm_fn, num_layers=2)
        self.offset_linear = MLP_Head(32, 3, norm_fn=norm_fn, num_layers=2)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, MLP_Head):
                m.init_weights()

    def forward(self, batch, return_loss, **kwargs):

        # Extract features and points and put them into point_dict
        feats = batch["feats"]
        if not self.use_feats:
            feats = torch.ones_like(feats)  # Replace features with ones if use_feats is False

        point_dict = {
            "coord": batch["coords"].to('cuda'),
            "feat": feats.to('cuda'),
            "grid_size": self.voxel_size,
            "batch": batch["batch_ids"].to('cuda')
        }

        output = self.forward_backbone( point_dict )
        output = self.forward_head( output )

        if return_loss:
            output = self.get_loss(model_output=output, **batch)

        return output

    @cuda_cast
    def forward_backbone(self, point_dict, **kwargs):

        output = self.backbone( point_dict )

        return output

    def forward_head(self, backbone_output, **kwargs):
        output = dict()
        backbone_feats = backbone_output["feat"] # Expected shape: B x N x 32
        output['backbone_feats'] = backbone_feats
        # if backbone_feats.dim() == 2:
        #     backbone_feats = backbone_feats.unsqueeze(0)  # Now shape becomes (1, N, C)
        # backbone_feats = backbone_feats.permute(0, 2, 1)  # Convert B x N x C -> B x C x N
        backbone_feats = backbone_feats.permute(0, 1)

        output['semantic_prediction_logits'] = self.semantic_linear(backbone_feats)  # (B, 2, N)
        output['offset_predictions'] = self.offset_linear(backbone_feats)  # (B, 3, N)

        return output

    def get_loss(self, model_output, semantic_labels, offset_labels, masks_off, **kwargs):
        loss_dict = dict()
        semantic_loss, offset_loss = point_wise_loss(model_output['semantic_prediction_logits'].float(), model_output['offset_predictions'][masks_off].float(), 
                                                            semantic_labels, offset_labels[masks_off])
        loss_dict['semantic_loss'] = semantic_loss * self.loss_multiplier_semantic
        loss_dict['offset_loss'] = offset_loss * self.loss_multiplier_offset

        loss = sum(_value for _value in loss_dict.values())
        return loss, loss_dict

# The original Transformer model

class PointTransformerV3(PointModule):
    def __init__(
        self,
        in_channels=6,
        order=("z", "z-trans", "hilbert", "hilbert-trans"),
        stride=(2, 2, 2, 2),
        enc_depths=(2, 2, 2, 6, 2),
        enc_channels=(32, 64, 128, 256, 512),
        enc_num_head=(2, 4, 8, 16, 32),
        enc_patch_size=(1024, 1024, 1024, 1024, 1024),
        dec_depths=(2, 2, 2, 2),
        dec_channels=(64, 64, 128, 256),
        dec_num_head=(4, 4, 8, 16),
        dec_patch_size=(1024, 1024, 1024, 1024),
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.3,
        pre_norm=True,
        shuffle_orders=True,
        enable_rpe=False,
        enable_flash=False,
        upcast_attention=False,
        upcast_softmax=False,
        cls_mode=False,
        pdnorm_bn=False,
        pdnorm_ln=False,
        pdnorm_decouple=True,
        pdnorm_adaptive=False,
        pdnorm_affine=True,
        pdnorm_conditions=("ScanNet", "S3DIS", "Structured3D"),
    ):
        super().__init__()
        self.num_stages = len(enc_depths)
        self.order = [order] if isinstance(order, str) else order
        self.cls_mode = cls_mode
        self.shuffle_orders = shuffle_orders

        assert self.num_stages == len(stride) + 1
        assert self.num_stages == len(enc_depths)
        assert self.num_stages == len(enc_channels)
        assert self.num_stages == len(enc_num_head)
        assert self.num_stages == len(enc_patch_size)
        assert self.cls_mode or self.num_stages == len(dec_depths) + 1
        assert self.cls_mode or self.num_stages == len(dec_channels) + 1
        assert self.cls_mode or self.num_stages == len(dec_num_head) + 1
        assert self.cls_mode or self.num_stages == len(dec_patch_size) + 1

        # norm layers
        if pdnorm_bn:
            bn_layer = partial(
                PDNorm,
                norm_layer=partial(
                    nn.BatchNorm1d, eps=1e-3, momentum=0.01, affine=pdnorm_affine
                ),
                conditions=pdnorm_conditions,
                decouple=pdnorm_decouple,
                adaptive=pdnorm_adaptive,
            )
        else:
            bn_layer = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        if pdnorm_ln:
            ln_layer = partial(
                PDNorm,
                norm_layer=partial(nn.LayerNorm, elementwise_affine=pdnorm_affine),
                conditions=pdnorm_conditions,
                decouple=pdnorm_decouple,
                adaptive=pdnorm_adaptive,
            )
        else:
            ln_layer = nn.LayerNorm
        # activation layers
        act_layer = nn.GELU

        self.embedding = Embedding(
            in_channels=in_channels,
            embed_channels=enc_channels[0],
            norm_layer=bn_layer,
            act_layer=act_layer,
        )

        # encoder
        enc_drop_path = [
            x.item() for x in torch.linspace(0, drop_path, sum(enc_depths))
        ]
        self.enc = PointSequential()
        for s in range(self.num_stages):
            enc_drop_path_ = enc_drop_path[
                sum(enc_depths[:s]) : sum(enc_depths[: s + 1])
            ]
            enc = PointSequential()
            if s > 0:
                enc.add(
                    SerializedPooling(
                        in_channels=enc_channels[s - 1],
                        out_channels=enc_channels[s],
                        stride=stride[s - 1],
                        norm_layer=bn_layer,
                        act_layer=act_layer,
                    ),
                    name="down",
                )
            for i in range(enc_depths[s]):
                enc.add(
                    Block(
                        channels=enc_channels[s],
                        num_heads=enc_num_head[s],
                        patch_size=enc_patch_size[s],
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        qk_scale=qk_scale,
                        attn_drop=attn_drop,
                        proj_drop=proj_drop,
                        drop_path=enc_drop_path_[i],
                        norm_layer=ln_layer,
                        act_layer=act_layer,
                        pre_norm=pre_norm,
                        order_index=i % len(self.order),
                        cpe_indice_key=f"stage{s}",
                        enable_rpe=enable_rpe,
                        enable_flash=enable_flash,
                        upcast_attention=upcast_attention,
                        upcast_softmax=upcast_softmax,
                    ),
                    name=f"block{i}",
                )
            if len(enc) != 0:
                self.enc.add(module=enc, name=f"enc{s}")

        # decoder
        if not self.cls_mode:
            dec_drop_path = [
                x.item() for x in torch.linspace(0, drop_path, sum(dec_depths))
            ]
            self.dec = PointSequential()
            dec_channels = list(dec_channels) + [enc_channels[-1]]
            for s in reversed(range(self.num_stages - 1)):
                dec_drop_path_ = dec_drop_path[
                    sum(dec_depths[:s]) : sum(dec_depths[: s + 1])
                ]
                dec_drop_path_.reverse()
                dec = PointSequential()
                dec.add(
                    SerializedUnpooling(
                        in_channels=dec_channels[s + 1],
                        skip_channels=enc_channels[s],
                        out_channels=dec_channels[s],
                        norm_layer=bn_layer,
                        act_layer=act_layer,
                    ),
                    name="up",
                )
                for i in range(dec_depths[s]):
                    dec.add(
                        Block(
                            channels=dec_channels[s],
                            num_heads=dec_num_head[s],
                            patch_size=dec_patch_size[s],
                            mlp_ratio=mlp_ratio,
                            qkv_bias=qkv_bias,
                            qk_scale=qk_scale,
                            attn_drop=attn_drop,
                            proj_drop=proj_drop,
                            drop_path=dec_drop_path_[i],
                            norm_layer=ln_layer,
                            act_layer=act_layer,
                            pre_norm=pre_norm,
                            order_index=i % len(self.order),
                            cpe_indice_key=f"stage{s}",
                            enable_rpe=enable_rpe,
                            enable_flash=enable_flash,
                            upcast_attention=upcast_attention,
                            upcast_softmax=upcast_softmax,
                        ),
                        name=f"block{i}",
                    )
                self.dec.add(module=dec, name=f"dec{s}")

    def forward(self, data_dict):
        """
        A data_dict is a dictionary containing properties of a batched point cloud.
        It should contain the following properties for PTv3:
        1. "feat": feature of point cloud
        2. "grid_coord": discrete coordinate after grid sampling (voxelization) or "coord" + "grid_size"
        3. "offset" or "batch": https://github.com/Pointcept/Pointcept?tab=readme-ov-file#offset
        """
        point = Point(data_dict)
        point.serialization(order=self.order, shuffle_orders=self.shuffle_orders)
        point.sparsify()

        point = self.embedding(point)
        point = self.enc(point)
        if not self.cls_mode:
            point = self.dec(point)
        return point