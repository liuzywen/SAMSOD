"""
    X分支输入rgb, Y分支输入t
    adapter换成high和low两个
"""
import torch
import torch.nn.functional as F
from torch import nn

from sam2.build_sam import build_sam2
from sam2.modeling.backbones.hieradet import do_pool, window_partition, window_unpartition
from sam2_model.MoE.Moase_moe_sep import Moe_Adapter



class Model(nn.Module):

    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.model = build_sam2(self.cfg.sam2_cfg, self.cfg.sam2_checkpoint)

        # Spatial dim for backbone feature maps
        self._bb_feat_sizes = [
            (128, 128),
            (64, 64),
            (32, 32)
        ]

        self.lenth0_1 = 2
        self.lenth2_7 = 6
        self.lenth8_43 = 36
        self.lenth44_47 = 4
        self.rgb0_1 = nn.ModuleList()
        self.rgb2_7 = nn.ModuleList()
        self.rgb8_43 = nn.ModuleList()
        self.rgb44_47 = nn.ModuleList()

        self.depth0_1d = nn.ModuleList()
        self.depth2_7d = nn.ModuleList()
        self.depth8_43d = nn.ModuleList()
        self.depth44_47d = nn.ModuleList()
        for i in range(self.lenth0_1):
            ad = Moe_Adapter(144, 32, 0.5)
            ads = Moe_Adapter(144, 32, 0.5)

            self.rgb0_1.append(ad)
            self.depth0_1d.append(ads)
        for i in range(self.lenth2_7):
            ad = Moe_Adapter(288, 32, 0.5)
            ads = Moe_Adapter(288, 32, 0.5)

            self.rgb2_7.append(ad)
            self.depth2_7d.append(ads)

        for i in range(self.lenth8_43):
            ad = Moe_Adapter(576, 32, 0.5)
            ads = Moe_Adapter(576, 32, 0.5)

            self.rgb8_43.append(ad)
            self.depth8_43d.append(ads)

        for i in range(self.lenth44_47):
            ad = Moe_Adapter(1152, 32, 0.5)
            ads = Moe_Adapter(1152, 32, 0.5)

            self.rgb44_47.append(ad)
            self.depth44_47d.append(ads)

    def setup(self):
        if self.cfg.freeze_image_encoder:
            print("冻结编码器")
            for name, param in self.model.named_parameters():
                if "sam_mask_decoder" not in name:
                    param.requires_grad_(False)

    def forward(self, rgb, depth):
        rgb = self.model.image_encoder.trunk.patch_embed(rgb)
        rgb = rgb + self.model.image_encoder.trunk._get_pos_embed(rgb.shape[1:3])
        depth = self.model.image_encoder.trunk.patch_embed(depth)
        depth = depth + self.model.image_encoder.trunk._get_pos_embed(depth.shape[1:3])
        depth_outputs = []
        rgb_outputs = []

        for i, blk in enumerate(self.model.image_encoder.trunk.blocks):
            rgb_shortcut = rgb  # B, H, W, C
            rgb = blk.norm1(rgb)
            # Skip connection
            if blk.dim != blk.dim_out:
                rgb_shortcut = do_pool(blk.proj(rgb), blk.pool)
            # Window partition
            window_size = blk.window_size
            if window_size > 0:
                Hb, Wb = rgb.shape[1], rgb.shape[2]
                rgb, pad_hw = window_partition(rgb, window_size)

            # Window Attention + Q Pooling (if stage change)
            B, H, W, _ = rgb.shape
            # qkv with shape (B, H * W, 3, nHead, C)
            qkv = blk.attn.qkv(rgb).reshape(B, H * W, 3, blk.attn.num_heads, -1)
            # q, k, v with shape (B, H * W, nheads, C)
            q, k, v = torch.unbind(qkv, 2)
            # Q pooling (for downsample at stage changes)
            if blk.attn.q_pool:
                q = do_pool(q.reshape(B, H, W, -1), blk.attn.q_pool)
                H, W = q.shape[1:3]  # downsampled shape
                q = q.reshape(B, H * W, blk.attn.num_heads, -1)
            # Torch's SDPA expects [B, nheads, H*W, C] so we transpose
            rgb = F.scaled_dot_product_attention(
                q.transpose(1, 2),
                k.transpose(1, 2),
                v.transpose(1, 2),
            )
            # Transpose back
            rgb = rgb.transpose(1, 2)
            rgb = rgb.reshape(B, H, W, -1)
            rgb = blk.attn.proj(rgb)

            if blk.q_stride:
                # Shapes have changed due to Q pooling
                window_size = blk.window_size // blk.q_stride[0]
                Hb, Wb = rgb_shortcut.shape[1:3]

                pad_h = (window_size - Hb % window_size) % window_size
                pad_w = (window_size - Wb % window_size) % window_size
                pad_hw = (Hb + pad_h, Wb + pad_w)
            # Reverse window partition
            if blk.window_size > 0:
                rgb = window_unpartition(rgb, window_size, pad_hw, (Hb, Wb))

            rgb = rgb_shortcut + blk.drop_path(rgb)
            # MLP
            if 0 <= i <= 1:
                rgb = rgb + blk.drop_path(blk.mlp(blk.norm2(rgb))) + self.rgb0_1[i](rgb)
            if 2 <= i <= 7:
                rgb = rgb + blk.drop_path(blk.mlp(blk.norm2(rgb))) + self.rgb2_7[i - 2](rgb)
            if 8 <= i <= 43:
                rgb = rgb + blk.drop_path(blk.mlp(blk.norm2(rgb))) + self.rgb8_43[i - 8](rgb)
            if 44 <= i <= 47:
                rgb = rgb + blk.drop_path(blk.mlp(blk.norm2(rgb))) + self.rgb44_47[i - 44](rgb)

            depth_shortcut = depth  # B, H, W, C
            depth = blk.norm1(depth)
            # Skip connection
            if blk.dim != blk.dim_out:
                depth_shortcut = do_pool(blk.proj(depth), blk.pool)
            # Window partition
            window_size = blk.window_size
            if window_size > 0:
                Hb, Wb = depth.shape[1], depth.shape[2]
                depth, pad_hw = window_partition(depth, window_size)

            # Window Attention + Q Pooling (if stage change)
            B, H, W, _ = depth.shape
            # qkv with shape (B, H * W, 3, nHead, C)
            qkv = blk.attn.qkv(depth).reshape(B, H * W, 3, blk.attn.num_heads, -1)
            # q, k, v with shape (B, H * W, nheads, C)
            q, k, v = torch.unbind(qkv, 2)
            # Q pooling (for downsample at stage changes)
            if blk.attn.q_pool:
                q = do_pool(q.reshape(B, H, W, -1), blk.attn.q_pool)
                H, W = q.shape[1:3]  # downsampled shape
                q = q.reshape(B, H * W, blk.attn.num_heads, -1)
            # Torch's SDPA expects [B, nheads, H*W, C] so we transpose
            depth = F.scaled_dot_product_attention(
                q.transpose(1, 2),
                k.transpose(1, 2),
                v.transpose(1, 2),
            )
            # Transpose back
            depth = depth.transpose(1, 2)
            depth = depth.reshape(B, H, W, -1)
            depth = blk.attn.proj(depth)

            if blk.q_stride:
                # Shapes have changed due to Q pooling
                window_size = blk.window_size // blk.q_stride[0]
                Hb, Wb = depth_shortcut.shape[1:3]

                pad_h = (window_size - Hb % window_size) % window_size
                pad_w = (window_size - Wb % window_size) % window_size
                pad_hw = (Hb + pad_h, Wb + pad_w)
            # Reverse window partition
            if blk.window_size > 0:
                depth = window_unpartition(depth, window_size, pad_hw, (Hb, Wb))

            depth = depth_shortcut + blk.drop_path(depth)
            # MLP
            if 0 <= i <= 1:
                depth = depth + blk.drop_path(blk.mlp(blk.norm2(depth))) + self.depth0_1d[i](depth)
            if 2 <= i <= 7:
                depth = depth + blk.drop_path(blk.mlp(blk.norm2(depth))) + self.depth2_7d[i - 2](depth)
            if 8 <= i <= 43:
                depth = depth + blk.drop_path(blk.mlp(blk.norm2(depth))) + self.depth8_43d[i - 8](depth)
            if 44 <= i <= 47:
                depth = depth + blk.drop_path(blk.mlp(blk.norm2(depth))) + self.depth44_47d[i - 44](depth)

            # [1, 7, 43, 47]
            if (i == self.model.image_encoder.trunk.stage_ends[-1]) or (
                    i in self.model.image_encoder.trunk.stage_ends and self.model.image_encoder.trunk.return_interm_layers
            ):
                feats_rgb = rgb.permute(0, 3, 1, 2)
                feats_depth = depth.permute(0, 3, 1, 2)
                depth_outputs.append(feats_depth)
                rgb_outputs.append(feats_rgb)

        all_two = [
            (x + y)
            for x, y in zip(rgb_outputs, depth_outputs)
        ]
        rgb_two = [
            x
            for x, y in zip(rgb_outputs, depth_outputs)
        ]
        depth_two = [
            y
            for x, y in zip(rgb_outputs, depth_outputs)
        ]
        two_feats, _ = self.model.image_encoder.neck(all_two)
        rgb_feats, _ = self.model.image_encoder.neck(rgb_two)
        depth_feats, _ = self.model.image_encoder.neck(depth_two)

        if self.model.image_encoder.scalp > 0:
            # Discard the lowest resolution features
            two_feats = two_feats[: -self.model.image_encoder.scalp]
            rgb_feats = rgb_feats[: -self.model.image_encoder.scalp]
            depth_feats = depth_feats[: -self.model.image_encoder.scalp]
        two_out = {
            "backbone_fpn": two_feats,
        }
        rgb_out = {
            "backbone_fpn": rgb_feats,
        }
        depth_out = {
            "backbone_fpn": depth_feats,
        }
        if self.model.use_high_res_features_in_sam:
            # precompute projected level 0 and level 1 features in SAM decoder
            # to avoid running it again on every SAM click
            two_out["backbone_fpn"][0] = self.model.sam_mask_decoder.conv_s0(
                two_out["backbone_fpn"][0]
            )
            two_out["backbone_fpn"][1] = self.model.sam_mask_decoder.conv_s1(
                two_out["backbone_fpn"][1]
            )
            rgb_out["backbone_fpn"][0] = self.model.sam_mask_decoder.conv_s0(
                rgb_out["backbone_fpn"][0]
            )
            rgb_out["backbone_fpn"][1] = self.model.sam_mask_decoder.conv_s1(
                rgb_out["backbone_fpn"][1]
            )
            depth_out["backbone_fpn"][0] = self.model.sam_mask_decoder.conv_s0(
                depth_out["backbone_fpn"][0]
            )
            depth_out["backbone_fpn"][1] = self.model.sam_mask_decoder.conv_s1(
                depth_out["backbone_fpn"][1]
            )
        two_out = two_out.copy()
        rgb_out = rgb_out.copy()
        depth_out = depth_out.copy()

        two_feature_maps = two_out["backbone_fpn"][-self.model.num_feature_levels:]
        rgb_feature_maps = rgb_out["backbone_fpn"][-self.model.num_feature_levels:]
        depth_feature_maps = depth_out["backbone_fpn"][-self.model.num_feature_levels:]

        # flatten NxCxHxW to HWxNxC
        two_vision_feats = [x.flatten(2).permute(2, 0, 1) for x in two_feature_maps]
        rgb_vision_feats = [x.flatten(2).permute(2, 0, 1) for x in rgb_feature_maps]
        depth_vision_feats = [x.flatten(2).permute(2, 0, 1) for x in depth_feature_maps]

        if self.model.directly_add_no_mem_embed:
            two_vision_feats[-1] = two_vision_feats[-1] + self.model.no_mem_embed
            rgb_vision_feats[-1] = rgb_vision_feats[-1] + self.model.no_mem_embed
            depth_vision_feats[-1] = depth_vision_feats[-1] + self.model.no_mem_embed
        b, Hn, Wn, c = rgb.shape
        two_feats_after = [
                              feat.permute(1, 2, 0).view(b, -1, *feat_size)
                              for feat, feat_size in zip(two_vision_feats[::-1], self._bb_feat_sizes[::-1])
                          ][::-1]
        rgb_feats_after = [
                              feat.permute(1, 2, 0).view(b, -1, *feat_size)
                              for feat, feat_size in zip(rgb_vision_feats[::-1], self._bb_feat_sizes[::-1])
                          ][::-1]
        depth_feats_after = [
                                feat.permute(1, 2, 0).view(b, -1, *feat_size)
                                for feat, feat_size in zip(depth_vision_feats[::-1], self._bb_feat_sizes[::-1])
                            ][::-1]
        two_end_features = {"image_embed": two_feats_after[-1], "high_res_feats": two_feats_after[:-1]}
        rgb_end_features = {"image_embed": rgb_feats_after[-1], "high_res_feats": rgb_feats_after[:-1]}
        depth_end_features = {"image_embed": depth_feats_after[-1], "high_res_feats": depth_feats_after[:-1]}

        two_high_res_features = [
            feat_level
            for feat_level in two_end_features["high_res_feats"]
        ]
        rgb_high_res_features = [
            feat_level
            for feat_level in rgb_end_features["high_res_feats"]
        ]
        depth_high_res_features = [
            feat_level
            for feat_level in depth_end_features["high_res_feats"]
        ]
        sparse_embeddings, dense_embeddings = self.model.sam_prompt_encoder(
            points=None,
            boxes=None,
            masks=None,
        )

        low_res_masks1, iou_predictions1, _, _ = self.model.sam_mask_decoder(
            image_embeddings=two_end_features["image_embed"],
            image_pe=self.model.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
            repeat_image=False,
            high_res_features=two_high_res_features,
        )
        low_res_masks2, iou_predictions2, _, _ = self.model.sam_mask_decoder(
            image_embeddings=rgb_end_features["image_embed"],
            image_pe=self.model.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
            repeat_image=False,
            high_res_features=rgb_high_res_features,
        )
        low_res_masks3, iou_predictions3, _, _ = self.model.sam_mask_decoder(
            image_embeddings=depth_end_features["image_embed"],
            image_pe=self.model.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
            repeat_image=False,
            high_res_features=depth_high_res_features,
        )
        # Upscale the masks to the original image resolution
        masks1 = F.interpolate(low_res_masks1, (512, 512), mode="bilinear", align_corners=False)
        masks2 = F.interpolate(low_res_masks2, (512, 512), mode="bilinear", align_corners=False)
        masks3 = F.interpolate(low_res_masks3, (512, 512), mode="bilinear", align_corners=False)
        masks = masks3 + masks2 + masks1
        return masks1, masks2, masks3, masks
