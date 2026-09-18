from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BitsAndBytesConfig, CLIPVisionModel

from utils.utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                         DEFAULT_IMAGE_PATCH_TOKEN)

from .llava.model.language_model.llava_llama import (LlavaLlamaForCausalLM,
                                                     LlavaLlamaModel)
from .segment_anything import build_sam_vit_h


def dice_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
    scale=1000,  # 100000.0,
    eps=1e-6,
):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1, 2)
    targets = targets.flatten(1, 2)
    numerator = 2 * (inputs / scale * targets).sum(-1)
    denominator = (inputs / scale).sum(-1) + (targets / scale).sum(-1)
    loss = 1 - (numerator + eps) / (denominator + eps)
    loss = loss.sum() / (num_masks + 1e-8)
    return loss


def sigmoid_ce_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    loss = loss.flatten(1, 2).mean(1).sum() / (num_masks + 1e-8)
    return loss


def dice_loss_per_mask(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    scale=1000,
    eps=1e-6,
):
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1, 2)
    targets = targets.flatten(1, 2)
    numerator = 2 * (inputs / scale * targets).sum(-1)
    denominator = (inputs / scale).sum(-1) + (targets / scale).sum(-1)
    return 1 - (numerator + eps) / (denominator + eps)


def sigmoid_ce_loss_per_mask(
    inputs: torch.Tensor,
    targets: torch.Tensor,
):
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    return loss.flatten(1, 2).mean(1)


def soft_iou_matrix(
    pred_logits: torch.Tensor,
    target_masks: torch.Tensor,
    eps: float = 1e-6,
):
    pred_probs = torch.nan_to_num(pred_logits.sigmoid())
    targets = target_masks.to(device=pred_logits.device, dtype=pred_logits.dtype)
    pred_flat = pred_probs.flatten(1)
    target_flat = targets.flatten(1)
    intersection = torch.matmul(pred_flat, target_flat.transpose(0, 1))
    pred_area = pred_flat.sum(-1, keepdim=True)
    target_area = target_flat.sum(-1).unsqueeze(0)
    union = pred_area + target_area - intersection
    return intersection / union.clamp_min(eps)


class LisaMetaModel:
    def __init__(
        self,
        config,
        **kwargs,
    ):
        super(LisaMetaModel, self).__init__(config)

        self.config = config
        if not hasattr(self.config, "train_mask_decoder"):
            self.config.train_mask_decoder = kwargs["train_mask_decoder"]
            self.config.out_dim = kwargs["out_dim"]
            self.vision_pretrained = kwargs.get("vision_pretrained", None)
        else:
            self.vision_pretrained = kwargs.get("vision_pretrained", None)
            self.initialize_lisa_modules(self.config)

    def initialize_lisa_modules(self, config):
        # SAM
        self.visual_model = build_sam_vit_h(self.vision_pretrained)
        for param in self.visual_model.parameters():
            param.requires_grad = False
        if config.train_mask_decoder:
            self.visual_model.mask_decoder.train()
            for param in self.visual_model.mask_decoder.parameters():
                param.requires_grad = True

        # Projection layer
        in_dim = config.hidden_size
        out_dim = config.out_dim
        text_fc = [
            nn.Linear(in_dim, in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim, out_dim),
            nn.Dropout(0.0),
        ]
        self.text_hidden_fcs = nn.ModuleList([nn.Sequential(*text_fc)])
        self.text_hidden_fcs.train()
        for param in self.text_hidden_fcs.parameters():
            param.requires_grad = True

        ref_hidden_fc = [
            nn.Linear(in_dim, in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim, out_dim),
            nn.Dropout(0.0),
        ]
        self.ref_hidden_fcs = nn.ModuleList([nn.Sequential(*ref_hidden_fc)])
        self.ref_hidden_fcs.train()
        for param in self.ref_hidden_fcs.parameters():
            param.requires_grad = True

        if not hasattr(config, "ref_mask_pool_size"):
            config.ref_mask_pool_size = 16
        ref_in_dim = config.ref_mask_pool_size * config.ref_mask_pool_size + 4
        self.ref_visual_fcs = nn.Sequential(
            nn.Linear(ref_in_dim, out_dim),
            nn.ReLU(inplace=True),
            nn.Linear(out_dim, out_dim),
            nn.Dropout(0.0),
        )
        self.ref_visual_fcs.train()
        for param in self.ref_visual_fcs.parameters():
            param.requires_grad = True
        ref_context_init_scale = getattr(config, "ref_context_init_scale", 1.0)
        self.ref_embedding_scale = nn.Parameter(
            torch.ones(1) * float(ref_context_init_scale)
        )

        ref_input_bbox_fc = [
            nn.Linear(4, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, in_dim),
        ]
        self.ref_input_bbox_fcs = nn.Sequential(*ref_input_bbox_fc)
        self.ref_input_fcs = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, in_dim),
        )
        if getattr(config, "ref_input_zero_init", False):
            nn.init.zeros_(self.ref_input_fcs[-1].weight)
            nn.init.zeros_(self.ref_input_fcs[-1].bias)
        self.ref_input_bbox_fcs.train()
        self.ref_input_fcs.train()
        for param in self.ref_input_bbox_fcs.parameters():
            param.requires_grad = True
        for param in self.ref_input_fcs.parameters():
            param.requires_grad = True
        if not hasattr(config, "ref_reconstruction_loss_weight"):
            config.ref_reconstruction_loss_weight = 1.0
        if not hasattr(config, "ref_prompt_mode"):
            config.ref_prompt_mode = "concat"
        if not hasattr(config, "ref_injection_mode"):
            config.ref_injection_mode = "replace"
        if not hasattr(config, "ref_input_scale"):
            config.ref_input_scale = 1.0
        if not hasattr(config, "ref_prompt_max_norm"):
            config.ref_prompt_max_norm = 20.0
        if not hasattr(config, "sam_prompt_embed_clamp"):
            config.sam_prompt_embed_clamp = 50.0
        if not hasattr(config, "counterfactual_rank_loss_weight"):
            config.counterfactual_rank_loss_weight = 0.0
        if not hasattr(config, "counterfactual_rank_margin"):
            config.counterfactual_rank_margin = 0.05


class LisaModel(LisaMetaModel, LlavaLlamaModel):
    def __init__(
        self,
        config,
        **kwargs,
    ):
        super(LisaModel, self).__init__(config, **kwargs)

        self.config.use_cache = False
        self.config.vision_tower = self.config.mm_vision_tower
        self.config.mm_vision_select_feature = "patch"
        self.config.image_aspect_ratio = "square"
        self.config.image_grid_pinpoints = None
        self.config.tune_mm_mlp_adapter = False
        self.config.freeze_mm_mlp_adapter = True
        self.config.pretrain_mm_mlp_adapter = None
        self.config.mm_use_im_patch_token = False


class LISAForCausalLM(LlavaLlamaForCausalLM):
    def __init__(
        self,
        config,
        **kwargs,
    ):
        self.ce_loss_weight = kwargs.pop("ce_loss_weight", None)
        self.dice_loss_weight = kwargs.pop("dice_loss_weight", None)
        self.bce_loss_weight = kwargs.pop("bce_loss_weight", None)
        if not hasattr(config, "train_mask_decoder"):
            config.mm_use_im_start_end = kwargs.pop("use_mm_start_end", True)
            config.mm_vision_tower = kwargs.get(
                "vision_tower", "openai/clip-vit-large-patch14"
            )
        else:
            config.mm_vision_tower = config.vision_tower
            
        self.seg_token_idx = kwargs.pop(
            "seg_token_idx", getattr(config, "seg_token_idx", None)
        )
        if self.seg_token_idx is None:
            raise ValueError("seg_token_idx must be provided")
        config.seg_token_idx = self.seg_token_idx
        self.ref_token_idx = kwargs.pop(
            "ref_token_idx", getattr(config, "ref_token_idx", -1)
        )
        config.ref_token_idx = self.ref_token_idx
        config.ref_reconstruction_loss_weight = kwargs.pop(
            "ref_reconstruction_loss_weight",
            getattr(config, "ref_reconstruction_loss_weight", 1.0),
        )
        config.ref_prompt_mode = kwargs.pop(
            "ref_prompt_mode", getattr(config, "ref_prompt_mode", "concat")
        )
        config.ref_injection_mode = kwargs.pop(
            "ref_injection_mode", getattr(config, "ref_injection_mode", "replace")
        )
        config.ref_input_scale = kwargs.pop(
            "ref_input_scale", getattr(config, "ref_input_scale", 1.0)
        )
        config.ref_context_init_scale = kwargs.pop(
            "ref_context_init_scale", getattr(config, "ref_context_init_scale", 1.0)
        )
        config.ref_prompt_max_norm = kwargs.pop(
            "ref_prompt_max_norm", getattr(config, "ref_prompt_max_norm", 20.0)
        )
        config.sam_prompt_embed_clamp = kwargs.pop(
            "sam_prompt_embed_clamp", getattr(config, "sam_prompt_embed_clamp", 50.0)
        )
        config.ref_input_zero_init = kwargs.pop(
            "ref_input_zero_init", getattr(config, "ref_input_zero_init", False)
        )
        config.counterfactual_rank_loss_weight = kwargs.pop(
            "counterfactual_rank_loss_weight",
            getattr(config, "counterfactual_rank_loss_weight", 0.0),
        )
        config.counterfactual_rank_margin = kwargs.pop(
            "counterfactual_rank_margin",
            getattr(config, "counterfactual_rank_margin", 0.05),
        )

        super().__init__(config)

        self.model = LisaModel(config, **kwargs)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_visual_embs(self, pixel_values: torch.FloatTensor):
        with torch.no_grad():
            image_embeddings_list = []
            for i in range(pixel_values.shape[0]):
                torch.cuda.empty_cache()
                image_embeddings = self.model.visual_model.image_encoder(
                    pixel_values[i].unsqueeze(0)
                )
                image_embeddings_list.append(image_embeddings)
            torch.cuda.empty_cache()
            image_embeddings = torch.cat(image_embeddings_list, 0)
        return image_embeddings

    def forward(self, **kwargs):
        if "past_key_values" in kwargs:
            return super().forward(**kwargs)
        return self.model_forward(**kwargs)

    def model_forward(
        self,
        images: torch.FloatTensor,
        images_clip: torch.FloatTensor,
        input_ids: torch.LongTensor,
        labels: torch.LongTensor,
        attention_masks: torch.LongTensor,
        offset: torch.LongTensor,
        masks_list: List[torch.FloatTensor],
        label_list: List[torch.Tensor],
        resize_list: List[tuple],
        ref_masks_list: List[torch.FloatTensor] = None,
        ref_bboxes_list: List[torch.FloatTensor] = None,
        ref_valids_list: List[torch.FloatTensor] = None,
        ref_images_clip_list: List[torch.FloatTensor] = None,
        mask_weights_list: List[torch.FloatTensor] = None,
        spatial_memory_boxes_list: List[torch.FloatTensor] = None,
        memory_local_features_list = None,
        inference: bool = False,
        **kwargs,
    ):
        image_embeddings = self.get_visual_embs(images)
        batch_size = image_embeddings.shape[0]
        assert batch_size == len(offset) - 1

        ref_input_embeddings = self.build_ref_input_embeddings(
            input_ids,
            offset,
            ref_images_clip_list,
            ref_bboxes_list,
            ref_valids_list,
        )

        seg_token_mask = input_ids[:, 1:] == self.seg_token_idx
        seg_token_mask = torch.cat(
            [
                seg_token_mask,
                torch.zeros((seg_token_mask.shape[0], 1)).bool().cuda(),
            ],
            dim=1,
        )
        # hack for IMAGE_TOKEN_INDEX (we suppose that there is only one image, and it is in the front)
        seg_token_mask = torch.cat(
            [torch.zeros((seg_token_mask.shape[0], 255)).bool().cuda(), seg_token_mask],
            dim=1,
        )

        if inference:
            n_batch = 1
            length = input_ids.shape[0]
            assert images_clip.shape[0] == 1
            images_clip_extend = images_clip.expand(length, -1, -1, -1).contiguous()

            output_hidden_states = []
            for i in range(n_batch):
                start_i, end_i = i * length, min((i + 1) * length, input_ids.shape[0])
                output_i = super().forward(
                    images=images_clip_extend[: end_i - start_i],
                    attention_mask=attention_masks[start_i:end_i],
                    input_ids=input_ids[start_i:end_i],
                    ref_input_embeddings=(
                        ref_input_embeddings[start_i:end_i]
                        if ref_input_embeddings is not None
                        else None
                    ),
                    ref_token_idx=self.ref_token_idx,
                    ref_injection_mode=getattr(
                        self.config, "ref_injection_mode", "replace"
                    ),
                    ref_input_scale=getattr(self.config, "ref_input_scale", 1.0),
                    output_hidden_states=True,
                )
                output_hidden_states.append(output_i.hidden_states)
                torch.cuda.empty_cache()

            output_hidden_states_list = []
            output_hidden_states_level = torch.cat(output_hidden_states, dim=0)
            output_hidden_states_list.append(output_hidden_states_level)
            output_hidden_states = output_hidden_states_list
            output = None

        else:
            images_clip_list = []
            for i in range(len(offset) - 1):
                start_i, end_i = offset[i], offset[i + 1]
                images_clip_i = (
                    images_clip[i]
                    .unsqueeze(0)
                    .expand(end_i - start_i, -1, -1, -1)
                    .contiguous()
                )
                images_clip_list.append(images_clip_i)
            images_clip = torch.cat(images_clip_list, dim=0)

            output = super().forward(
                images=images_clip,
                attention_mask=attention_masks,
                input_ids=input_ids,
                labels=labels,
                ref_input_embeddings=ref_input_embeddings,
                ref_token_idx=self.ref_token_idx,
                ref_injection_mode=getattr(
                    self.config, "ref_injection_mode", "replace"
                ),
                ref_input_scale=getattr(self.config, "ref_input_scale", 1.0),
                output_hidden_states=True,
            )
            output_hidden_states = output.hidden_states

        hidden_states = []

        assert len(self.model.text_hidden_fcs) == 1
        hidden_states.append(self.model.text_hidden_fcs[0](output_hidden_states[-1]))

        last_hidden_state = torch.stack(hidden_states, dim=-1).sum(dim=-1)
        pred_embeddings = last_hidden_state[seg_token_mask]
        seg_token_counts = seg_token_mask.int().sum(-1)  # [bs, ]

        ref_hidden_by_row = None
        if self.ref_token_idx is not None and self.ref_token_idx >= 0:
            ref_token_mask = input_ids[:, 1:] == self.ref_token_idx
            ref_token_mask = torch.cat(
                [
                    ref_token_mask,
                    torch.zeros((ref_token_mask.shape[0], 1)).bool().cuda(),
                ],
                dim=1,
            )
            ref_token_mask = torch.cat(
                [torch.zeros((ref_token_mask.shape[0], 255)).bool().cuda(), ref_token_mask],
                dim=1,
            )
            ref_hidden_state = self.model.ref_hidden_fcs[0](output_hidden_states[-1])
            ref_hidden_rows = []
            for row_idx in range(ref_token_mask.shape[0]):
                row_mask = ref_token_mask[row_idx]
                if row_mask.any():
                    ref_hidden_rows.append(ref_hidden_state[row_idx][row_mask][-1])
                else:
                    ref_hidden_rows.append(torch.zeros(ref_hidden_state.shape[-1], device=ref_hidden_state.device, dtype=ref_hidden_state.dtype))
            ref_hidden_by_row = torch.stack(ref_hidden_rows, dim=0)

        seg_token_offset = seg_token_counts.cumsum(-1)
        seg_token_offset = torch.cat(
            [torch.zeros(1).long().cuda(), seg_token_offset], dim=0
        )

        seg_token_offset = seg_token_offset[offset]

        pred_embeddings_ = []
        ref_hidden_embeddings_ = []
        for i in range(len(seg_token_offset) - 1):
            start_i, end_i = seg_token_offset[i], seg_token_offset[i + 1]
            pred_embeddings_.append(pred_embeddings[start_i:end_i])
            if ref_hidden_by_row is not None:
                row_start, row_end = offset[i], offset[i + 1]
                ref_hidden_embeddings_.append(ref_hidden_by_row[row_start:row_end])
            else:
                ref_hidden_embeddings_.append(None)
        pred_embeddings = pred_embeddings_

        ref_pred_embeddings = self.get_ref_token_embeddings(
            last_hidden_state,
            input_ids,
            offset,
        )
        ref_prompt_embeddings = self.build_ref_prompt_embeddings(
            pred_embeddings,
            ref_hidden_embeddings_,
            ref_masks_list,
            ref_bboxes_list,
            ref_valids_list,
        )

        multimask_output = False
        pred_masks = []
        for i in range(len(pred_embeddings)):
            text_embeds = pred_embeddings[i].unsqueeze(1)
            if (
                ref_prompt_embeddings is not None
                and ref_prompt_embeddings[i] is not None
            ):
                ref_text_embeds = ref_prompt_embeddings[i].unsqueeze(1)
                if getattr(self.config, "ref_prompt_mode", "concat") == "concat":
                    text_embeds = torch.cat([ref_text_embeds, text_embeds], dim=1)
                elif getattr(self.config, "ref_prompt_mode", "concat") == "add":
                    text_embeds = text_embeds + ref_text_embeds
                else:
                    raise ValueError(
                        "Unsupported ref_prompt_mode: {}".format(
                            getattr(self.config, "ref_prompt_mode", "concat")
                        )
                    )
            prompt_clamp = float(getattr(self.config, "sam_prompt_embed_clamp", 50.0))
            text_embeds = torch.nan_to_num(text_embeds)
            if prompt_clamp > 0:
                text_embeds = text_embeds.clamp(-prompt_clamp, prompt_clamp)
            (
                sparse_embeddings,
                dense_embeddings,
            ) = self.model.visual_model.prompt_encoder(
                points=None,
                boxes=(spatial_memory_boxes_list[i].to(device=text_embeds.device)
                       if spatial_memory_boxes_list is not None else None),
                masks=None,
                text_embeds=text_embeds,
            )
            sparse_embeddings = sparse_embeddings.to(pred_embeddings[i].dtype)
            if memory_local_features_list is None:
                low_res_masks, iou_predictions = self.model.visual_model.mask_decoder(
                    image_embeddings=image_embeddings[i].unsqueeze(0),
                    image_pe=self.model.visual_model.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=multimask_output,
                )
            else:
                evidence = memory_local_features_list[i]
                # Preserve the original prompt batch shape (BF16 GEMMs differ
                # for singleton rows). Keep baseline miner outputs; each fused
                # invocation contributes only its own helmet output.
                baseline_masks, _ = self.model.visual_model.mask_decoder(
                    image_embeddings=image_embeddings[i].unsqueeze(0),
                    image_pe=self.model.visual_model.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=multimask_output,
                )
                row_masks = list(baseline_masks.split(1, dim=0))
                for identity in range(evidence['mapped'].shape[0]):
                    row = 2 * identity + 1
                    features = image_embeddings[i].unsqueeze(0)
                    features = self.memory_dual_scale_adapter(
                        features.float(), evidence['mapped'][identity:identity+1].float(),
                        evidence['gate'][identity:identity+1].float()).to(features.dtype)
                    identity_masks, _ = self.model.visual_model.mask_decoder(
                        image_embeddings=features,
                        image_pe=self.model.visual_model.prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings=sparse_embeddings,
                        dense_prompt_embeddings=dense_embeddings,
                        multimask_output=multimask_output,
                    )
                    row_masks[row] = identity_masks[row:row+1]
                low_res_masks = torch.cat(row_masks, dim=0)
            low_res_masks = torch.nan_to_num(low_res_masks)
            pred_mask = self.model.visual_model.postprocess_masks(
                low_res_masks,
                input_size=resize_list[i],
                original_size=label_list[i].shape,
            )
            pred_mask = torch.nan_to_num(pred_mask)
            pred_masks.append(pred_mask[:, 0])

        ref_pred_masks = []
        for i in range(len(ref_pred_embeddings)):
            if ref_pred_embeddings[i].numel() == 0:
                ref_pred_masks.append(None)
                continue
            (
                sparse_embeddings,
                dense_embeddings,
            ) = self.model.visual_model.prompt_encoder(
                points=None,
                boxes=None,
                masks=None,
                text_embeds=torch.nan_to_num(
                    ref_pred_embeddings[i].unsqueeze(1)
                ).clamp(
                    -float(getattr(self.config, "sam_prompt_embed_clamp", 50.0)),
                    float(getattr(self.config, "sam_prompt_embed_clamp", 50.0)),
                ),
            )
            sparse_embeddings = sparse_embeddings.to(ref_pred_embeddings[i].dtype)
            low_res_masks, _ = self.model.visual_model.mask_decoder(
                image_embeddings=image_embeddings[i].unsqueeze(0),
                image_pe=self.model.visual_model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=multimask_output,
            )
            low_res_masks = torch.nan_to_num(low_res_masks)
            ref_pred_mask = self.model.visual_model.postprocess_masks(
                low_res_masks,
                input_size=resize_list[i],
                original_size=label_list[i].shape,
            )
            ref_pred_mask = torch.nan_to_num(ref_pred_mask)
            ref_pred_masks.append(ref_pred_mask[:, 0])

        model_output = output
        gt_masks = masks_list

        if inference:
            return {
                "pred_masks": pred_masks,
                "gt_masks": gt_masks,
            }

        output = model_output.logits

        ce_loss = model_output.loss
        ce_loss = ce_loss * self.ce_loss_weight
        mask_bce_loss = 0
        mask_dice_loss = 0
        ref_mask_bce_loss = 0
        ref_mask_dice_loss = 0
        rank_loss_sum = 0
        num_masks = 0
        num_ref_masks = 0
        num_rank_terms = 0
        rank_weight = float(
            getattr(self.config, "counterfactual_rank_loss_weight", 0.0)
        )
        rank_margin = float(getattr(self.config, "counterfactual_rank_margin", 0.05))
        for batch_idx in range(len(pred_masks)):
            gt_mask = gt_masks[batch_idx]
            pred_mask = pred_masks[batch_idx]
            if mask_weights_list is not None:
                mask_weights = mask_weights_list[batch_idx].to(
                    device=pred_mask.device, dtype=pred_mask.dtype
                )
            else:
                mask_weights = torch.ones(
                    (gt_mask.shape[0],), device=pred_mask.device, dtype=pred_mask.dtype
                )

            assert (
                gt_mask.shape[0] == pred_mask.shape[0]
            ), "gt_mask.shape: {}, pred_mask.shape: {}".format(
                gt_mask.shape, pred_mask.shape
            )
            mask_bce_loss += (
                sigmoid_ce_loss_per_mask(pred_mask, gt_mask) * mask_weights
            ).sum()
            mask_dice_loss += (
                dice_loss_per_mask(pred_mask, gt_mask) * mask_weights
            ).sum()
            num_masks += mask_weights.sum()

            if rank_weight > 0 and gt_mask.shape[0] >= 4 and gt_mask.shape[0] % 2 == 0:
                helmet_pred = pred_mask[1::2]
                helmet_gt = gt_mask[1::2]
                if helmet_pred.shape[0] >= 2:
                    helmet_iou = soft_iou_matrix(helmet_pred, helmet_gt)
                    diag = torch.diagonal(helmet_iou)
                    offdiag = helmet_iou.masked_fill(
                        torch.eye(
                            helmet_iou.shape[0],
                            device=helmet_iou.device,
                            dtype=torch.bool,
                        ),
                        -1.0,
                    )
                    wrong = offdiag.max(dim=1).values
                    rank_loss_sum += F.relu(rank_margin + wrong - diag).sum()
                    num_rank_terms += helmet_pred.shape[0]

            if (
                ref_masks_list is not None
                and ref_valids_list is not None
                and ref_pred_masks[batch_idx] is not None
            ):
                ref_valids = ref_valids_list[batch_idx].to(
                    device=pred_mask.device, dtype=pred_mask.dtype
                )
                valid_idx = torch.where(ref_valids > 0)[0]
                if valid_idx.numel() > 0:
                    gt_ref_mask = ref_masks_list[batch_idx].to(
                        device=pred_mask.device, dtype=pred_mask.dtype
                    )[valid_idx]
                    pred_ref_mask = ref_pred_masks[batch_idx]
                    n_ref = min(gt_ref_mask.shape[0], pred_ref_mask.shape[0])
                    if n_ref > 0:
                        gt_ref_mask = gt_ref_mask[:n_ref]
                        pred_ref_mask = pred_ref_mask[:n_ref]
                        ref_mask_bce_loss += sigmoid_ce_loss_per_mask(
                            pred_ref_mask, gt_ref_mask
                        ).sum()
                        ref_mask_dice_loss += dice_loss_per_mask(
                            pred_ref_mask, gt_ref_mask
                        ).sum()
                        num_ref_masks += n_ref

        mask_bce_loss = self.bce_loss_weight * mask_bce_loss / (num_masks + 1e-8)
        mask_dice_loss = self.dice_loss_weight * mask_dice_loss / (num_masks + 1e-8)
        mask_loss = mask_bce_loss + mask_dice_loss
        if num_ref_masks > 0:
            ref_weight = getattr(self.config, "ref_reconstruction_loss_weight", 1.0)
            ref_mask_bce_loss = self.bce_loss_weight * ref_mask_bce_loss / (
                num_ref_masks + 1e-8
            )
            ref_mask_dice_loss = self.dice_loss_weight * ref_mask_dice_loss / (
                num_ref_masks + 1e-8
            )
            ref_mask_loss = ref_weight * (ref_mask_bce_loss + ref_mask_dice_loss)
        else:
            ref_mask_loss = mask_loss.new_tensor(0.0)

        if num_rank_terms > 0:
            counterfactual_rank_loss = (
                mask_loss.new_tensor(rank_weight)
                * rank_loss_sum
                / (num_rank_terms + 1e-8)
            )
        else:
            counterfactual_rank_loss = mask_loss.new_tensor(0.0)

        loss = ce_loss + mask_loss + ref_mask_loss + counterfactual_rank_loss

        return {
            "loss": loss,
            "ce_loss": ce_loss,
            "mask_bce_loss": mask_bce_loss,
            "mask_dice_loss": mask_dice_loss,
            "mask_loss": mask_loss,
            "ref_mask_loss": ref_mask_loss,
            "counterfactual_rank_loss": counterfactual_rank_loss,
        }

    def build_ref_input_embeddings(
        self,
        input_ids: torch.Tensor,
        offset: torch.Tensor,
        ref_images_clip_list: List[torch.Tensor],
        ref_bboxes_list: List[torch.Tensor],
        ref_valids_list: List[torch.Tensor],
    ):
        if (
            ref_images_clip_list is None
            or ref_bboxes_list is None
            or ref_valids_list is None
            or self.ref_token_idx is None
            or self.ref_token_idx < 0
        ):
            return None

        row_embeddings = []
        hidden_size = self.config.hidden_size
        max_refs = 0
        for image_idx in range(len(offset) - 1):
            ref_valids = ref_valids_list[image_idx].to(device=input_ids.device)
            valid_idx = torch.where(ref_valids > 0)[0]
            if valid_idx.numel() > 0:
                ref_images = ref_images_clip_list[image_idx][valid_idx].to(
                    device=input_ids.device,
                    dtype=next(self.model.ref_input_fcs.parameters()).dtype,
                )
                ref_bboxes = ref_bboxes_list[image_idx][valid_idx].to(
                    device=input_ids.device,
                    dtype=next(self.model.ref_input_fcs.parameters()).dtype,
                )
                crop_features = self.encode_images(ref_images).mean(dim=1)
                bbox_features = self.model.ref_input_bbox_fcs(ref_bboxes)
                valid_embeddings = self.model.ref_input_fcs(
                    crop_features + bbox_features
                )
            else:
                valid_embeddings = torch.zeros(
                    0,
                    hidden_size,
                    device=input_ids.device,
                    dtype=next(self.model.ref_input_fcs.parameters()).dtype,
                )

            cursor = 0
            for row_idx in range(int(offset[image_idx]), int(offset[image_idx + 1])):
                n_refs = int((input_ids[row_idx] == self.ref_token_idx).sum().item())
                if n_refs > 0 and valid_embeddings.shape[0] > cursor:
                    row_ref = valid_embeddings[cursor : cursor + n_refs]
                    cursor += n_refs
                    if row_ref.shape[0] < n_refs:
                        row_ref = torch.cat(
                            [
                                row_ref,
                                torch.zeros(
                                    n_refs - row_ref.shape[0],
                                    hidden_size,
                                    device=input_ids.device,
                                    dtype=valid_embeddings.dtype,
                                ),
                            ],
                            dim=0,
                        )
                else:
                    row_ref = torch.zeros(
                        n_refs,
                        hidden_size,
                        device=input_ids.device,
                        dtype=valid_embeddings.dtype,
                    )
                max_refs = max(max_refs, row_ref.shape[0])
                row_embeddings.append(row_ref)

        if max_refs == 0:
            return None
        padded = []
        dtype = next(self.model.ref_input_fcs.parameters()).dtype
        for row_ref in row_embeddings:
            row_ref = row_ref.to(dtype=dtype)
            if row_ref.shape[0] < max_refs:
                row_ref = torch.cat(
                    [
                        row_ref,
                        torch.zeros(
                            max_refs - row_ref.shape[0],
                            hidden_size,
                            device=input_ids.device,
                            dtype=dtype,
                        ),
                    ],
                    dim=0,
                )
            padded.append(row_ref)
        return torch.stack(padded, dim=0)

    def build_ref_prompt_embeddings(
        self,
        pred_embeddings: List[torch.Tensor],
        ref_hidden_embeddings: List[torch.Tensor],
        ref_masks_list: List[torch.Tensor],
        ref_bboxes_list: List[torch.Tensor],
        ref_valids_list: List[torch.Tensor],
    ):
        if (
            ref_masks_list is None
            or ref_bboxes_list is None
            or ref_valids_list is None
            or ref_hidden_embeddings is None
        ):
            return None

        output = []
        pool_size = getattr(self.config, "ref_mask_pool_size", 16)
        scale = self.model.ref_embedding_scale.to(
            device=pred_embeddings[0].device, dtype=pred_embeddings[0].dtype
        )
        for i, pred in enumerate(pred_embeddings):
            if pred.numel() == 0:
                output.append(None)
                continue

            n = pred.shape[0]
            ref_masks = ref_masks_list[i].to(device=pred.device, dtype=pred.dtype)
            ref_bboxes = ref_bboxes_list[i].to(device=pred.device, dtype=pred.dtype)
            ref_valids = ref_valids_list[i].to(
                device=pred.device, dtype=pred.dtype
            ).view(-1, 1)

            if ref_masks.shape[0] < n:
                pad_n = n - ref_masks.shape[0]
                ref_masks = torch.cat(
                    [
                        ref_masks,
                        torch.zeros(
                            pad_n,
                            *ref_masks.shape[1:],
                            device=pred.device,
                            dtype=pred.dtype,
                        ),
                    ],
                    dim=0,
                )
                ref_bboxes = torch.cat(
                    [
                        ref_bboxes,
                        torch.zeros(pad_n, 4, device=pred.device, dtype=pred.dtype),
                    ],
                    dim=0,
                )
                ref_valids = torch.cat(
                    [
                        ref_valids,
                        torch.zeros(pad_n, 1, device=pred.device, dtype=pred.dtype),
                    ],
                    dim=0,
                )
            elif ref_masks.shape[0] > n:
                ref_masks = ref_masks[:n]
                ref_bboxes = ref_bboxes[:n]
                ref_valids = ref_valids[:n]

            mask_feat = F.interpolate(
                ref_masks.unsqueeze(1),
                size=(pool_size, pool_size),
                mode="bilinear",
                align_corners=False,
            ).flatten(1)
            ref_visual = self.model.ref_visual_fcs(
                torch.cat([mask_feat, ref_bboxes], dim=-1)
            )
            ref_hidden = ref_hidden_embeddings[i]
            if ref_hidden is None:
                ref_hidden = torch.zeros_like(pred)
            else:
                ref_hidden = ref_hidden.to(device=pred.device, dtype=pred.dtype)
                if ref_hidden.shape[0] == n:
                    pass
                else:
                    ref_hidden_full = torch.zeros_like(pred)
                    valid_idx = torch.where(ref_valids.view(-1) > 0)[0]
                    if ref_hidden.shape[0] == valid_idx.numel():
                        ref_hidden_full[valid_idx] = ref_hidden
                    elif ref_hidden.shape[0] == 1 and valid_idx.numel() > 0:
                        ref_hidden_full[valid_idx] = ref_hidden[0]
                    ref_hidden = ref_hidden_full
            ref_context = torch.nan_to_num(ref_hidden + ref_visual)
            max_norm = float(getattr(self.config, "ref_prompt_max_norm", 20.0))
            if max_norm > 0:
                ref_norm = ref_context.norm(dim=-1, keepdim=True).clamp_min(1e-6)
                ref_context = ref_context * torch.clamp(max_norm / ref_norm, max=1.0)
            output.append(scale * ref_context * ref_valids)
        return output

    def _build_expanded_token_mask(self, input_ids, token_idx, target_len, shifted=False):
        vision_tower = self.get_model().get_vision_tower()
        image_token_len = getattr(vision_tower, "num_patches", 256)
        masks = []
        for row in input_ids:
            pieces = []
            token_row = row[1:] if shifted else row
            for token in token_row:
                if int(token.item()) == -200:
                    pieces.append(
                        torch.zeros(
                            image_token_len,
                            dtype=torch.bool,
                            device=row.device,
                        )
                    )
                else:
                    pieces.append((token == token_idx).view(1))
            mask = torch.cat(pieces, dim=0)
            if shifted:
                mask = torch.cat(
                    [mask, torch.zeros(1, dtype=torch.bool, device=row.device)],
                    dim=0,
                )
            if mask.shape[0] < target_len:
                mask = torch.cat(
                    [
                        mask,
                        torch.zeros(
                            target_len - mask.shape[0],
                            dtype=torch.bool,
                            device=row.device,
                        ),
                    ],
                    dim=0,
                )
            elif mask.shape[0] > target_len:
                mask = mask[:target_len]
            masks.append(mask)
        return torch.stack(masks, dim=0)

    def get_ref_token_embeddings(self, last_hidden_state, input_ids, offset):
        if self.ref_token_idx is None or self.ref_token_idx < 0:
            return [
                torch.zeros(
                    0,
                    last_hidden_state.shape[-1],
                    device=last_hidden_state.device,
                    dtype=last_hidden_state.dtype,
                )
                for _ in range(len(offset) - 1)
            ]
        ref_token_mask = self._build_expanded_token_mask(
            input_ids,
            self.ref_token_idx,
            last_hidden_state.shape[1],
            shifted=False,
        )
        ref_embeddings = last_hidden_state[ref_token_mask]
        ref_token_counts = ref_token_mask.int().sum(-1)
        ref_token_offset = ref_token_counts.cumsum(-1)
        ref_token_offset = torch.cat(
            [torch.zeros(1).long().cuda(), ref_token_offset], dim=0
        )
        ref_token_offset = ref_token_offset[offset]
        ref_embeddings_by_image = []
        for i in range(len(ref_token_offset) - 1):
            start_i, end_i = ref_token_offset[i], ref_token_offset[i + 1]
            ref_embeddings_by_image.append(ref_embeddings[start_i:end_i])
        return ref_embeddings_by_image

    def apply_ref_embeddings(
        self,
        pred_embeddings: List[torch.Tensor],
        ref_hidden_embeddings: List[torch.Tensor],
        ref_masks_list: List[torch.Tensor],
        ref_bboxes_list: List[torch.Tensor],
        ref_valids_list: List[torch.Tensor],
    ):
        output = []
        pool_size = getattr(self.config, "ref_mask_pool_size", 16)
        scale = self.model.ref_embedding_scale.to(pred_embeddings[0].dtype)
        for i, pred in enumerate(pred_embeddings):
            if pred.numel() == 0:
                output.append(pred)
                continue
            ref_masks = ref_masks_list[i].to(device=pred.device, dtype=pred.dtype)
            ref_bboxes = ref_bboxes_list[i].to(device=pred.device, dtype=pred.dtype)
            ref_valids = ref_valids_list[i].to(device=pred.device, dtype=pred.dtype).view(-1, 1)
            n = pred.shape[0]
            if ref_masks.shape[0] != n:
                ref_masks = ref_masks[:n]
                ref_bboxes = ref_bboxes[:n]
                ref_valids = ref_valids[:n]
            mask_feat = F.interpolate(
                ref_masks.unsqueeze(1),
                size=(pool_size, pool_size),
                mode="bilinear",
                align_corners=False,
            ).flatten(1)
            ref_visual = self.model.ref_visual_fcs(torch.cat([mask_feat, ref_bboxes], dim=-1))
            ref_hidden = ref_hidden_embeddings[i]
            if ref_hidden is None or ref_hidden.shape[0] != n:
                ref_hidden = torch.zeros_like(pred)
            else:
                ref_hidden = ref_hidden.to(device=pred.device, dtype=pred.dtype)
            ref_context = (ref_hidden + ref_visual) * ref_valids
            output.append(pred + scale * ref_context)
        return output

    def evaluate(
        self,
        images_clip,
        images,
        input_ids,
        resize_list,
        original_size_list,
        max_new_tokens=32,
        tokenizer=None,
    ):
        with torch.no_grad():
            outputs = self.generate(
                images=images_clip,
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                max_new_tokens=max_new_tokens,
                num_beams=1,
                output_hidden_states=False,
                return_dict_in_generate=True,
            )
            output_ids = outputs.sequences
            forward_outputs = super().forward(
                images=images_clip,
                input_ids=output_ids,
                attention_mask=torch.ones_like(output_ids),
                output_hidden_states=True,
            )
            output_hidden_states = forward_outputs.hidden_states[-1]
            if output_hidden_states.dim() == 2:
                output_hidden_states = output_hidden_states.unsqueeze(0)

            seg_token_mask = output_ids[:, 1:] == self.seg_token_idx
            seg_token_mask = torch.cat(
                [
                    seg_token_mask,
                    torch.zeros((seg_token_mask.shape[0], 1)).bool().cuda(),
                ],
                dim=1,
            )
            # hack for IMAGE_TOKEN_INDEX (we suppose that there is only one image, and it is in the front)
            seg_token_mask = torch.cat(
                [
                    torch.zeros((seg_token_mask.shape[0], 255)).bool().cuda(),
                    seg_token_mask,
                ],
                dim=1,
            )

            hidden_states = []

            assert len(self.model.text_hidden_fcs) == 1
            hidden_states.append(self.model.text_hidden_fcs[0](output_hidden_states))

            last_hidden_state = torch.stack(hidden_states, dim=-1).sum(dim=-1)
            pred_embeddings = last_hidden_state[seg_token_mask]

            seg_token_counts = seg_token_mask.int().sum(-1)  # [bs, ]
            seg_token_offset = seg_token_counts.cumsum(-1)
            seg_token_offset = torch.cat(
                [torch.zeros(1).long().cuda(), seg_token_offset], dim=0
            )

            pred_embeddings_ = []
            for i in range(len(seg_token_offset) - 1):
                start_i, end_i = seg_token_offset[i], seg_token_offset[i + 1]
                pred_embeddings_.append(pred_embeddings[start_i:end_i])
            pred_embeddings = pred_embeddings_

            image_embeddings = self.get_visual_embs(images)

            multimask_output = False
            pred_masks = []
            for i in range(len(pred_embeddings)):
                (
                    sparse_embeddings,
                    dense_embeddings,
                ) = self.model.visual_model.prompt_encoder(
                    points=None,
                    boxes=None,
                    masks=None,
                    text_embeds=pred_embeddings[i].unsqueeze(1),
                )

                sparse_embeddings = sparse_embeddings.to(pred_embeddings[i].dtype)
                low_res_masks, iou_predictions = self.model.visual_model.mask_decoder(
                    image_embeddings=image_embeddings[i].unsqueeze(0),
                    image_pe=self.model.visual_model.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=multimask_output,
                )
                pred_mask = self.model.visual_model.postprocess_masks(
                    low_res_masks,
                    input_size=resize_list[i],
                    original_size=original_size_list[i],
                )
                pred_masks.append(pred_mask[:, 0])

        return output_ids, pred_masks
