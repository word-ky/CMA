import argparse
import glob
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import transformers
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer

from model.LISA import LISAForCausalLM
from utils.utils import DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN


def get_added_token_id(tokenizer, token):
    token_id = tokenizer.convert_tokens_to_ids(token)
    if token_id is None or token_id < 0 or token_id == tokenizer.unk_token_id:
        token_id = tokenizer(token, add_special_tokens=False).input_ids[-1]
    return token_id


def parse_args(args):
    parser = argparse.ArgumentParser(
        description="merge lora weights and save model with hf format"
    )
    parser.add_argument(
        "--version", default="liuhaotian/llava-llama-2-13b-chat-lightning-preview"
    )
    parser.add_argument("--vis_save_path", default="./vis_output", type=str)
    parser.add_argument(
        "--precision",
        default="bf16",
        type=str,
        choices=["fp32", "bf16", "fp16"],
        help="precision for inference",
    )
    parser.add_argument("--vision_pretrained", default="PATH_TO_SAM_ViT-H", type=str)
    parser.add_argument("--out_dim", default=256, type=int)
    parser.add_argument("--image_size", default=1024, type=int, help="image size")
    parser.add_argument("--model_max_length", default=512, type=int)
    parser.add_argument(
        "--vision-tower", default="openai/clip-vit-large-patch14", type=str
    )
    parser.add_argument("--lora_r", default=8, type=int)
    parser.add_argument("--lora_alpha", default=16, type=int)
    parser.add_argument("--lora_dropout", default=0.05, type=float)
    parser.add_argument("--lora_target_modules", default="q_proj,v_proj", type=str)
    parser.add_argument("--ref_reconstruction_loss_weight", default=1.0, type=float)
    parser.add_argument(
        "--ref_prompt_mode", default="concat", choices=["concat", "add"], type=str
    )
    parser.add_argument(
        "--ref_injection_mode",
        default="replace",
        choices=["replace", "add", "gated_add"],
        type=str,
    )
    parser.add_argument("--ref_input_scale", default=1.0, type=float)
    parser.add_argument("--ref_context_init_scale", default=1.0, type=float)
    parser.add_argument("--ref_prompt_max_norm", default=20.0, type=float)
    parser.add_argument("--sam_prompt_embed_clamp", default=50.0, type=float)
    parser.add_argument("--ref_input_zero_init", action="store_true", default=False)
    parser.add_argument("--counterfactual_rank_loss_weight", default=0.0, type=float)
    parser.add_argument("--counterfactual_rank_margin", default=0.05, type=float)
    parser.add_argument("--local-rank", default=0, type=int, help="node rank")
    parser.add_argument("--train_mask_decoder", action="store_true", default=True)
    parser.add_argument("--use_mm_start_end", action="store_true", default=True)
    parser.add_argument(
        "--conv_type",
        default="llava_v1",
        type=str,
        choices=["llava_v1", "llava_llama_2"],
    )
    parser.add_argument("--weight", default="", type=str, required=True)
    parser.add_argument("--save_path", default="./lisa_model", type=str, required=True)
    return parser.parse_args(args)


def main(args):
    args = parse_args(args)
    os.makedirs(args.vis_save_path, exist_ok=True)

    # Create model
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.version,
        cache_dir=None,
        model_max_length=args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.unk_token
    num_added_tokens = tokenizer.add_tokens(["[SEG]", "[REF]"])
    args.seg_token_idx = get_added_token_id(tokenizer, "[SEG]")
    args.ref_token_idx = get_added_token_id(tokenizer, "[REF]")

    if (
        args.use_mm_start_end
        and tokenizer.convert_tokens_to_ids(DEFAULT_IM_START_TOKEN) == tokenizer.unk_token_id
    ):
        args.use_mm_start_end = False

    if args.use_mm_start_end:
        tokenizer.add_tokens(
            [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True
        )

    model_args = {
        "train_mask_decoder": args.train_mask_decoder,
        "out_dim": args.out_dim,
        "seg_token_idx": args.seg_token_idx,
        "ref_token_idx": args.ref_token_idx,
        "ref_reconstruction_loss_weight": args.ref_reconstruction_loss_weight,
        "ref_prompt_mode": args.ref_prompt_mode,
        "ref_injection_mode": args.ref_injection_mode,
        "ref_input_scale": args.ref_input_scale,
        "ref_context_init_scale": args.ref_context_init_scale,
        "ref_prompt_max_norm": args.ref_prompt_max_norm,
        "sam_prompt_embed_clamp": args.sam_prompt_embed_clamp,
        "ref_input_zero_init": args.ref_input_zero_init,
        "counterfactual_rank_loss_weight": args.counterfactual_rank_loss_weight,
        "counterfactual_rank_margin": args.counterfactual_rank_margin,
        "vision_tower": args.vision_tower,
        "vision_pretrained": args.vision_pretrained,
        "use_mm_start_end": args.use_mm_start_end,
    }

    torch_dtype = torch.float32
    if args.precision == "bf16":
        torch_dtype = torch.bfloat16
    elif args.precision == "fp16":
        torch_dtype = torch.half
    model = LISAForCausalLM.from_pretrained(
        args.version, torch_dtype=torch_dtype, low_cpu_mem_usage=False, **model_args
    )
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    model.get_model().initialize_vision_modules(model.get_model().config)
    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(dtype=torch_dtype)
    if not hasattr(model.get_model(), "visual_model"):
        model.get_model().initialize_lisa_modules(model.get_model().config)

    lora_r = args.lora_r
    if lora_r > 0:

        def find_linear_layers(model, lora_target_modules):
            cls = torch.nn.Linear
            lora_module_names = set()
            for name, module in model.named_modules():
                if (
                    isinstance(module, cls)
                    and all(
                        [
                            x not in name
                            for x in [
                                "visual_model",
                                "vision_tower",
                                "mm_projector",
                                "text_hidden_fcs",
                                "ref_hidden_fcs",
                                "ref_visual_fcs",
                                "ref_input_bbox_fcs",
                                "ref_input_fcs",
                            ]
                        ]
                    )
                    and any([x in name for x in lora_target_modules])
                ):
                    lora_module_names.add(name)
            return sorted(list(lora_module_names))

        lora_alpha = args.lora_alpha
        lora_dropout = args.lora_dropout
        lora_target_modules = find_linear_layers(
            model, args.lora_target_modules.split(",")
        )
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=lora_target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    model.resize_token_embeddings(len(tokenizer))

    state_dict = torch.load(args.weight, map_location="cpu")
    if isinstance(state_dict, dict) and "module" in state_dict:
        state_dict = state_dict["module"]
    model.load_state_dict(state_dict, strict=True)

    model = model.merge_and_unload()
    state_dict = {}
    for k, v in model.state_dict().items():
        if "vision_tower" not in k:
            state_dict[k] = v
    model.save_pretrained(args.save_path, state_dict=state_dict)
    tokenizer.save_pretrained(args.save_path)


if __name__ == "__main__":
    main(sys.argv[1:])
