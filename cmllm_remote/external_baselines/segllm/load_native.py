"""Load the pinned native model for the baseline health check (no inference).

The loading sequence follows upstream llava/train/inference_cli.py at
4593a069f09628ce3a5b46e657f5417fefd7be46. Only the CLIP asset location is
relocated; architecture, checkpoint, dtype and native components are retained.
"""

import argparse
import json
import time
from pathlib import Path


def load_native(checkpoint, clip_path, output_dir):
    import torch
    import transformers
    from llava.train.inference_cli import (
        ModelArguments, DataArguments, TrainingArguments, LlavaLlamaForCausalLM,
    )
    from llava.model import LlavaConfig

    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    argv = [
        '--model_name_or_path', str(checkpoint), '--load', str(checkpoint),
        '--image_folder', str(output_dir), '--mm_use_seg', 'True',
        '--segmentator', 'hipie', '--vision_tower', str(clip_path),
        '--mm_projector_type', 'mlp2x_gelu', '--tune_mm_mlp_adapter', 'False',
        '--mm_vision_select_layer', '-2', '--mm_use_im_start_end', 'False',
        '--mm_vision_select_feature', 'patch', '--mm_use_im_patch_token', 'False',
        '--bf16', 'True', '--lora_enable', 'False', '--split_loading', 'False',
        '--version', 'plain', '--mm_use_gen', 'True', '--output_dir', str(output_dir),
        '--per_device_train_batch_size', '1', '--per_device_eval_batch_size', '1',
        '--model_max_length', '2048', '--dataloader_num_workers', '0',
        '--lazy_preprocess', 'True', '--output_text', '--report_to', 'none',
        '--deepspeed', './scripts/deepspeed_configs/zero2.json',
    ]
    model_args, data_args, training_args = parser.parse_args_into_dataclasses(argv)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        checkpoint, model_max_length=training_args.model_max_length,
        padding_side='right', use_fast=True,
    )
    config = LlavaConfig.from_pretrained(checkpoint)
    original_clip = config.mm_vision_tower
    config.mm_vision_tower = str(clip_path)
    model = LlavaLlamaForCausalLM.from_pretrained(checkpoint, config=config)
    model.eval()
    model.initialize_vision_tokenizer(model_args, tokenizer)
    vision_tower = model.get_vision_tower()
    vision_tower.to(dtype=torch.bfloat16, device=training_args.device)
    model.to(dtype=torch.bfloat16, device=training_args.device)
    data_args.image_processor = vision_tower.image_processor
    data_args.mask_processor = model.get_segmentator().process_images
    data_args.is_multimodal = True
    return model, tokenizer, model_args, data_args, training_args, {
        'argv': argv, 'original_clip_id': original_clip,
        'clip_local_path': str(clip_path),
        'clip_snapshot': str(Path(clip_path).resolve()),
        'transport_changes': ['local checkpoint path', 'local cached CLIP path'],
        'reporting_change': 'report_to=none; no external telemetry',
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--clip-path', required=True)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    import torch

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    model, tokenizer, _, _, _, receipt = load_native(args.checkpoint, args.clip_path, out)
    torch.cuda.synchronize()
    receipt.update({
        'status': 'NATIVE_LOAD_PASSED', 'forward_calls': 0,
        'load_seconds': time.monotonic() - started,
        'peak_load_vram_bytes': torch.cuda.max_memory_allocated(),
        'gpu': torch.cuda.get_device_name(0), 'torch': torch.__version__,
        'parameters': sum(p.numel() for p in model.parameters()),
        'tokenizer_size': len(tokenizer),
    })
    (out / 'native_load_receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
