import json
import os
import random

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from transformers import CLIPImageProcessor

from model.llava import conversation as conversation_lib
from model.segment_anything.utils.transforms import ResizeLongestSide

from .utils import DEFAULT_IMAGE_TOKEN


class MultiRoundRefSegDataset(torch.utils.data.Dataset):
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    img_size = 1024
    ignore_label = 255

    def __init__(
        self,
        jsonl_path,
        tokenizer,
        vision_tower,
        samples_per_epoch=None,
        precision: str = "fp32",
        image_size: int = 1024,
        deterministic: bool = False,
        ref_image_mode: str = None,
        ref_focus_dilate: int = None,
        ref_focus_background: float = None,
    ):
        self.jsonl_path = jsonl_path
        self.tokenizer = tokenizer
        self.precision = precision
        self.image_size = image_size
        self.samples_per_epoch = samples_per_epoch
        self.deterministic = deterministic
        self.ref_image_mode = (
            ref_image_mode or os.environ.get("MR_REF_IMAGE_MODE", "crop")
        ).strip()
        self.ref_focus_dilate = int(
            ref_focus_dilate
            if ref_focus_dilate is not None
            else os.environ.get("MR_REF_FOCUS_DILATE", "15")
        )
        self.ref_focus_background = float(
            ref_focus_background
            if ref_focus_background is not None
            else os.environ.get("MR_REF_FOCUS_BACKGROUND", "0.0")
        )
        valid_ref_image_modes = {"crop", "full_blackout", "full_darken", "full_blur"}
        if self.ref_image_mode not in valid_ref_image_modes:
            raise ValueError(
                "Unsupported MR_REF_IMAGE_MODE '{}'; expected one of {}".format(
                    self.ref_image_mode, sorted(valid_ref_image_modes)
                )
            )
        self.transform = ResizeLongestSide(image_size)
        self.clip_image_processor = CLIPImageProcessor.from_pretrained(vision_tower)

        self.episodes = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.episodes.append(json.loads(line))
        if not self.episodes:
            raise ValueError("No multi-round reference episodes found: {}".format(jsonl_path))
        print(
            "number of mr_ref_seg episodes: ",
            len(self.episodes),
            "from",
            jsonl_path,
            "ref_image_mode=",
            self.ref_image_mode,
        )

    def __len__(self):
        return self.samples_per_epoch if self.samples_per_epoch is not None else len(self.episodes)

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.pixel_mean) / self.pixel_std
        h, w = x.shape[-2:]
        padh = self.img_size - h
        padw = self.img_size - w
        return F.pad(x, (0, padw, 0, padh))

    @staticmethod
    def _read_mask(mask_path, image_hw):
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(mask_path)
        h, w = image_hw
        if mask.shape[:2] != (h, w):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        return (mask > 0).astype(np.float32)

    @staticmethod
    def _norm_bbox(bbox_xyxy, image_hw):
        h, w = image_hw
        if not bbox_xyxy:
            return [0.0, 0.0, 0.0, 0.0]
        x1, y1, x2, y2 = [float(x) for x in bbox_xyxy]
        return [
            max(0.0, min(1.0, x1 / max(w, 1))),
            max(0.0, min(1.0, y1 / max(h, 1))),
            max(0.0, min(1.0, x2 / max(w, 1))),
            max(0.0, min(1.0, y2 / max(h, 1))),
        ]

    @staticmethod
    def _answer():
        return "[SEG]."

    def _build_conversation(self, query, answer, add_image=True):
        conv = conversation_lib.default_conversation.copy()
        conv.messages = []
        question = query.strip()
        if add_image:
            question = DEFAULT_IMAGE_TOKEN + "\n" + question
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], answer)
        return conv.get_prompt()

    def _build_multiround_conversation(self, round1_query, round2_query):
        conv = conversation_lib.default_conversation.copy()
        conv.messages = []
        conv.append_message(
            conv.roles[0],
            DEFAULT_IMAGE_TOKEN + "\n" + round1_query.strip(),
        )
        conv.append_message(conv.roles[1], self._answer())
        conv.append_message(conv.roles[0], "[REF] " + round2_query.strip())
        conv.append_message(conv.roles[1], self._answer())
        return conv.get_prompt()

    @staticmethod
    def _bbox_from_mask(mask):
        ys, xs = np.where(mask > 0)
        if len(xs) == 0 or len(ys) == 0:
            return None
        return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]

    def _make_ref_crop_clip(self, image, mask, bbox_xyxy):
        if bbox_xyxy is None or not bbox_xyxy:
            bbox_xyxy = self._bbox_from_mask(mask)
        if bbox_xyxy is None:
            return torch.zeros(3, 224, 224)

        h, w = image.shape[:2]
        x1, y1, x2, y2 = [int(round(float(v))) for v in bbox_xyxy]
        x1 = max(0, min(w - 1, x1))
        y1 = max(0, min(h - 1, y1))
        x2 = max(x1 + 1, min(w, x2))
        y2 = max(y1 + 1, min(h, y2))

        masked = image.copy()
        masked[mask <= 0] = 0
        crop = masked[y1:y2, x1:x2]
        side = max(crop.shape[0], crop.shape[1], 1)
        padded = np.zeros((side, side, 3), dtype=crop.dtype)
        padded[: crop.shape[0], : crop.shape[1]] = crop
        return self.clip_image_processor.preprocess(padded, return_tensors="pt")[
            "pixel_values"
        ][0]

    def _make_ref_focus_clip(self, image, mask):
        mask_bool = mask > 0
        if self.ref_focus_dilate > 0 and mask_bool.any():
            kernel_size = max(1, int(self.ref_focus_dilate))
            if kernel_size % 2 == 0:
                kernel_size += 1
            kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
            mask_bool = (
                cv2.dilate(mask_bool.astype(np.uint8), kernel, iterations=1) > 0
            )

        if self.ref_image_mode == "full_blur":
            focused = cv2.GaussianBlur(image, (31, 31), 0)
            focused[mask_bool] = image[mask_bool]
        else:
            bg = np.clip(self.ref_focus_background, 0.0, 1.0)
            focused = (image.astype(np.float32) * bg).astype(np.uint8)
            focused[mask_bool] = image[mask_bool]

        return self.clip_image_processor.preprocess(focused, return_tensors="pt")[
            "pixel_values"
        ][0]

    def _make_ref_image_clip(self, image, mask, bbox_xyxy):
        if self.ref_image_mode == "crop":
            return self._make_ref_crop_clip(image, mask, bbox_xyxy)
        return self._make_ref_focus_clip(image, mask)

    def __getitem__(self, idx):
        if self.deterministic:
            ep = self.episodes[idx % len(self.episodes)]
        else:
            ep = self.episodes[random.randint(0, len(self.episodes) - 1)]

        if "pairs" in ep:
            return self._getitem_counterfactual(ep)
        return self._getitem_single_pair(ep)

    def _getitem_counterfactual(self, ep):
        image_path = ep["image_path"]
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ori_size = image.shape[:2]

        conversations = []
        questions = []
        sampled_classes = []
        masks_np = []
        ref_masks_np = []
        ref_bboxes_rows = []
        ref_valid_rows = []
        mask_weights_rows = []
        ref_images_rows = []
        for pair in ep["pairs"]:
            miner_mask = self._read_mask(pair["miner_mask_path"], ori_size)
            helmet_mask = self._read_mask(pair["helmet_mask_path"], ori_size)
            round1_query = pair.get(
                "round1_query",
                "Segment the miner whose helmet should be segmented next.",
            )
            round2_query = pair.get(
                "round2_query",
                ep.get(
                    "same_round2_query",
                    "Based on the miner mask from the previous round, segment only the mining helmet worn by that miner.",
                ),
            )

            conversations.append(
                self._build_multiround_conversation(round1_query, round2_query)
            )
            questions.extend([round1_query, "[REF] " + round2_query])
            sampled_classes.extend(["coal_miner", "mining_helmet"])
            masks_np.extend([miner_mask, helmet_mask])
            ref_masks_np.extend([np.zeros_like(miner_mask), miner_mask])
            ref_bboxes_rows.extend(
                [
                    [0.0, 0.0, 0.0, 0.0],
                    self._norm_bbox(pair.get("miner_bbox_xyxy"), ori_size),
                ]
            )
            ref_valid_row = [0.0, 1.0]
            ref_valid_rows.extend(ref_valid_row)
            mask_weights_rows.extend(
                [
                    float(pair.get("pair_score", pair.get("miner_quality", 1.0))),
                    1.0,
                ]
            )
            ref_images_rows.extend(
                [
                    torch.zeros(3, 224, 224),
                    self._make_ref_image_clip(
                        image,
                        miner_mask,
                        pair.get("miner_bbox_xyxy"),
                    ),
                ]
            )

        masks = torch.from_numpy(np.stack(masks_np, axis=0))
        labels = torch.ones(masks.shape[1], masks.shape[2]) * self.ignore_label
        ref_masks = torch.from_numpy(np.stack(ref_masks_np, axis=0))
        ref_bboxes = torch.tensor(ref_bboxes_rows, dtype=torch.float32)
        ref_valids = torch.tensor(ref_valid_rows, dtype=torch.float32)
        mask_weights = torch.tensor(mask_weights_rows, dtype=torch.float32)
        ref_images_clip = torch.stack(ref_images_rows, dim=0)

        image_clip = self.clip_image_processor.preprocess(image, return_tensors="pt")[
            "pixel_values"
        ][0]
        image_sam = self.transform.apply_image(image)
        resize = image_sam.shape[:2]
        image_sam = self.preprocess(
            torch.from_numpy(image_sam).permute(2, 0, 1).contiguous()
        )

        return (
            image_path,
            image_sam,
            image_clip,
            conversations,
            masks,
            labels,
            resize,
            questions,
            sampled_classes,
            ref_masks,
            ref_bboxes,
            ref_valids,
            mask_weights,
            ref_images_clip,
        )

    def _getitem_single_pair(self, ep):
        image_path = ep["image_path"]
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ori_size = image.shape[:2]

        round1, round2 = ep["rounds"]
        round1_mask = self._read_mask(round1["target_mask"], ori_size)
        round2_mask = self._read_mask(round2["target_mask"], ori_size)

        masks = torch.from_numpy(np.stack([round1_mask, round2_mask], axis=0))
        labels = torch.ones(masks.shape[1], masks.shape[2]) * self.ignore_label

        ref_masks = torch.from_numpy(np.stack([np.zeros_like(round1_mask), round1_mask], axis=0))
        ref_bboxes = torch.tensor(
            [
                [0.0, 0.0, 0.0, 0.0],
                self._norm_bbox(round1.get("target_bbox_xyxy"), ori_size),
            ],
            dtype=torch.float32,
        )
        ref_valids = torch.tensor([0.0, 1.0], dtype=torch.float32)
        mask_weights = torch.tensor(
            [
                float(round1.get("loss_weight", ep.get("pair_score", 1.0))),
                float(round2.get("loss_weight", 1.0)),
            ],
            dtype=torch.float32,
        )

        ref_images_clip = torch.stack(
            [
                torch.zeros(3, 224, 224),
                self._make_ref_image_clip(
                    image,
                    round1_mask,
                    round1.get("target_bbox_xyxy"),
                ),
            ],
            dim=0,
        )

        conversations = [
            self._build_multiround_conversation(round1["query"], round2["query"])
        ]
        questions = [round1["query"], "[REF] " + round2["query"]]
        sampled_classes = [round1.get("target_category", "coal_miner"), round2.get("target_category", "mining_helmet")]

        image_clip = self.clip_image_processor.preprocess(image, return_tensors="pt")[
            "pixel_values"
        ][0]
        image_sam = self.transform.apply_image(image)
        resize = image_sam.shape[:2]
        image_sam = self.preprocess(
            torch.from_numpy(image_sam).permute(2, 0, 1).contiguous()
        )

        return (
            image_path,
            image_sam,
            image_clip,
            conversations,
            masks,
            labels,
            resize,
            questions,
            sampled_classes,
            ref_masks,
            ref_bboxes,
            ref_valids,
            mask_weights,
            ref_images_clip,
        )
