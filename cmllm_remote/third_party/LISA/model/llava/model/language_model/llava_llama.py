#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import (AutoConfig, AutoModelForCausalLM, LlamaConfig,
                          LlamaForCausalLM, LlamaModel)
from transformers.modeling_outputs import CausalLMOutputWithPast

from ..llava_arch import LlavaMetaForCausalLM, LlavaMetaModel
from utils.utils import IMAGE_TOKEN_INDEX


class LlavaConfig(LlamaConfig):
    model_type = "llava"


class LlavaLlamaModel(LlavaMetaModel, LlamaModel):
    config_class = LlavaConfig

    def __init__(self, config: LlamaConfig):
        super(LlavaLlamaModel, self).__init__(config)


class LlavaLlamaForCausalLM(LlamaForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)

        self.model = LlavaLlamaModel(config)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def _build_expanded_token_mask(self, raw_input_ids, token_idx, target_len):
        vision_tower = self.get_vision_tower()
        image_token_len = getattr(vision_tower, "num_patches", 256)
        masks = []
        for row in raw_input_ids:
            pieces = []
            for token in row:
                if int(token.item()) == IMAGE_TOKEN_INDEX:
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

    def _replace_ref_input_embeddings(
        self,
        raw_input_ids,
        inputs_embeds,
        ref_input_embeddings=None,
        ref_token_idx=None,
        ref_injection_mode="replace",
        ref_input_scale=1.0,
    ):
        if ref_input_embeddings is None or ref_token_idx is None or ref_token_idx < 0:
            return inputs_embeds
        ref_token_mask = self._build_expanded_token_mask(
            raw_input_ids,
            ref_token_idx,
            inputs_embeds.shape[1],
        )
        if not ref_token_mask.any():
            return inputs_embeds

        ref_input_embeddings = ref_input_embeddings.to(
            device=inputs_embeds.device,
            dtype=inputs_embeds.dtype,
        )
        inputs_embeds = inputs_embeds.clone()
        for batch_idx in range(inputs_embeds.shape[0]):
            positions = torch.where(ref_token_mask[batch_idx])[0]
            if positions.numel() == 0:
                continue
            row_refs = ref_input_embeddings[batch_idx]
            for ref_idx, pos in enumerate(positions):
                if ref_idx >= row_refs.shape[0]:
                    break
                if ref_injection_mode == "replace":
                    inputs_embeds[batch_idx, pos] = row_refs[ref_idx]
                elif ref_injection_mode == "add":
                    inputs_embeds[batch_idx, pos] = (
                        inputs_embeds[batch_idx, pos] + row_refs[ref_idx]
                    )
                elif ref_injection_mode == "gated_add":
                    inputs_embeds[batch_idx, pos] = (
                        inputs_embeds[batch_idx, pos]
                        + float(ref_input_scale) * row_refs[ref_idx]
                    )
                else:
                    raise ValueError(
                        "Unsupported ref_injection_mode: {}".format(
                            ref_injection_mode
                        )
                    )
        return inputs_embeds

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        ref_input_embeddings: Optional[torch.FloatTensor] = None,
        ref_token_idx: Optional[int] = None,
        ref_injection_mode: Optional[str] = "replace",
        ref_input_scale: Optional[float] = 1.0,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        raw_input_ids = input_ids
        (
            input_ids,
            attention_mask,
            past_key_values,
            inputs_embeds,
            labels,
        ) = self.prepare_inputs_labels_for_multimodal(
            input_ids, attention_mask, past_key_values, labels, images
        )
        inputs_embeds = self._replace_ref_input_embeddings(
            raw_input_ids,
            inputs_embeds,
            ref_input_embeddings=ref_input_embeddings,
            ref_token_idx=ref_token_idx,
            ref_injection_mode=ref_injection_mode,
            ref_input_scale=ref_input_scale,
        )
        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model/pipeline parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        if self.training:
            output_hidden_states = outputs.hidden_states
        else:
            output_hidden_states = hidden_states

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=output_hidden_states,  # outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        images=None,
        **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "images": images,
            }
        )
        return model_inputs


AutoConfig.register("llava", LlavaConfig, exist_ok=True)
AutoModelForCausalLM.register(LlavaConfig, LlavaLlamaForCausalLM)
