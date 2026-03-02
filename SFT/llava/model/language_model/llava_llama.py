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
from torch.nn import CrossEntropyLoss, MSELoss

from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast

from ..llava_arch import LlavaMetaModel, LlavaMetaForCausalLM
import time
# from llava.train.train import DEFAULT_BOX_TOKEN_id


class LlavaConfig(LlamaConfig):
    model_type = "llava_custom"


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
        self.lm_head2 = nn.Linear(config.hidden_size, 4, bias=False)
        # self.lm_head2 = nn.Sequential(
        #     nn.Linear(config.hidden_size, 4, bias=False),
        #     nn.Sigmoid()
        # )

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    # def forward(
    #     self,
    #     input_ids: torch.LongTensor = None,
    #     attention_mask: Optional[torch.Tensor] = None,
    #     past_key_values: Optional[List[torch.FloatTensor]] = None,
    #     inputs_embeds: Optional[torch.FloatTensor] = None,
    #     labels: Optional[torch.LongTensor] = None,
    #     use_cache: Optional[bool] = None,
    #     output_attentions: Optional[bool] = None,
    #     output_hidden_states: Optional[bool] = None,
    #     images: Optional[torch.FloatTensor] = None,
    #     return_dict: Optional[bool] = None,
    #     numerical_values: Optional[torch.FloatTensor] = None,
    #     default_box_token_id: Optional[int] = None,
    # ) -> Union[Tuple, CausalLMOutputWithPast]:
    #     output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    #     output_hidden_states = (
    #         output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    #     )
    #     return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    #     input_ids, attention_mask, past_key_values, inputs_embeds, labels = self.prepare_inputs_labels_for_multimodal(input_ids, attention_mask, past_key_values, labels, images)
    #     box_index = torch.tensor([])
    #     if default_box_token_id is not None:
    #         box_positions = (labels == default_box_token_id).nonzero(as_tuple=False)
    #         if box_positions.numel() > 0:
    #             box_index = box_positions[:, 1] + 1

    #     # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
    #     outputs = self.model(
    #         input_ids=input_ids,
    #         attention_mask=attention_mask,
    #         past_key_values=past_key_values,
    #         inputs_embeds=inputs_embeds,
    #         use_cache=use_cache,
    #         output_attentions=output_attentions,
    #         output_hidden_states=output_hidden_states,
    #         return_dict=return_dict
    #     )

    #     hidden_states = outputs[0]
    #     # print(hidden_states)
    #     # with open("./log4.txt", "w") as file:
    #     #         file.write(str(hidden_states))
    #     logits = self.lm_head(hidden_states)

    #     # numeric_logits = None
    #     # if box_index.numel() > 0:
    #     #     # numeric_hidden = hidden_states[torch.arange(hidden_states.size(0)).unsqueeze(1), box_index.unsqueeze(0)]
    #     #     batch_idx = box_positions[:, 0]   # (N,)
    #     #     seq_idx = box_positions[:, 1]
    #     #     numeric_hidden = hidden_states[batch_idx, seq_idx]
    #     #     numeric_logits = self.lm_head2(numeric_hidden)  # (N, 1) 或 (N, output_dim)

    #     loss = None
    #     if labels is not None:
    #         # Shift so that tokens < n predict n
    #         shift_logits = logits[..., :-1, :].contiguous()
    #         shift_labels = labels[..., 1:].contiguous()
    #         # Flatten the tokens
    #         loss_fct = CrossEntropyLoss()
    #         shift_logits = shift_logits.view(-1, self.config.vocab_size)
    #         shift_labels = shift_labels.view(-1)
    #         # Enable model/pipeline parallelism
    #         shift_labels = shift_labels.to(shift_logits.device)
    #         loss = loss_fct(shift_logits, shift_labels)
        


    #     if not return_dict:
    #         output = (logits,) + outputs[1:]
    #         return (loss,) + output if loss is not None else output

    #     return CausalLMOutputWithPast(
    #         loss=loss,
    #         logits=logits,
    #         past_key_values=outputs.past_key_values,
    #         hidden_states=outputs.hidden_states,
    #         attentions=outputs.attentions,
    #     )
    
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
        return_dict: Optional[bool] = None,
        numerical_values: Optional[torch.FloatTensor] = None,
        default_box_token_id: Optional[int] = 32000,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # print(input_ids)
        # print("forward...", time.time())
        input_ids, attention_mask, past_key_values, inputs_embeds, labels = self.prepare_inputs_labels_for_multimodal(input_ids, attention_mask, past_key_values, labels, images)
        # print("after prepare:", input_ids)
        # box_index = (input_ids== default_box_token_id).nonzero(as_tuple=True)[0]
        # box_index = torch.tensor([], dtype=torch.long, device=input_ids.device)
        box_index = torch.tensor([])
        # if default_box_token_id is not None:
        box_positions = (labels == default_box_token_id).nonzero(as_tuple=False)
        # print("default_box_token_id:", default_box_token_id)
        # print("box_positions:", box_positions)
        # print("labels:", labels)
        # if box_positions.numel() > 0:
        box_index = box_positions[:, 1]

        # print("box_index:", box_index)
        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

        hidden_states = outputs[0]
        # print(hidden_states)
        # with open("./log4.txt", "w") as file:
        #         file.write(str(hidden_states))
        logits = self.lm_head(hidden_states)

        numeric_logits = None
        # print("box_index:", box_index)
        # if box_index.numel() > 0:
            # hidden_states[batch_idx, seq_idx, :] → (N, D)
        # print("index:", box_index)
        if box_index.numel() > 0:
            numeric_hidden = hidden_states[torch.arange(hidden_states.size(0)).unsqueeze(1), box_index.unsqueeze(0)]    
            numeric_logits = self.lm_head2(numeric_hidden)  # (N, 1) 或 (N, output_dim)

        loss = None
        lm_loss = None
        numeric_loss = None
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
            lm_loss = loss_fct(shift_logits, shift_labels)
            # print("numerical_values:", numerical_values)
            if numeric_logits is not None and numerical_values is not None:
                # if numerical_values.dim() == 1:
                #     numerical_values = numerical_values.unsqueeze(1)
                numerical_values = torch.tensor(numerical_values, dtype=shift_logits.dtype).to(shift_logits.device)
                numeric_loss_fct = MSELoss()
                # print("numeric_logits:", numeric_logits.shape)
                # print("numerical_values:", numerical_values)
                numeric_loss = numeric_loss_fct(numeric_logits, numerical_values)
            loss = lm_loss
            if numeric_loss is not None:
                # print("numeric_loss:", numeric_loss)
                # print("lm_loss:", lm_loss)
                numeric_loss = numeric_loss.to(lm_loss.device)
                loss = loss + 5*numeric_loss
        # print("loss:", loss)
        # print("loss_time...", time.time())

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
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
                "images": kwargs.get("images", None),
            }
        )
        return model_inputs

AutoConfig.register("llava_custom", LlavaConfig)
AutoModelForCausalLM.register(LlavaConfig, LlavaLlamaForCausalLM)
