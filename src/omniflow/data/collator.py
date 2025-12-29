import collections
from dataclasses import dataclass
from typing import Dict, Sequence, Any

import numpy as np
import torch


@dataclass
class VisionCollator:
    processor: Any 

    def pad_sequence(self, input_ids, batch_first, padding_value):
        if self.processor.tokenizer.padding_side == "left":
            input_ids = [torch.flip(_input_ids, [0]) for _input_ids in input_ids]
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=batch_first, padding_value=padding_value)
        if self.processor.tokenizer.padding_side == "left":
            input_ids = torch.flip(input_ids, [1])
        return input_ids

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        if isinstance(instances[0], list):
            instances = [inst for instance in instances for inst in instance]
        inputs = collections.defaultdict(list)
        for instance in instances:
            for key, values in instance.items():
                inputs[key].append(values)

        batched_inputs = {}
        if "input_ids" in inputs.keys():
            input_ids = inputs.pop("input_ids")
            input_ids = self.pad_sequence(
                input_ids,
                batch_first=True,
                padding_value=self.processor.tokenizer.pad_token_id,
            )
            batched_inputs["input_ids"] = input_ids
        if "labels" in inputs.keys():
            labels = inputs.pop("labels")
            labels = self.pad_sequence(
                labels,
                batch_first=True,
                padding_value=-100,
            )
            batched_inputs["labels"] = labels

        if "attention_mask" in inputs.keys():
            inputs.pop("attention_mask")

        attention_mask = input_ids.ne(self.processor.tokenizer.pad_token_id).long()
        batched_inputs["attention_mask"] = attention_mask

        # for the other keys
        for key, values in inputs.items():
            # Handle scalar/boolean values ( use_audio_in_video)
            if isinstance(values[0], bool) or (
                isinstance(values[0], (int, float)) and not isinstance(values[0], torch.Tensor)
            ):
                batched_inputs[key] = values[0]
            else:
                batched_inputs[key] = torch.stack(values, dim=0)
                if key == "video" and batched_inputs[key].ndim == 5:
                    tensor = batched_inputs[key]
                    # Target: (B, C, T, H, W)
                    if tensor.shape[4] == 3: # (B, T, H, W, C)
                        #  print("DEBUG: Permuting Path A (Last Dim 3)")
                         tensor = tensor.permute(0, 4, 1, 2, 3).contiguous()
                    elif tensor.shape[2] == 3: # (B, T, C, H, W)
                        #  print("DEBUG: Permuting Path B (Dim 2 is 3)")
                         tensor = tensor.permute(0, 2, 1, 3, 4).contiguous()
                    batched_inputs[key] = tensor
        return batched_inputs

    @property
    def image_token_id(self):
        return self.processor.tokenizer.convert_tokens_to_ids(self.processor.image_token)
