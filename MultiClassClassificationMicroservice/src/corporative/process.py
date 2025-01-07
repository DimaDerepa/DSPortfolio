import pandas as pd
import torch
from typing import (
    List,
    Tuple
)

from sentence_transformers.models import tokenizer
from torch.cuda.amp import autocast
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer
)

from src.constants import INFERENCE_BATCH
from src.core.interfaces import ProcessorCore


class CorporativeProcessorCore(ProcessorCore):
    """Processor class for Corporative"""
    def process(self, data, config) -> pd.DataFrame:
        """
        Process data for Corporative by this flow
        encode -> predict -> injection -> predict
        """
        #encode -> predict -> injection -> predict
        # Encode and predict for group
        encodings = self._encode(config.tokenizers['group'], config.device, data["text"].tolist())
        group_types_id, group_type_probs = self._predict_batches(encodings, config.models["group"])
        data.loc[:, "group"] = [config.types_group_map_de[t] for t in group_types_id]
        data.loc[:, "group_probability"] = group_type_probs
        # Inject predictions and re-encode for subgroup prediction
        injected_inputs = self._inject(encodings, data["group"], config.encoded_groups, config.tokenizers['group'], config.device)
        sub_types_ids, sub_type_probs = self._predict_batches(injected_inputs, config.models["subgroup"])
        # Update DataFrame
        data.loc[:, "type_id"] = sub_types_ids
        data.loc[:, "max_probability"] = sub_type_probs
        return data

    @staticmethod
    def _encode(tokenizer: AutoTokenizer, device: torch.device, texts: List[str], padding: bool=True) -> Tuple[List[int], List[int]]:
        """
            Encodes texts into a batch of encoded sequences.
            Args:
                tokenizer: tokenizer for encoding
                device: device can be cuda or cpu. used for move inputs on cuda device
                texts: texts which will be encoded
            Returns:
                inputs: encoded text sequences
            """
        with torch.inference_mode():
            with autocast():
                inputs = tokenizer(
                    texts,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                    max_length=512
                ).to(device)

        return inputs

    @staticmethod
    def _inject(encoded_objs, predicts, groups_encodes, tokenizer, device: torch.device):
        """
        Function which updates encoded objects and adds the encoded sequence "###CATEGORY: <prediction>"
        at the required position.

        Args:
            encoded_objs: encoded objects by the group model
            predicts: predictions of the group model
            groups_encodes: dictionary of pre-encoded predictions of the group model

        Returns:
            Batch of encoded tensors with updated input_ids and attention_mask
        """
        # Search sequence which represents '###FEATURES'
        target_sequence = torch.tensor([6, 187284, 30018, 55126, 34465], device=device)
        target_len = target_sequence.size(0)

        # Max length for padding and truncation
        max_len = 512

        processed_input_ids = []
        processed_attention_masks = []

        # Iterate over objects and their respective predictions and masks
        for obj, pred, attention_mask in zip(encoded_objs['input_ids'], predicts, encoded_objs['attention_mask']):
            # Get encode 'category' for each prediction
            insert_sequence = torch.tensor(groups_encodes.get(pred, []), device=device)
            insert_len = insert_sequence.size(0)
            # Find indexes for input 'category sequence'
            try:
                match_index = (obj.unfold(0, target_len, 1) == target_sequence).all(1).nonzero(as_tuple=True)[0].item()
            except IndexError:
                raise ValueError("Target sequence not found in the input IDs.")

            # Insert sequence in the required place
            new_input_ids = torch.cat((obj[:match_index], insert_sequence, obj[match_index:]))
            # Adjust attention mask, preserving padding zeros
            new_attention_mask = torch.cat((
                attention_mask[:match_index],
                torch.ones(insert_len, dtype=torch.int64, device=device),
                attention_mask[match_index:]
            ))

            # Truncate or pad to max_len
            new_input_ids = new_input_ids[:max_len]
            new_attention_mask = new_attention_mask[:max_len]

            if new_input_ids.size(0) < max_len:
                pad_len = max_len - new_input_ids.size(0)
                new_input_ids = torch.cat((new_input_ids, torch.ones(pad_len, dtype=torch.int64, device=device)))
                new_attention_mask = torch.cat(
                    (new_attention_mask, torch.zeros(pad_len, dtype=torch.int64, device=device))
                )
            processed_input_ids.append(new_input_ids)
            processed_attention_masks.append(new_attention_mask)

        processed_input_ids = torch.stack(processed_input_ids)
        processed_attention_masks = torch.stack(processed_attention_masks)

        return {'input_ids': processed_input_ids, 'attention_mask': processed_attention_masks}

    @staticmethod
    def _predict(model: AutoModelForSequenceClassification, inputs) -> Tuple[List[int], List[float]]:
        """
        Runs inference on pre-encoded batch of data.
        Args:
            model: Pretrained model, for Corporative it could be a group or subgroup model.
            inputs: Pre-encoded sequences.
        Returns:
            predicted_classes (List[int]): List of the top predicted class for each object.
            round_probs (List[float]): List of probabilities for the top predicted class for each object.
        """
        with torch.inference_mode():
            with autocast():

                output = model(**inputs)
                logits = output.logits

                # Apply softmax to get probabilities
                probs = torch.softmax(logits, dim=-1)

                # Get the highest probability and its corresponding index (class)
                top_probs, top_indices = torch.max(probs, dim=-1)

                # Convert to lists for easy handling outside of torch tensors
                predicted_classes = top_indices.tolist()
                round_probs = [round(prob.item(), 4) for prob in top_probs]
        return predicted_classes, round_probs

    def _predict_batches(self, encodings, model):
        """Split inference into batches and call predict function."""
        input_id_batches = torch.split(encodings["input_ids"], INFERENCE_BATCH)
        attention_mask_batches = torch.split(encodings["attention_mask"], INFERENCE_BATCH)

        types_ids, type_probs = [], []

        for input_ids_batch, attention_mask_batch in zip(input_id_batches, attention_mask_batches):
            batch_inputs = {
                "input_ids": input_ids_batch,
                "attention_mask": attention_mask_batch
            }
            ids, probs = self._predict(model, batch_inputs)
            types_ids.extend(ids)
            type_probs.extend(probs)

        return types_ids, type_probs
