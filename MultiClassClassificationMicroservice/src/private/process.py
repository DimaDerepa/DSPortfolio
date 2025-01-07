import pandas as pd

from src.constants import INFERENCE_BATCH
from src.core.interfaces import ProcessorCore

import torch
from typing import Tuple, List
from torch.cuda.amp import autocast
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer
)


class PrivateProcessorCore(ProcessorCore):
    """Processor class for Private."""
    def process(self, data, config) -> pd.DataFrame:
        # Additional column for group by batches
        data['batch'] = (data.index // INFERENCE_BATCH)

        type_id = []
        type_prob = []
        # Inference by batches
        for _, batch in data.groupby('batch'):
            texts = batch["text"].tolist()
            types_id, type_probs = self._inference(config.models["main"], config.tokenizers["main"], config.device, texts)
            # Save results
            type_id.extend(types_id)
            type_prob.extend(type_probs)

        #update data_to_process
        data = data.assign(type_id=type_id)
        data = data.assign(max_probability = type_prob)
        data = data.assign(group = 'model processed')

        return data

    @staticmethod
    def _inference(
        model: AutoModelForSequenceClassification,
        tokenizer: AutoTokenizer,
        device: torch.device,
        texts: List[str],
    ) -> Tuple[List[List[int]], List[List[float]]]:
        """
       Runs inference on batch of composed objects.
       Args:
           model: Pretrained model.
           texts: texts for classification.
       Returns:
           predicted_classes (List[int]): List of the top predicted class for each object.
           round_probs (List[float]): List of probabilities for the top predicted class for each object.
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

                output = model(**inputs)
                logits = output.logits
                probs = torch.softmax(logits, dim=-1)

                top_k = 3
                top_probs, top_indices = torch.topk(probs, k=top_k, dim=-1)

                predicted_classes = top_indices.tolist()
                sorted_probs = top_probs.tolist()
                round_probs = [[round(prob, 4) for prob in j] for j in sorted_probs]

        return predicted_classes, round_probs