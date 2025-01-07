from src.core.interfaces import Postprocessor
from src.constants import keywords_recheck_pr, threshold_private

import pandas as pd
import torch
from typing import Dict, List, Tuple
from sentence_transformers import SentenceTransformer, util


class PrivatePostprocessor(Postprocessor):
    """Postprocessor for Private flow."""

    def postprocess(self, data, config) -> pd.DataFrame:
        """
        Postprocesses the data after classification by applying subtype and keyword processing.
        """
        # Prepare batches
        process_by_subtype_batch, process_by_keywords_batch = self._prepare_batches(data)

        # Postprocess by subtype
        if not process_by_subtype_batch.empty:
            subtype_updates = self._process_by_subtype(
                process_by_subtype_batch,
                config.private_subtypes_dict,
                config.sbert_model,
                config.device
            )
            data = self._update_data_with_results(data, subtype_updates)

        # Postprocess by keywords
        if not process_by_keywords_batch.empty:
            keywords_updates = self._process_by_keywords(
                config.keywords_recheck_pr,
                process_by_keywords_batch
            )
            data = self._update_data_with_results(data, keywords_updates)

        # Update force_pushed rows

        data.loc[(data['type_id'] == 5) & (data['rooms'].astype(int) > 1), 'type_id'] = 18
        # Ensure consistent format
        data['type_id'] = data['type_id'].apply(lambda x: x[0] if isinstance(x, list) else x)
        data['type_id'] = data['type_id'].astype(int)
        data['max_probability'] = data['max_probability'].apply(lambda x: x[0] if isinstance(x, list) else x)
        data['group_probability'] = 0
        data['group_probability'] = data['group_probability'].astype(float)
        return data


    @staticmethod
    def _process_by_subtype(batch_df: pd.DataFrame, encoded_dict: Dict[str, List[torch.Tensor]],
                            sbert_model: SentenceTransformer, device: torch.device) -> pd.DataFrame:
        """
        Postprocesses by checking subtype and finding nearest type_id based on similarity with encoded_dict embeddings.
        """
        # Convert and clean subtypes
        subtypes_cleaned = batch_df['subtype'].apply(lambda x: x.lower())
        embeddings = sbert_model.encode(subtypes_cleaned.to_list(), convert_to_tensor=True,
                                        show_progress_bar=False).to(device)

        keys, all_embeddings = [], []
        for key, emb_list in encoded_dict.items():
            for emb in emb_list:
                keys.append(key)
                if not isinstance(emb, torch.Tensor):
                    emb = torch.tensor(emb, device=device)
                all_embeddings.append(emb)

        all_embeddings_tensor = torch.stack(all_embeddings).to(device)
        similarities = util.pytorch_cos_sim(embeddings, all_embeddings_tensor)
        best_matches = similarities.argmax(dim=1)

        batch_df['type_id'] = [int(keys[idx]) for idx in best_matches]
        batch_df['max_probability'] = 0.0
        batch_df['group'] = 'postprocess_subtype'

        return batch_df[['uid', 'type_id', 'group', 'max_probability']]

    @staticmethod
    def _process_by_keywords(keywords_dict: Dict[int, List[str]], batch_df: pd.DataFrame) -> pd.DataFrame:
        """
        Updates `type_id` for objects by counting keyword occurrences in text fields
        and selecting the most suitable `type_id` based on defined thresholds.
        """
        combined_texts = batch_df.apply(
            lambda row: (
                f"{row.get('title', '')} {row.get('description', '')} {row.get('subtype', '')} {row.get('features', '')}"
            ).lower(), axis=1
        )

        batch_df['selected_type_id'] = batch_df['type_id'].apply(lambda x: x[2])

        condition_1 = (batch_df['type_id'].apply(lambda x: x[0] == 2)) & (
                combined_texts.str.len() < threshold_private['min_len_for_others'])
        batch_df.loc[condition_1, 'selected_type_id'] = 2

        for type_id, keywords in keywords_dict.items():
            keyword_counts = combined_texts.apply(lambda text: sum(1 for keyword in keywords if keyword in text))
            condition_2 = (batch_df['type_id'].apply(lambda x: x[0] == type_id)) & (
                    keyword_counts >= threshold_private['min_keywords_for_type'])
            batch_df.loc[condition_2, 'selected_type_id'] = type_id

        batch_df['type_id'] = batch_df['selected_type_id']
        batch_df['max_probability'] = 0.0
        batch_df['group'] = 'postprocess_keywords'

        return batch_df[['uid', 'type_id', 'group', 'max_probability']]

    @staticmethod
    def _prepare_batches(data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Prepares data for subtype and keyword processing based on certain criteria.
        """
        batch_1_mask = (
                (data['subtype'].notna()) &
                (~data['subtype'].isin(['Unknown', 'Κτίριο', 'Αέρας', 'Air'])) &
                (data['subtype'].str.len() > 2) &
                (data['max_probability'].apply(lambda x: x[0] < threshold_private['min_probability_for_subtype']))
        )
        batch_1 = data[batch_1_mask].copy()
        remaining_data = data[~batch_1_mask]
        batch_2_mask = (
                remaining_data['type_id'].apply(lambda x: x[0] in keywords_recheck_pr.keys())
        )
        batch_2 = remaining_data[batch_2_mask].copy()
        return batch_1, batch_2

    @staticmethod
    def _update_data_with_results(data: pd.DataFrame, updates: pd.DataFrame) -> pd.DataFrame:
        """
        Merges update data with main data, replacing specific columns where values are available.
        """
        data = data.merge(updates[['uid', 'type_id', 'max_probability', 'group']], on='uid', how='left',
                          suffixes=('', '_new'))
        data['type_id'] = data['type_id_new'].combine_first(data['type_id'])
        data['max_probability'] = data['max_probability_new'].combine_first(data['max_probability'])
        data['group'] = data['group_new'].combine_first(data['group'])
        data.drop(columns=['group_new', 'max_probability_new', 'type_id_new'], inplace=True)
        return data