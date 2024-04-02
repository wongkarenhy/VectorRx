import pickle
from enum import Enum
from itertools import chain, count
from tempfile import NamedTemporaryFile

import faiss
import numpy as np
import pandas as pd
import streamlit as st
from openai import OpenAI

from connections import get_contents


class SearchMode(Enum):
    MOA = 'moa'
    AR = 'adverse_reactions'


class DrugsSearchEngine:
    def __init__(self):
        self.client = self.__setup_openai_client()
        self.ar_faiss_index, self.moa_faiss_index = self.__get_indices()
        self.__df, self.__metadata = self.__get_drug_data()
        self.moa_mapping, self.ar_mapping = self.__get_mappings()

        self.__validate_data_integrity()

    @property
    def df(self) -> pd.DataFrame:
        return self.__df

    @property
    def metadata(self) -> dict:
        return self.__metadata

    def __repr__(self):
        return (f"{self.__class__.__name__}("
                f"Number of Drugs={len(self.df)}, "
                f"Data Last Accessed={self.metadata['last_updated']}, "
                f"Embedding Model={self.metadata['embedding_model']}) ")

    def __validate_data_integrity(self) -> None:
        assert len(self.metadata['moa_summaries']) == len(self.df), 'Length of moa_summaries and df should match'
        assert len(self.metadata['ar_summaries']) == len(self.df), 'Length of ar_summaries and df should match'
        assert len(list(chain(*self.metadata['moa_summaries']))) == self.moa_faiss_index.ntotal, \
            'Length of flattened moa_summaries and its faiss_index should match'
        assert len(list(chain(*self.metadata['ar_summaries']))) == self.ar_faiss_index.ntotal, \
            'Length of flattened ar_summaries and its faiss_index should match'

    @staticmethod
    def __setup_openai_client() -> OpenAI:
        return OpenAI(api_key=st.secrets['OPENAI_API_KEY'])

    @st.cache_data
    def __get_indices(_self) -> tuple[faiss.IndexPQ, faiss.IndexFlatIP]:
        """Download the adverse reactions index from GCS to a temporary directory and load it into faiss."""
        indices = {}
        for i in ['AR_EMBEDDING_IDX', 'MOA_EMBEDDING_IDX']:
            bucket_name, path = st.secrets[i].replace('gs://', '').split('/', 1)
            f_obj = get_contents(bucket_name, path)
            with NamedTemporaryFile() as temp_idx:
                temp_idx.write(f_obj.read())
                indices[i] = faiss.read_index(temp_idx.name)
        return indices['AR_EMBEDDING_IDX'], indices['MOA_EMBEDDING_IDX']

    @st.cache_data
    def __get_drug_data(_self) -> tuple[pd.DataFrame, dict]:
        """Download the tsv and load into a dataframe"""
        bucket_name, path = st.secrets['DRUG_METADATA'].replace('gs://', '').split('/', 1)
        f_obj = get_contents(bucket_name, path)
        pkl = pickle.load(f_obj)
        return pkl['df'], pkl['metadata']

    def __get_mappings(self) -> tuple[dict[int, int], dict[int, int]]:
        """
        Each drug can have multiple active ingredients or components and each component has a mechanism of action. This
        mapping is used to map the flattened moa_summaries to the original index in the dataframe (per drug).
        """
        moa_counter = count()
        moa_mapping = {next(moa_counter): i for i, moa in enumerate(self.metadata['moa_summaries']) for _ in moa}
        ar_counter = count()
        ar_mapping = {next(ar_counter): i for i, ar in enumerate(self.metadata['ar_summaries']) for _ in ar}
        return moa_mapping, ar_mapping

    def get_original_index(self, mode, flattened_index: int) -> int:
        """Map the flattened index to the original index in the dataframe."""
        mapping = self.moa_mapping if mode == SearchMode.MOA else self.ar_mapping
        return mapping[flattened_index]

    @st.cache_data
    def get_embedding(_self, chunk) -> np.array:
        response = _self.client.embeddings.create(input=chunk, model=_self.metadata['embedding_model'])
        embedding = response.data[0].embedding
        return np.array([embedding]).astype(np.float32)

    def search_most_similar(self, mode: SearchMode, xq: np.array, k: int) -> pd.DataFrame:
        """Find the k-nearest most similar MOA or AR based on the query embedding."""
        index = self.moa_faiss_index if mode == SearchMode.MOA else self.ar_faiss_index
        d, i = index.search(xq, k)
        # Remove i == -1 and the corresponding cosine similarity scores
        removed_indices = np.where(i[0] == -1)
        i = np.delete(i[0], removed_indices)
        d = np.delete(d[0], removed_indices)

        transformed_i = [self.get_original_index(mode, x) for x in i]
        # Add cosine similarity as a column to drug_data
        filtered_df = self.df.iloc[transformed_i]
        filtered_df['cosine_similarity'] = d

        cols_for_dedupe = [c for c in filtered_df.columns if c != 'cosine_similarity']
        return filtered_df.drop_duplicates(subset=cols_for_dedupe, keep='first')

    def query(self, mode: SearchMode, query_string: str, threshold: float, in_keyword: str, ex_keyword: str) -> pd.DataFrame:
        if not query_string and not in_keyword:
            raise ValueError('Must provide either a query string or an include keyword.')

        if query_string:
            xq = self.get_embedding(query_string)  # convert string to vector
            faiss.normalize_L2(xq)  # for cosine similarity
            raw_df = self.search_most_similar(mode, xq, k=10000)
        else:  # keyword searches only
            raw_df = self.df
            raw_df[f'cosine_similarity'] = 1  # assumes full similarity for keyword-based searches

        filtered_df = raw_df[raw_df['cosine_similarity'] > threshold]

        # Filter by keyword
        if in_keyword:
            filtered_df = filtered_df[filtered_df[mode.value].str.contains(in_keyword, case=False)]

        if ex_keyword:
            filtered_df = filtered_df[~filtered_df[mode.value].str.contains(ex_keyword, case=False)]
        return filtered_df


_search_engine_instance = None


def get_search_engine_instance():
    global _search_engine_instance
    if _search_engine_instance is None:
        _search_engine_instance = DrugsSearchEngine()
    return _search_engine_instance
