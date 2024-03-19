import pickle
import re
from itertools import chain, count

import faiss
import numpy as np
import pandas as pd
import streamlit as st
from openai import OpenAI
from st_files_connection import FilesConnection
from tempfile import NamedTemporaryFile


FDA_DRUG_PREFIX = 'https://www.accessdata.fda.gov/scripts/cder/daf/index.cfm?event=overview.process&ApplNo='


class DrugsSearchEngine:
    def __init__(self):
        self.client = self.__setup_openai_client()
        self.conn = st.connection('gcs', type=FilesConnection)
        self.moa_faiss_index = self.__get_moa_faiss_index()
        self.__df, self.__embedding_model, self.__moa_summaries, self.__last_updated = self.__get_drug_data()
        self.moa_mapping = self.__get_moa_mapping()

        self.__validate_data_integrity()

    @property
    def df(self) -> pd.DataFrame:
        return self.__df

    @property
    def embedding_model(self) -> str:
        return self.__embedding_model

    @property
    def moa_summaries(self) -> list[list[str]]:
        return self.__moa_summaries

    @property
    def last_updated(self) -> str:
        return self.__last_updated

    def __repr__(self):
        return (f"{self.__class__.__name__}("
                f"Number of Drugs={len(self.df)}, "
                f"Data Last Accessed='{self.last_updated}', "
                f"Embedding Model='{self.embedding_model}')")

    def __validate_data_integrity(self) -> None:
        assert len(self.moa_summaries) == len(self.df), 'Length of moa_summaries and df should match'
        assert len(list(chain(*self.moa_summaries))) == self.moa_faiss_index.ntotal, \
            'Length of flattened moa_summaries and faiss_index should match'

    @staticmethod
    def __setup_openai_client() -> OpenAI:
        return OpenAI(api_key=st.secrets['OPENAI_API_KEY'])

    @st.cache_data
    def __get_moa_faiss_index(_self) -> faiss.IndexFlatIP:
        """Download the moa index from GCS to a temporary directory and load it into faiss."""
        with NamedTemporaryFile() as temp_idx, _self.conn.open(path=st.secrets['MOA_EMBEDDING_IDX'], mode='rb') as f:
            temp_idx.write(f.read())
            return faiss.read_index(temp_idx.name)

    @st.cache_data
    def __get_drug_data(_self) -> tuple[pd.DataFrame, str, list[list[str]], str]:
        """Download the tsv and load into a dataframe"""
        with _self.conn.open(path=st.secrets['DRUG_METADATA'], mode='rb') as f:
            pkl = pickle.load(f)

        df = pkl['df']
        model = pkl['metadata']['embedding_model']
        moa_summaries = pkl['metadata']['moa_summaries']
        last_updated = pkl['metadata']['last_updated']
        return df, model, moa_summaries, last_updated  # moa_summaries are what get used for the vector embeddings

    def __get_moa_mapping(self) -> dict[int, int]:
        """
        Each drug can have multiple active ingredients or components and each component has a mechanism of action. This
        mapping is used to map the flattened moa_summaries to the original index in the dataframe (per drug).
        """
        counter = count()
        return {next(counter): i for i, moa in enumerate(self.__moa_summaries) for _ in moa}

    def get_original_index(self, flattened_index: int) -> int:
        """Map the flattened index to the original index in the dataframe."""
        return self.moa_mapping[flattened_index]

    @st.cache_data
    def get_embedding(_self, chunk) -> np.array:
        response = _self.client.embeddings.create(input=chunk, model=_self.embedding_model)
        embedding = response.data[0].embedding
        return np.array([embedding]).astype(np.float32)

    @staticmethod
    def __format_link(x) -> str:
        """Extract only digits from the string."""
        x = re.sub(r'\D', '', x)
        return f'{FDA_DRUG_PREFIX}{x}'

    @staticmethod
    def __clean_up_column_names(column_names: list) -> list[str]:
        return [c.replace('_', ' ').title() for c in column_names]

    def search_most_similar_moa(self, xq: np.array, k: int) -> pd.DataFrame:
        """Find the most similar MOA based on the query embedding."""
        d, i = self.moa_faiss_index.search(xq, k)
        transformed_i = [self.get_original_index(x) for x in i[0]]
        # Add cosine similarity as a column to drug_data
        filtered_df = self.df.iloc[transformed_i]
        filtered_df['moa_cosine_similarity'] = d[0]

        cols_for_dedupe = [c for c in filtered_df.columns if c not in {'moa_cosine_similarity'}]
        return filtered_df.drop_duplicates(subset=cols_for_dedupe, keep='first')

    def query(self, query_string: str, k: int, similarity_threshold: float, in_keyword: str, ex_keyword: str) \
            -> pd.DataFrame:
        if not query_string and not in_keyword:
            raise ValueError('Query string or in_keyword must be provided')

        if query_string:
            xq = self.get_embedding(query_string)  # convert string to vector
            faiss.normalize_L2(xq)  # for cosine similarity
            raw_df = self.search_most_similar_moa(xq, k)
        else:  # keyword searches only
            raw_df = self.df
            raw_df[f'moa_cosine_similarity'] = 1  # assumes full similarity for keyword-based searches

        filtered_df = raw_df[raw_df['moa_cosine_similarity'] > similarity_threshold]

        # Filter by keyword
        if in_keyword:
            filtered_df = filtered_df[filtered_df['moa'].str.contains(in_keyword, case=False)]

        if ex_keyword:
            filtered_df = filtered_df[~filtered_df['moa'].str.contains(ex_keyword, case=False)]

        # Clean up column names
        filtered_df.columns = self.__clean_up_column_names(filtered_df.columns)

        # Add FDA link
        filtered_df['FDA Link'] = [self.__format_link(x) if not pd.isna(x)
                                   else None for x in filtered_df['Application Number']]

        # Drop the Application Number column and reset index
        return filtered_df.drop(columns=['Application Number']).reset_index(drop=True)
