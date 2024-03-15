"""Everything here is WIP/for demo purposes only!!!"""
import pickle
import re
from itertools import chain, count

import faiss
import numpy as np
from openai import OpenAI
import pandas as pd
import streamlit as st
from st_files_connection import FilesConnection

from vectorrx.utils import download_file_from_gcs


MOA_EMBEDDING_IDX = 'gs://hx-karen/fda_drugs/moa_embeddings_norm_v2.index'
DRUG_METADATA = 'gs://hx-karen/fda_drugs/metadata_v2.pkl'

FDA_DRUG_PREFIX = 'https://www.accessdata.fda.gov/scripts/cder/daf/index.cfm?event=overview.process&ApplNo='


class DrugsSearchEngine:
    def __init__(self):
        self.__setup_openai_client()

        self.moa_index = self.__get_moa_index()
        self.drug_df, self.embedding_model, self.moa_summaries = self.__get_drug_data()
        self.moa_mapping = self.__get_moa_mapping()

        assert len(self.moa_summaries) == len(self.drug_df), 'Length of moa_summaries and df should be the same'
        assert len(list(chain(*self.moa_summaries))) == self.moa_index.ntotal, \
            'Length of flattened moa_summaries and faiss_index should be the same'

    def __setup_openai_client(self):
        openai_api_key = st.secrets['OPENAI_API_KEY']
        self.client = OpenAI(api_key=openai_api_key)

    @staticmethod
    @st.cache_data
    def __get_moa_index():
        """Download the moa index from GCS to a temporary directory and load it into faiss."""
        temp_idx = download_file_from_gcs('hx-karen', 'fda_drugs/moa_embeddings_norm_v2.index')
        return faiss.read_index(temp_idx)

    @staticmethod
    @st.cache_data
    def __get_drug_data():
        """Download the tsv and load into a dataframe"""
        conn = st.connection('gcs', type=FilesConnection)
        with conn.open(path=DRUG_METADATA, mode='rb') as f:
            pkl = pickle.load(f)
        df = pkl['df']
        model = pkl['metadata']['embedding_model']
        moa_summaries = pkl['metadata']['moa_summaries']
        return df, model, moa_summaries  # moa_summaries are what get used for the vector embeddings

    def __get_moa_mapping(self):
        """
        Each drug can have multiple active ingredients or components and each component has a mechanism of action. This
        mapping is used to map the flattened moa_summaries to the original index in the dataframe (per drug).
        """
        counter = count()
        return {next(counter): i for i, moa in enumerate(self.moa_summaries) for _ in moa}

    @staticmethod
    def keyword_match(content, keyword):
        return keyword.lower() in content.lower()

    def get_embedding(self, chunk):
        response = self.client.embeddings.create(
            input=chunk,
            model=self.embedding_model,
        )
        embedding = response.data[0].embedding
        return np.array([embedding]).astype(np.float32)

    @staticmethod
    def __format_link(x):
        # Extract only digits from the string
        x = re.sub(r'\D', '', x)
        return f'{FDA_DRUG_PREFIX}{x}'

    @staticmethod
    def __clean_up_column_names(column_names: list):
        return [c.replace('_', ' ').title() for c in column_names]

    def search_most_similar_moa(self, query: np.array, k: int):
        d, i = self.moa_index.search(query, k)
        d = d[0]  # always sorted from highest to lowest cosine similarity
        i = i[0]  # this corresponds to the index of the flattened moa_summaries
        transformed_i = [self.moa_mapping[x] for x in i]  # transform the index to the original index in the dataframe
        # Add distance as a column to drug_data
        filtered_df = self.drug_df.iloc[transformed_i]
        # filtered_df['moa_cosine_similarity'] = [round(num, 3) for num in d]
        filtered_df['moa_cosine_similarity'] = d

        cols_for_dedupe = [c for c in filtered_df.columns if c not in {'moa_cosine_similarity'}]
        filtered_df = filtered_df.drop_duplicates(subset=cols_for_dedupe, keep='first')
        return filtered_df

    def query(self, query_string, k, similarity_threshold, in_keyword, ex_keyword):
        # Convert query_string into an embedding
        if query_string:
            q_embedding = self.get_embedding(query_string)
            faiss.normalize_L2(q_embedding)  # for cosine similarity
            raw_df = self.search_most_similar_moa(q_embedding, k)
        else:  # keyword searches only
            raw_df = self.drug_df
            raw_df[f'moa_cosine_similarity'] = 1

        # Filter by similarity
        similarity_cols = [c for c in raw_df.columns if c.endswith('cosine_similarity')]
        assert len(similarity_cols) == 1, 'There should only be one similarity column'

        similarity_col = similarity_cols[0]
        filtered_df = raw_df[raw_df[similarity_col] > similarity_threshold]

        # Filter by keyword
        if in_keyword:
            filtered_df = filtered_df[filtered_df.apply(lambda x: self.keyword_match(x['moa'], in_keyword), axis=1)]

        if ex_keyword:
            filtered_df = filtered_df[~filtered_df.apply(lambda x: self.keyword_match(x['moa'], ex_keyword), axis=1)]

        # Clean up column names
        filtered_df.columns = self.__clean_up_column_names(filtered_df.columns)

        # Add FDA link
        filtered_df['FDA Link'] = [self.__format_link(x) if not pd.isna(x) else None for x in filtered_df['Application Number']]

        # Drop the Application Number column and reset index
        filtered_df = filtered_df.drop(columns=['Application Number']).reset_index(drop=True)

        st.data_editor(
            filtered_df,
            column_config={
                "FDA Link": st.column_config.LinkColumn(
                    label="FDA Link",
                    display_text=f"{FDA_DRUG_PREFIX}(.*?)",
                    help="Click to open the FDA page for this drug",
                )
            }
        )


# search_engine = DrugsSearchEngine()
# search_engine.query('topoisomerase inhibitor', 100, 0.4, '', '')
