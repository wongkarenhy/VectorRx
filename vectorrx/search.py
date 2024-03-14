"""Everything here is WIP/for demo purposes only!!!"""
import pickle
import os
import re
from itertools import chain, count
from tempfile import TemporaryDirectory

import faiss
import numpy as np
from openai import OpenAI
import pandas as pd
import requests
import streamlit as st
from st_files_connection import FilesConnection


MOA_EMBEDDING_IDX = 'gs://hx-karen/fda_drugs/moa_embeddings_norm_v2.index'
DRUG_METADATA = 'gs://hx-karen/fda_drugs/metadata_v2.pkl'


class DrugsSearchEngine:
    def __init__(self):
        self.__setup_openai_client()

        self.conn = st.connection('gcs', type=FilesConnection)

        self.moa_index = self.__get_moa_index()
        self.drug_df, self.embedding_model, self.moa_summaries = self.__get_drug_data()
        self.moa_mapping = self.__get_moa_mapping()

        assert len(self.moa_summaries) == len(self.drug_df), 'Length of moa_summaries and df should be the same'
        assert len(list(chain(*self.moa_summaries))) == self.moa_index.ntotal, \
            'Length of flattened moa_summaries and faiss_index should be the same'

    def __setup_openai_client(self):
        openai_api_key = st.secrets['openai_api_key']
        self.client = OpenAI(api_key=openai_api_key)

    def __get_moa_index(self):
        """Fetch the moa index from GCS"""
        with TemporaryDirectory('get_faiss_index-') as td:
            moa_index_file = self.conn.download(gs_url=MOA_EMBEDDING_IDX, outfname=os.path.join(td, 'moa-faiss-index'))
            return faiss.read_index(moa_index_file)

    def __get_drug_data(self):
        """Download the tsv and load into a dataframe"""
        with self.conn.open(gs_url=DRUG_METADATA, mode='rb') as f:
            pkl = pickle.loads(f)
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

    def search_most_similar_moa(self, query: np.array, k: int):
        d, i = self.moa_index.search(query, k)
        d = d[0]  # always sorted from highest to lowest cosine similarity
        i = i[0]  # this corresponds to the index of the flattened moa_summaries
        transformed_i = [self.moa_mapping[x] for x in i]  # transform the index to the original index in the dataframe
        # Add distance as a column to drug_data
        filtered_df = self.drug_df.iloc[transformed_i]
        filtered_df['moa_cosine_similarity'] = [round(num, 3) for num in d]

        cols_for_dedupe = [c for c in filtered_df.columns if c not in {'moa_cosine_similarity'}]
        filtered_df = filtered_df.drop_duplicates(subset=cols_for_dedupe, keep='first')
        return filtered_df

    def query(self, query_string, k, similarity_threshold, in_keyword, ex_keyword):
        msg = 'Mode should be either "moa" or "adverse_reactions". You can only query by one mode at a time.'

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

        return filtered_df


def __format_link(x):
    # Extract only digits from the string
    x = re.sub(r'\D', '', x)
    return f'<a href="https://www.accessdata.fda.gov/scripts/cder/daf/index.cfm?event=overview.process&ApplNo={x}">{x}</a>'


def __clean_up_column_names(column_names: list):
    return [c.replace('_', ' ').title() for c in column_names]


search_engine = DrugsSearchEngine()
df = search_engine.query('topoisomerase inhibitor', 100, 0.4, '', '')
st.dataframe(df)


# def get_cosine_similarity_histogram(df: pd.DataFrame):
#     xcol = 'MOA Cosine Similarity'
#     bucketed_df = vizualization_utils.histogram_from_df(df, xcol=xcol, num_buckets=50)
#     return PreactChart.new_histogram_from_df(bucketed_df, xcol, 'num').size(600, 200)
#
#
# def add_search_by_moa_and_adverse_reactions(page: PageBuilder):
#     page.markdown('This is a search engine for FDA approved drugs that allows you to search a drug by its mechanism of '
#                   'action and/or adverse reactions. There are two ways to perform a search:\n1) Enter a mechanism '
#                   'of action and/or adverse reactions as free text and set a similarity threshold. This method uses '
#                   'the underlying semantics of your query to find similar drugs. Higher similarity threshold will return '
#                   'fewer but more semantically similar drugs. You can also include or exclude keywords to narrow down '
#                   'the results.\n2) Use exact keyword matching (case insensitive).')
#
#     page.spacer()
#     with page.inset_container('moa_search_params'):
#         moa_query_string = page.textarea('moa_query', 'Mechanism of Action Query (Free text)', initial_value='topoisomerase inhibitor').text.strip()
#         with page.row_layout('moa_keyword_layout'):
#             moa_in = page.textarea('moa_keyword_to_include', 'MOA Keyword to Include (Optional)', initial_value='').text.strip()
#             page.spacer(5)
#             moa_ex = page.textarea('moa_keyword_to_exclude', 'MOA Keyword to Exclude (Optional)', initial_value='').text.strip()
#         moa_threshold = page.add(Slider(min=0, max=100, increment=5, initial_value=40, label='MOA Similarity Threshold', recompute_page_on_change=True), 'moa_similarity_threshold')
#     page.spacer()
#
#     with page.inset_container('ar_search_params'):
#         page.html('🚧 <code> This section is under maintenance. Free text search will be available soon. </code> 🚧')
#         # TODO (karen): Optimize adverse reactions free-text search. Let's disable this for now.
#         ar_query_string = ''
#         # ar_query_string = page.textarea('ar_query_string', 'Adverse Reactions Query', initial_value='').text.strip()
#         with page.row_layout('ar_keyword_layout'):
#             ar_in = page.textarea('ar_keyword_to_include', 'Adverse Reactions Keyword to Include (Optional)', initial_value='').text.strip()
#             page.spacer(5)
#             ar_ex = page.textarea('ar_keyword_to_exclude', 'Adverse Reactions Keyword to Exclude (Optional)', initial_value='').text.strip()
#         ar_threshold = Slider(min=0, max=100, increment=5, initial_value=0, label='Adverse Reactions Similarity Threshold', recompute_page_on_change=True)
#     page.spacer()
#
#     with page.inset_container('max_record_param'):
#         max_records = int(page.textarea('max_records', 'Max records', initial_value='100').text)
#
#     if page.button('search', 'Search').just_clicked:
#         assert moa_query_string or ar_query_string or moa_in or ar_in, 'Invalid search parameters'
#         fda = DrugsSearchEngine()
#
#         moa_results, ar_results = pd.DataFrame(), pd.DataFrame()
#         if moa_query_string or moa_in or moa_ex:
#             threshold = moa_threshold.selected_value / 100
#             moa_results = fda.query(MOA_QUERY_MODE, moa_query_string, max_records, threshold, moa_in, moa_ex)
#
#             if moa_results.empty:
#                 page.html(f'<code>No records found for this MOA: {moa_query_string}!</code>')
#                 return
#
#         if ar_query_string or ar_in or ar_ex:
#             threshold = ar_threshold.selected_value / 100
#             ar_results = fda.query(ADVERSE_REACTIONS_QUERY_MODE, ar_query_string, max_records, threshold, ar_in, ar_ex)
#
#             if ar_results.empty:
#                 page.html(f'<code>No records found for this adverse reaction: {ar_query_string}!</code>')
#                 return
#
#         # If both dataframes are not empty, find the overlapping records and return
#         if not moa_results.empty and not ar_results.empty:
#             common_indices = moa_results.index.intersection(ar_results.index)
#             if common_indices.empty:
#                 page.html(f'<code>No records found for {moa_query_string} AND {ar_query_string}!</code>')
#                 return
#
#             results = moa_results.loc[common_indices]
#             results['adverse_reactions_cosine_similarity'] = ar_results.loc[common_indices]['adverse_reactions_cosine_similarity']
#
#         elif not moa_results.empty:
#             results = moa_results
#         else:
#             results = ar_results
#
#         # Clean up dataframe
#         results['moa'] = [CollapsibleContainer.container_with([results['moa'].loc[i]], label=' '.join(fda.moa_summaries[i])) for i in results.index]
#         results['adverse_reactions'] = [CollapsibleContainer.container_with([results['adverse_reactions'].loc[i]], label=f"{results['adverse_reactions'].loc[i][:500]}...") for i in results.index]
#         results.columns = __clean_up_column_names(results.columns)
#         results = results.rename(columns={'Moa': 'Summarized Mechanism of Action (expand to see original text from FDA)',
#                                           'Adverse Reactions': 'Adverse Reactions (expand to see full text)',
#                                           'Moa Cosine Similarity': 'MOA Cosine Similarity'})
#         results['FDA Link'] = [__format_link(x) if not pd.isna(x) else None for x in results['Application Number']]
#         results = results.drop(columns=['Application Number'])
#         results = results[['FDA Link'] + [col for col in results.columns if col != 'FDA Link']]
#
#         page.add(get_cosine_similarity_histogram(results) if moa_query_string else None, 'similarity_histogram')
#         page.spacer()
#         page.add(HXDataFrame(results), 'results')
#
