import pandas as pd
import streamlit as st

from search import SearchMode, get_search_engine_instance
from utils import add_histograms, clean_up

search_engine = get_search_engine_instance()

st.set_page_config(page_title='Vector Rx', page_icon=':pill:', layout='wide')
st.subheader(':pill: Search FDA Approved Drugs by Mechanism of Action')
st.write(f'[Data Source: OpenFDA](https://open.fda.gov/data/downloads/) - '
         f'Last Accessed: {search_engine.metadata["last_updated"]}')

st.write("""
This is a search engine for FDA approved drugs that allows you to search a drug by its mechanism of action.\
There are two ways to perform a search:
1. Enter a mechanism of action as free text and set a similarity threshold. This method uses the underlying \
semantics of your query to find similar drugs. Higher similarity threshold will return fewer but more \
semantically similar drugs. You can also include or exclude keywords to narrow down the results.
2. Leave the mechanism of action field empty and use the include and exclude keywords to search for drugs.""")

st.success("""
**Why embeddings?** Embeddings are a way to represent text as vectors in a high-dimensional space. They provide a \
nuanced search capability by not strictly adhering to exact terms, enabling users to adjust the similarity threshold \
to explore drugs with related actions. However, it's worth noting that this could increase the chance of \
encountering false positives, as it broadens the search to include a wide array of potentially related things.
""", icon='📈')

# MOA search fields
col1, col2 = st.columns(2)
with col1:
    num_records = st.slider('Number of Records to Display', 1, 1000, 100)
with col2:
    collapse = st.checkbox('Collapse Drugs with the Same Generic Name', value=True)

moa_query = st.text_input('Mechanism of Action Query [free text]', 'something that blocks topoisomerase')
moa_similarity_threshold = st.slider('Mechanism of Action Similarity Threshold', 0.0, 1.0, 0.2)
col1, col2 = st.columns(2)
with col1:
    moa_in = st.text_input('Mechanism of Action Keyword to Include', '')
with col2:
    moa_ex = st.text_input('Mechanism of Action Keyword to Exclude', '')

# Adverse reaction search fields
with st.expander('Adverse Reaction Search'):
    ar_query = st.text_input('Adverse Reaction Query [free text]', '')
    ar_similarity_threshold = st.slider('Adverse Reaction Similarity Threshold', 0.0, 1.0, 0.2)
    col1, col2 = st.columns(2)
    with col1:
        ar_in = st.text_input('Adverse Reaction Keyword to Include', '')
    with col2:
        ar_ex = st.text_input('Adverse Reaction Keyword to Exclude', '')

search_button = st.button('Search')

# Search action
if search_button:
    moa_df, ar_df, res_df = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    if moa_query or moa_in:
        moa_df = search_engine.query(SearchMode.MOA, moa_query, moa_similarity_threshold, moa_in, moa_ex)
        moa_df = moa_df.rename(columns={'cosine_similarity': 'cosine_similarity_moa'})
    if ar_query or ar_in:
        ar_df = search_engine.query(SearchMode.AR, ar_query, ar_similarity_threshold, ar_in, ar_ex)
        ar_df = ar_df.rename(columns={'cosine_similarity': 'cosine_similarity_ar'})

    no_results_msg = 'No results found. Please try a different query.'

    if not moa_df.empty and not ar_df.empty:
        merge_cols = [c for c in moa_df.columns if not c.startswith('cosine_similarity')]
        res_df = pd.merge(moa_df, ar_df, how='inner', on=merge_cols)
    elif not moa_df.empty:
        res_df = moa_df
    elif not ar_df.empty:
        res_df = ar_df

    if res_df.empty:
        st.warning(no_results_msg, icon="⚠️")
        st.stop()

    # Add a histogram of the cosine similarity scores
    warning_msg = ('Please be aware that text embeddings are based on the underlying semantics of the text and '
                   'there are no hard boundaries for similarity scores. Please adjust the thresholds accordingly.')
    st.info(warning_msg, icon='⚠')

    df = clean_up(res_df, records=num_records, collapse_by_generic_name=collapse)

    add_histograms(df)

    help_mgs = "Click to open the FDA page for this drug"
    fda_link = st.column_config.LinkColumn(label='FDA Link', display_text='.*=(\\d+)', help=help_mgs)
    st.data_editor(df, column_config={'FDA Link': fda_link})
