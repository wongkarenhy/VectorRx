import streamlit as st
from vectorrx.search import DrugsSearchEngine


st.set_page_config(page_title='Vector Rx', page_icon=':pill:', layout='wide')

search_engine = DrugsSearchEngine()
moa_query = st.text_input('Mechanism of Action Query', 'Topoisomerase inhibitor')

col1, col2 = st.columns(2)
with col1:
    similarity_threshold = st.slider('Similarity Threshold', 0.0, 1.0, 0.4)
    moa_in = st.text_input('Keyword to Include', '')
with col2:
    num_records = st.slider('Number of Records', 1, 500, 100)
    moa_ex = st.text_input('Keyword to Exclude', '')

# Add a search button
search_button = st.button('Search')

# Check if the search button has been clicked
if search_button:
    search_engine.query(moa_query, num_records, similarity_threshold, moa_in, moa_ex)
