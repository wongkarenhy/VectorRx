import streamlit as st
from viz import plot_moa_tsne


st.set_page_config(page_title='Vector Rx', page_icon=':pill:', layout='wide')

st.subheader(':pill: Welcome to Vector Rx - A Search Engine for FDA Approved Drugs')
col1, _ = st.columns(2)
with col1:
    k = st.slider('For the visualization of how data is grouped into clusters, please select a k value', 1, 24, 24)

tsne = plot_moa_tsne(k)
st.plotly_chart(tsne)
