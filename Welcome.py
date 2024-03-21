import streamlit as st
from viz import plot_moa_tsne, display_cluster_info


st.set_page_config(page_title='Vector Rx', page_icon=':pill:', layout='wide')

st.subheader(':pill: Welcome to Vector Rx - A Search Engine for FDA Approved Drugs')
col1, _ = st.columns(2)
with col1:
    k = st.slider('For the visualization of how data is grouped into clusters, please select a k value', 1, 24, 24)

df = plot_moa_tsne(k)

# Add a dropdown menu to select what type of information to display
st.subheader('Expand to see the cluster details!')
display_cluster_info(df)
