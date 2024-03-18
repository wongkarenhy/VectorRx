import streamlit as st
from vectorrx.viz import plot_moa_tsne


st.set_page_config(page_title='Vector Rx', page_icon=':pill:', layout='wide')

st.subheader(':pill: Welcome to Vector Rx - A Search Engine for FDA Approved Drugs')

tsne = plot_moa_tsne()
st.plotly_chart(tsne)
