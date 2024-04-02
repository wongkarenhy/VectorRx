import faiss
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
from sklearn.manifold import TSNE
import streamlit as st

from search import get_search_engine_instance


def match_structure(original_list, list_of_lists):
    """Flatten the structure with duplicates based on the original list"""
    return [item for item, sublist in zip(original_list, list_of_lists) for _ in sublist]


@st.cache_data
def get_cluster_assignments(xb, k):
    """Perform K-means clustering on the embeddings"""
    kmeans = faiss.Kmeans(d=xb.shape[1], k=k, niter=20, verbose=True)
    kmeans.train(xb)
    _, labels = kmeans.index.search(xb, 1)
    return labels


@st.cache_data
def get_cleaned_data() -> tuple[pd.DataFrame, np.array]:
    search_engine = get_search_engine_instance()

    # Operate match_structure on the entire dataframe
    df_copy = search_engine.df.copy(deep=True)
    df_copy = df_copy.iloc[match_structure(df_copy.index.tolist(), search_engine.metadata['moa_summaries'])]
    df_copy = df_copy.reset_index(drop=True)
    df_copy['brand_name'] = df_copy['brand_name'].str.upper().str.strip()

    # Returns the original L2 normalized embeddings
    xb = search_engine.moa_faiss_index.reconstruct_n()

    assert len(df_copy['brand_name']) == xb.shape[0]

    # Remove duplicates by drug brand name
    unique_drugs, indices = np.unique(df_copy['brand_name'], return_index=True)
    xb = xb[indices]
    df_copy = df_copy.iloc[indices]
    return df_copy, xb


@st.cache_data
def get_tsne_projections(xb: np.array) -> np.array:
    tsne = TSNE(n_components=2, random_state=0)
    return tsne.fit_transform(xb)


@st.cache_data
def plot_moa_tsne(k=24) -> pd.DataFrame:
    df, xb = get_cleaned_data()

    # Perform K-means clustering to color the points
    color_map = px.colors.qualitative.Light24
    cluster_assignment = get_cluster_assignments(xb, k)
    colors = [color_map[label[0]] for label in cluster_assignment]  # uses K-means clustering
    df['cluster_assignment'] = cluster_assignment

    # TSNE
    projections = get_tsne_projections(xb)
    x_ = projections[:, 0]
    y_ = projections[:, 1]

    hover_text = [(f'Cluster #{row.cluster_assignment}<br><br>{row.brand_name}<br>'
                   f'<br>[FDA Pharmacologic Class Labels]<br>'
                   f'MOA: {row.pharm_class_moa}<br>Physiologic Effect: {row.pharm_class_pe}<br>'
                   f'Chemical Structure: {row.pharm_class_cs}<br>'
                   f'Established Pharmacologic Class: {row.pharm_class_epc}')
                  for row in df.itertuples()]
    marker_config = dict(color=colors, opacity=0.6)
    fig = go.Figure(data=go.Scatter(x=x_, y=y_, mode='markers', text=hover_text, hoverinfo='text', marker=marker_config))
    fig.update_traces(marker=dict(size=10, line=dict(width=1, color='DarkSlateGrey')))

    plot_title = 't-SNE Visualization of Mechanism of Action Embeddings'
    plot_subtitle = (f'Color represents unsupervised K-means clustering of the embeddings using k={k}.<br>'
                     'Embeddings are generated using section 12.1 (mechanism_of_action) of the FDA drug labels.')
    title_text = {'text': f'{plot_title}<br><sub>{plot_subtitle}</sub>'}

    # Inject custom CSS
    st.markdown("""<style>.stPlotlyChart {height: 90vh; width: 70vw}</style>""", unsafe_allow_html=True)
    grid = dict(showgrid=False, zeroline=False, showticklabels=False)
    fig = fig.update_layout(plot_bgcolor='white', title=title_text, xaxis=grid, yaxis=grid)

    st.plotly_chart(fig)

    return df
