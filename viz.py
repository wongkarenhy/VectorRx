import faiss
import numpy as np
import plotly.express as px
import plotly.graph_objs as go
from sklearn.manifold import TSNE
import streamlit as st

from search import DrugsSearchEngine


def match_structure(original_list, list_of_lists):
    """Flatten the structure with duplicates based on the original list"""
    return [item for item, sublist in zip(original_list, list_of_lists) for _ in sublist]


def get_cluster_assignments(xb, k):
    """Perform K-means clustering on the embeddings"""
    kmeans = faiss.Kmeans(d=xb.shape[1], k=k, niter=20, verbose=True)
    kmeans.train(xb)
    _, labels = kmeans.index.search(xb, 1)
    return labels


@st.cache_data
def plot_moa_tsne(k=24):
    search_engine = DrugsSearchEngine()
    drug_brand_names = match_structure(search_engine.df['brand_name'].tolist(), search_engine.moa_summaries)
    drug_moa = match_structure(search_engine.df['pharm_class_moa'].fillna('').tolist(), search_engine.moa_summaries)
    drug_pe = match_structure(search_engine.df['pharm_class_pe'].fillna('').tolist(), search_engine.moa_summaries)
    drug_cs = match_structure(search_engine.df['pharm_class_cs'].fillna('').tolist(), search_engine.moa_summaries)
    drug_epc = match_structure(search_engine.df['pharm_class_epc'].fillna('').tolist(), search_engine.moa_summaries)

    drug_brand_names = [d.upper() for d in drug_brand_names]

    # Returns the original L2 normalized embeddings
    xb = search_engine.moa_faiss_index.reconstruct_n()

    # Remove duplicates by drug brand name
    unique_drugs, indices = np.unique(drug_brand_names, return_index=True)
    xb = xb[indices]
    drug_moa = [drug_moa[i] for i in indices]
    drug_pe = [drug_pe[i] for i in indices]
    drug_cs = [drug_cs[i] for i in indices]
    drug_epc = [drug_epc[i] for i in indices]

    hover_text = [(f'{drug}<br><br>[FDA Pharmacologic Class Labels]<br>MOA: {moa}<br>Physiologic Effect: '
                   f'{pe}<br>Chemical Structure: {cs}<br>Established Pharmacologic Class: {epc}')
                  for drug, moa, pe, cs, epc in zip(unique_drugs, drug_moa, drug_pe, drug_cs, drug_epc)]

    # Let's perform K-means clustering to color the points
    color_map = px.colors.qualitative.Light24
    colors = [color_map[label[0]] for label in get_cluster_assignments(xb, k)]  # uses K-means clustering

    # TSNE
    tsne = TSNE(n_components=2, random_state=0)
    projections = tsne.fit_transform(xb)

    x_ = projections[:, 0]
    y_ = projections[:, 1]
    fig = go.Figure(data=go.Scatter(x=x_, y=y_, mode='markers', text=hover_text, hoverinfo='text',
                                    marker=dict(color=colors, opacity=0.5)))

    grid = dict(showgrid=False, zeroline=False, showticklabels=False)
    plot_title = 't-SNE Visualization of Mechanism of Action Embeddings'
    plot_subtitle = (f'Color represents unsupervised K-means clustering of the embeddings using k={k}. '
                     'Embeddings are generated using section 12.1 (mechanism_of_action) of the FDA drug labels.')
    title_text = {'text': f'{plot_title}<br><sub>{plot_subtitle}</sub>'}

    return fig.update_layout(width=1000, height=800, plot_bgcolor='white', title=title_text, xaxis=grid, yaxis=grid)
