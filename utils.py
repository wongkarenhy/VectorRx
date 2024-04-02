import re

import pandas as pd
from plotly import express as px
import streamlit as st


def __format_link(x) -> str:
    """Extract only digits from the string."""
    x = re.sub(r'\D', '', x)
    return f'https://www.accessdata.fda.gov/scripts/cder/daf/index.cfm?event=overview.process&ApplNo={x}'


def __clean_up_column_names(column_names: list) -> list[str]:
    return [c.replace('_', ' ').title().replace('Moa', 'MOA').replace('Ar', 'AR') for c in column_names]


def clean_up(df: pd.DataFrame, records: int) -> pd.DataFrame:
    df.columns = __clean_up_column_names(df.columns)

    # Make the FDA Link clickable
    df['FDA Link'] = [__format_link(x) if not pd.isna(x) else None for x in df['Application Number']]
    df = df.drop(columns=['Application Number']).reset_index(drop=True)

    # Move FDA Link to the first column
    cols = ['FDA Link'] + [c for c in df.columns if c != 'FDA Link']
    df = df[cols]
    return df.head(records)


def add_histograms(df: pd.DataFrame):
    for x in ['Cosine Similarity MOA', 'Cosine Similarity AR']:
        if x in df.columns:
            fig = px.histogram(df, x=x, nbins=20)
            plot_title = f'Distribution of {x} scores from the search results'
            fig.update_layout(width=800, height=200, plot_bgcolor='white', title=plot_title)
            fig.update_traces(marker=dict(line=dict(width=2, color='white')))
            st.plotly_chart(fig)
