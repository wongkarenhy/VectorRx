import plotly.express as px
import streamlit as st
from search import DrugsSearchEngine


st.set_page_config(page_title='Vector Rx', page_icon=':pill:', layout='wide')

st.subheader(':pill: Search FDA Approved Drugs by Mechanism of Action')

search_engine = DrugsSearchEngine()
st.write(f'[Data Source: OpenFDA](https://open.fda.gov/data/downloads/) - Last Accessed: {search_engine.last_updated}')

st.write("""
This is a search engine for FDA approved drugs that allows you to search a drug by its mechanism of action.\
There are two ways to perform a search:
1. Enter a mechanism of action as free text and set a similarity threshold. This method uses the underlying \
semantics of your query to find similar drugs. Higher similarity threshold will return fewer but more \
semantically similar drugs. You can also include or exclude keywords to narrow down the results.
2. Leave the mechanism of action field empty and use the include and exclude keywords to search for drugs.

**Why embeddings?** Embeddings are a way to represent text as vectors in a high-dimensional space. They provide a \
nuanced search capability by not strictly adhering to exact terms, enabling users to adjust the similarity threshold \
to explore drugs with related actions. However, it's worth noting that this could increase the chance of \
encountering false positives, as it broadens the search to include a wide array of potentially related things.
""")

moa_query = st.text_input('Mechanism of Action Query [free text]', 'something that blocks topoisomerase')

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
    df = search_engine.query(moa_query, num_records, similarity_threshold, moa_in, moa_ex)

    if not df.empty:
        # Add a histogram of the cosine similarity scores
        warning_msg = ('Please be aware that text embeddings are based on the underlying semantics of the text and '
                       'there are no hard boundaries for similarity scores. Please adjust the threshold accordingly.')
        st.write(f'⚠️ {warning_msg}')

        fig = px.histogram(df, x='Moa Cosine Similarity', nbins=20)
        plot_title = 'Distribution of cosine similarity scores from the search results'
        fig.update_layout(width=800, height=200, plot_bgcolor='white', title=plot_title)
        fig.update_traces(marker=dict(line=dict(width=2, color='white')))  # add white border to bars
        st.plotly_chart(fig)

        help_mgs = "Click to open the FDA page for this drug"
        fda_link = st.column_config.LinkColumn(label='FDA Link', display_text='.*=(\\d+)', help=help_mgs)
        st.data_editor(df, column_config={'FDA Link': fda_link})
    else:
        st.write('No results found. Please try a different query.')
        st.stop()
