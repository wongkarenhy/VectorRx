import streamlit as st
import tempfile
from google.cloud import storage


def create_gcs_client():
    creds = st.secrets['connections']['gcs']
    client = storage.Client.from_service_account_info(creds)
    return client


def get_blob_from_gcs(bucket_name, source_blob_name):
    client = create_gcs_client()
    bucket = client.get_bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    return blob


def download_file_from_gcs(bucket_name, source_blob_name):
    blob = get_blob_from_gcs(bucket_name, source_blob_name)

    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        blob.download_to_filename(temp_file.name)
        return temp_file.name  # Returns the path to the temporary file
