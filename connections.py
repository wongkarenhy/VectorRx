import io
import streamlit as st
from google.oauth2 import service_account
from google.cloud import storage


def get_cred():
    return service_account.Credentials.from_service_account_info(
        {
            "type": st.secrets['GCS_TYPE'],
            "project_id": st.secrets['GCS_PROJECT_ID'],
            "private_key_id": st.secrets['GCS_PRIVATE_KEY_ID'],
            "private_key": st.secrets['GCS_PRIVATE_KEY'].replace('\\n', '\n'),
            "client_email": st.secrets['GCS_CLIENT_EMAIL'],
            "client_id": st.secrets['GCS_CLIENT_ID'],
            "auth_uri": st.secrets['GCS_AUTH_URI'],
            "token_uri": st.secrets['GCS_TOKEN_URI'],
            "auth_provider_x509_cert_url": st.secrets['GCS_AUTH_PROVIDER_X509_CERT_URL'],
            "client_x509_cert_url": st.secrets['GCS_CLIENT_X509_CERT_URL'],
        }
    )


@st.cache_resource
def get_storage_client() -> storage.Client:
    """
    Gets the storage client with service account credentials.
    """
    return storage.Client(project=st.secrets['GCS_PROJECT_ID'], credentials=get_cred())


def get_blob(bucket_name: str, path: str) -> storage.Blob:
    bucket = get_storage_client().get_bucket(bucket_name)
    return bucket.blob(path)


def get_contents(bucket_name: str, path: str) -> io.BytesIO:
    file_obj = io.BytesIO()
    blob = get_blob(bucket_name, path)
    blob.download_to_file(file_obj)
    file_obj.seek(0)
    return file_obj
