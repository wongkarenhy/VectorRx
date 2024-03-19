import unittest
from unittest.mock import patch, MagicMock

import faiss
import numpy as np
import pandas as pd
from st_files_connection import FilesConnection

from vectorrx.search import DrugsSearchEngine  # Adjust the import statement based on your project structure


class TestDrugsSearchEngine(unittest.TestCase):
    def setUp(self):
        # Create fake drug data
        self.fake_df = pd.DataFrame({
            'brand_name': ['DrugA', 'DrugB', 'DrugC'],
            'pharm_class_moa': ['MOA1', 'MOA2', 'MOA3'],
            'pharm_class_pe': ['PE1', 'PE2', 'PE3'],
            'pharm_class_cs': ['CS1', 'CS2', 'CS3'],
            'pharm_class_epc': ['EPC1', 'EPC2', 'EPC3'],
            'Application Number': ['BLA123', 'ANDA456', 'part789'],
            'moa': ['moa1', 'moa2', 'moa1'],
        })

        self.fake_model = 'fake_embedding_model'
        self.fake_last_updated = '123456'
        self.fake_moa_summaries = [['summary1'], ['summary2'], ['summary3']]

        # Patch the __get_drug_data method to return the fake data
        return_value = (self.fake_df, self.fake_model, self.fake_moa_summaries, self.fake_last_updated)
        get_drug_data_patcher = patch.object(DrugsSearchEngine, '_DrugsSearchEngine__get_drug_data',
                                             return_value=return_value)
        self.mock_get_drug_data = get_drug_data_patcher.start()
        self.addCleanup(get_drug_data_patcher.stop)

        # Patch st.secrets to provide a fake API key
        secrets_patcher = patch('vectorrx.search.st.secrets', {'OPENAI_API_KEY': 'fake_api_key'})
        self.mock_secrets = secrets_patcher.start()
        self.addCleanup(secrets_patcher.stop)

        # Patch faiss.read_index to mock loading a FAISS index
        self.fake_faiss_index = faiss.index_factory(128, 'Flat', faiss.METRIC_INNER_PRODUCT)
        np.random.seed(0)
        search_embedding = np.random.rand(3, 128).astype(np.float32)
        faiss.normalize_L2(search_embedding)
        self.fake_faiss_index.add(search_embedding)
        self.fake_faiss_index.ntotal = 3
        self.fake_faiss_index.d = 128
        get_moa_faiss_index_patcher = patch.object(DrugsSearchEngine, '_DrugsSearchEngine__get_moa_faiss_index',
                                                   return_value=self.fake_faiss_index)
        self.mock_faiss_index = get_moa_faiss_index_patcher.start()
        self.addCleanup(get_moa_faiss_index_patcher.stop)

        # Patch get_embedding to mock the OpenAI API call for the query string
        np.random.seed(0)
        self.fake_query_embedding = np.random.rand(1, 128).astype(np.float32)
        get_embedding_patcher = patch('vectorrx.search.DrugsSearchEngine.get_embedding', return_value=self.fake_query_embedding)
        self.mock_get_embedding = get_embedding_patcher.start()
        self.addCleanup(get_embedding_patcher.stop)

    def test_search_most_similar_moa(self):
        engine = DrugsSearchEngine()
        faiss.normalize_L2(self.fake_query_embedding)
        result = engine.search_most_similar_moa(self.fake_query_embedding, 2)
        expected_drugs = ['DrugA', 'DrugC']
        expected_moa_cosine_similarity = [1.000, 0.768]

        self.assertEqual(result.shape, (2, 8))
        self.assertEqual(result['brand_name'].tolist(), expected_drugs)
        for val1, val2 in zip(result['moa_cosine_similarity'].tolist(), expected_moa_cosine_similarity):
            self.assertAlmostEqual(val1, val2, places=3)

    def test_query(self):
        engine = DrugsSearchEngine()

        # Test basic use case with query string as input
        result = engine.query(query_string='test_string', k=2, similarity_threshold=.4, in_keyword='', ex_keyword='')
        self.assertEqual(result.shape, (2, 8))

        # Test raising ValueError when neither query string nor include/exclude keywords are provided
        with self.assertRaises(ValueError):
            engine.query(query_string='', k=2, similarity_threshold=.4, in_keyword='', ex_keyword='')

        # Test a different similarity threshold
        result = engine.query(query_string='test_string', k=2, similarity_threshold=.9, in_keyword='', ex_keyword='')
        self.assertEqual(result.shape, (1, 8))

        # Test in_keyword search only
        result = engine.query(query_string='', k=0, similarity_threshold=0, in_keyword='moa1', ex_keyword='')
        self.assertEqual(result.shape, (2, 8))

        # Test query_string and ex_keyword search
        result = engine.query(query_string='test_string', k=3, similarity_threshold=0, in_keyword='', ex_keyword='moa1')
        self.assertEqual(result.shape, (1, 8))


if __name__ == '__main__':
    unittest.main()
