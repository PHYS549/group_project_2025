import sys
import os
import unittest
import os
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock

script_dir = os.path.dirname(__file__)
src_path = os.path.abspath(os.path.join(script_dir, '..', 'gw_grb_correlation/Fermi'))
sys.path.insert(0, src_path)
from example import download_and_preprocess_fermi_data

class TestExampleFunctions(unittest.TestCase):

    def test_download_and_preprocess_fermi_data(self):
        # Mock the preprocess functions to avoid actual data downloading
        from unittest.mock import patch

        with patch('example.download_and_preprocess_time_data', return_value=None), \
             patch('example.download_and_preprocess_location_data', return_value=None), \
             patch('example.download_and_preprocess_tte_data', return_value=None), \
             patch('example.download_and_preprocess_poshist_data', return_value=None), \
             patch('example.download_and_preprocess_trigger_data', return_value=None), \
             patch('example.create_dataframe_and_name_column_from_data_files') as mock_create_df, \
             patch('example.merge_all_datatypes_in_fermi', return_value=pd.DataFrame({'ID': [1], 'TSTART': [0], 'TSTOP': [10], 'T90': [5], 'DATE': ['2025-04-16']})):

            # Mock the DataFrame creation
            mock_create_df.side_effect = lambda data_type: pd.DataFrame({'ID': [1], 'TSTART': [0], 'TSTOP': [10], 'T90': [5], 'DATE': ['2025-04-16']})

            # Call the function
            result = download_and_preprocess_fermi_data(2015, 2026, download_or_not=False)

            # Assert the result is a DataFrame
            self.assertIsInstance(result, pd.DataFrame)
            self.assertFalse(result.empty)
if __name__ == '__main__':
    unittest.main()