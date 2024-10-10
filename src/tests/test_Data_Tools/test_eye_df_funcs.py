import unittest
import pandas as pd
from pandas.testing import assert_frame_equal

# Assuming the function add_q_fr_to_et_data is defined in the module et_module
from src.Data_Tools.eye_df_funcs import add_q_fr_to_et_df_if_not_exist

class TestAddQFrToEtData(unittest.TestCase):
    def setUp(self):
        # Create sample input data with simpler IDs and question text
        self.full_et_df = pd.read_excel("src/tests/test_Data_Tools/test_add_q_fr_to_et_data_full_et_df.xlsx")
        self.expected_result = pd.read_excel("src/tests/test_Data_Tools/test_add_q_fr_to_et_data_expected.xlsx")
        self.expected_result = self.expected_result.fillna(pd.NA)
        self.expected_result["q_ind_fr"] = self.expected_result["q_ind_fr"].astype("Int64")
        self.expected_result["_q_ind_rr"] = self.expected_result["_q_ind_rr"].astype("Int64")
    
    def test_add_q_fr_to_et_data(self):
        # Call the function
        result = add_q_fr_to_et_df_if_not_exist(self.full_et_df, debug_mode=True)
        # Assert that the result matches the expected DataFrame
        assert_frame_equal(result, self.expected_result)

