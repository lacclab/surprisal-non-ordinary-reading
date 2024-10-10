import unittest
import pandas as pd

from src.GAM.plot_funcs.GAM_plot_grid_funcs import _agg_valid_means

class TestAggregateValidMeans(unittest.TestCase):
    def setUp(self):
        # Create a sample DataFrame for testing
        self.df = pd.DataFrame({
            'group': ['A', 'A', 'B', 'B'],
            'surp_p_val': [0.1, 1.2, -0.5, 0.8],
            'prev_surp_p_val': [0.2, 0.3, 0.4, 0.5],
            'corrected_surp_p_val': [0.3, 0.4, 0.2, -0.1],
            'corrected_prev_surp_p_val': [0.4, -0.3, 1.1, 0.6]
        })
        self.group_by_cols = ['group']
    
    def test_aggregate_valid_means(self):
        result = _agg_valid_means(self.df, self.group_by_cols)
        expected_data = {
            'group': ['A', 'B'],
            'surp_p_val': [0.1, 0.8],
            'prev_surp_p_val': [0.25, 0.45],
            'corrected_surp_p_val': [0.35, 0.2],
            'corrected_prev_surp_p_val': [0.4, 0.6]
        }
        expected_df = pd.DataFrame(expected_data)
        pd.testing.assert_frame_equal(result, expected_df)

if __name__ == '__main__':
    unittest.main()
