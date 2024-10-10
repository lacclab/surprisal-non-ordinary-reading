import unittest
from src.Surprisal_Estimates.context_text_filler_funcs import _add_text_filler_by_col
from src.Surprisal_Estimates.context_funcs import union_text_dfs_enriched_lst, get_surp_variant_name
from src.constants import SurpVariantConfig
import pandas as pd
from time import time

class TestAddTextFillerByCol(unittest.TestCase):
    def setUp(self):
        # Sample data for paragraphs_df
        self.paragraphs_df = pd.DataFrame(
            {
                "article_id": [1, 2, 3, 3],
                "new_unique_p_id": [101, 102, 103, 103_2],
                "len_P": [3, 4, 5, 5],
                "paragraph": ["a b c", "a b c d", "a b c d e", "a b c d e"],
            }
        )

        # Sample data for questions_df
        self.questions_df = pd.DataFrame(
            {
                "article_id": [1, 2, 3, 3],
                "unique_q_id": [201, 202, 203, 203_2],
                "len_Q": [3, 4, 2, 2],
                "question": ["1 2 3", "1 2 3 4", "1 2", "1 2"],
            }
        )

        # Sample data for eye_df
        self.eye_df = pd.DataFrame(
            {
                "new_unique_p_id": [101, 102, 103],
                "unique_q_id": [201, 202, 203],
                "unique_Qfr_id": [202, 203, 203_2],
                "data": ["Data 1", "Data 2", "Data 3"],
            }
        )

    def test_add_text_filler_paragraphs(self):
        result_df = _add_text_filler_by_col(
            self.eye_df, "P", paragraphs_df=self.paragraphs_df, debug_mode=True
        )

        expected_df = pd.DataFrame(
            {
                "new_unique_p_id": [101, 102, 103],
                "unique_q_id": [201, 202, 203],
                "unique_Qfr_id": [202, 203, 203_2],
                "data": ["Data 1", "Data 2", "Data 3"],
                "textFillP_id": [102, 101, 102],
                "textFillP": ["a b c d", "a b c", "a b c d"],
            }
        )

        pd.testing.assert_frame_equal(result_df, expected_df[result_df.columns])

    def test_add_text_filler_questions(self):
        result_df = _add_text_filler_by_col(
            self.eye_df, "Q", questions_df=self.questions_df, debug_mode=True
        )
        expected_df = pd.DataFrame(
            {
                "unique_paragraph_id": [101, 102, 103],
                "unique_q_id": [201, 202, 203],
                "unique_Qfr_id": [202, 203, 203_2],
                "data": ["Data 1", "Data 2", "Data 3"],
                "textFillQ_id": [202, 201, 201],
                "textFillQ": ["1 2 3 4", "1 2 3", "1 2 3"],
            }
        )

        pd.testing.assert_frame_equal(result_df, expected_df[result_df.columns])

    def test_add_text_filler_q_fr(self):
        result_df = _add_text_filler_by_col(
            self.eye_df, "Qfr", questions_df=self.questions_df, debug_mode=True
        )

        expected_df = pd.DataFrame(
            {
                "unique_paragraph_id": [101, 102, 103],
                "unique_q_id": [201, 202, 203],
                "unique_Qfr_id": [202, 203, 203_2],
                "data": ["Data 1", "Data 2", "Data 3"],
                "textFillQfr_id": [201, 201, 201],
                "textFillQfr": ["1 2 3", "1 2 3", "1 2 3"],
            }
        )

        pd.testing.assert_frame_equal(result_df, expected_df[result_df.columns])

    def test_missing_paragraphs_df(self):
        with self.assertRaises(ValueError):
            _add_text_filler_by_col(self.eye_df, "P", questions_df=self.questions_df)

    def test_missing_questions_df(self):
        with self.assertRaises(ValueError):
            _add_text_filler_by_col(self.eye_df, "Q", paragraphs_df=self.paragraphs_df)


class TestUnionTexsDfsEnriched(unittest.TestCase):
    def test_union_text_dfs_enriched_lst(self):
        # Example DataFrames with additional common columns
        df1 = pd.DataFrame({
            'word': ['apple', 'banana', 'cherry'],
            'trial': [1, 2, 3],
            'value1': [10, 20, 30],
            'common_col1': ['A', 'B', 'C'],
            'common_col2': ['X', 'Y', 'Z']
        })

        df2 = pd.DataFrame({
            'word': ['apple', 'banana', 'date'],
            'trial': [1, 2, 4],
            'value2': [100, 200, 400],
            'common_col2': ['X', 'Y', 'D']
        })

        df3 = pd.DataFrame({
            'word': ['apple', 'elderberry', 'fig'],
            'trial': [1, 5, 6],
            'value3': [1000, 5000, 6000],
            'common_col1': ['A', 'E', 'F'],
        })

        # List of DataFrames to merge
        text_dfs_enriched_lst = [df1, df2, df3]
        # Columns to merge on
        word_in_trial_key_cols = ['word', 'trial']

        start = time()
        enriched_et_df = union_text_dfs_enriched_lst(text_dfs_enriched_lst, word_in_trial_key_cols)
        print(time() - start)
        expected_df = pd.read_csv('src/tests/test_Surprisal_Estimates/test_union_text_dfs_enriched_lst_expected.csv')

        pd.testing.assert_frame_equal(enriched_et_df, expected_df[enriched_et_df.columns])

class TestGetSurpVariantName(unittest.TestCase):
    
    def test_paragraph(self):
        cfg = SurpVariantConfig(
            target_col="paragraph",
            ordered_prefix_cols=[], 
            is_paragraph_level=True, 
            reread="both",
        )
        result = get_surp_variant_name(cfg)
        self.assertEqual(result, "P")
        
    def test_question_paragraph(self):
        cfg = SurpVariantConfig(
            target_col="paragraph",
            ordered_prefix_cols=["question"], 
            is_paragraph_level=True, 
            reread=0, 
        )
        result = get_surp_variant_name(cfg)
        self.assertEqual(result, "Q-P")
        
    def test_paragraph_paragraph(self):
        cfg = SurpVariantConfig(
            target_col="paragraph",
            ordered_prefix_cols=["paragraph"], 
            is_paragraph_level=True, 
            reread=1, 
        )
        result = get_surp_variant_name(cfg)
        self.assertEqual(result, "P-P")
        
    def test(self):
        # Q'-P-Q-P
        cfg = SurpVariantConfig(
            target_col = "paragraph",
            ordered_prefix_cols=["Qfr", "paragraph", "question"], 
            is_paragraph_level=True, 
            reread=1, 
        )
        result = get_surp_variant_name(cfg)
        self.assertEqual(result, "Qfr-P-Q-P")

if __name__ == '__main__':
    unittest.main()
