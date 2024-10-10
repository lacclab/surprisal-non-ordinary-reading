import unittest
import pandas as pd
from src.Surprisal_Estimates.A_context_cols import (
    add_Gathering_Article_cols,
    add_Hunting_Article_cols,
    add_article_dot_fillers,
    add_article_text_fillers)

class TestAddArticleCols(unittest.TestCase):

    def setUp(self):
        # Sample DataFrame setup
        self.expected_df = pd.read_excel("src/tests/test_Surprisal_Estimates/test_add_Article_cols_data.xlsx")
        self.expected_df = self.expected_df.fillna("")
        
        input_cols = [
            "test_goal",
            "subject_id",
            "unique_paragraph_id",
            "paragraph_id",
            "reread",
            "article_ind",
            "paragraph",
            "question",
            "Qfr",
            "fillerQ", "fillerQfr", "fillerP",
            "textFillQ", "textFillQfr", "textFillP",
            "instReReadS"
            ]
        new_cols = [
            "pastP",
            "pastQP",
            "Article",
            "ArticleQP", "ArticleQfrP",
            "fillPastP", "fillPastQP", 
            "fillArticle", "fillArticleQP", "fillArticleQfrP", 
            "textFillPastP", "textFillPastQP",
            "textFillArticle", "textFillArticleQP", "textFillArticleQfrP",	
        ]
        self.et_df = self.expected_df[input_cols]
        self.expected_df = self.expected_df[input_cols+new_cols]
        
    def test_add_Gathering_Article_cols(self):
        result_df = add_Gathering_Article_cols(self.et_df)
        pd.testing.assert_frame_equal(result_df, self.expected_df[result_df.columns])

    def test_add_Hunting_Article_cols(self):
        result_df = add_Hunting_Article_cols(self.et_df)
        pd.testing.assert_frame_equal(result_df, self.expected_df[result_df.columns])
        
    def test_add_article_dot_fillers(self):
        result_df = add_article_dot_fillers(self.et_df)
        pd.testing.assert_frame_equal(result_df, self.expected_df[result_df.columns])

    def test_add_article_text_fillers(self):
        result_df = add_article_text_fillers(self.et_df)
        pd.testing.assert_frame_equal(result_df, self.expected_df[result_df.columns])


if __name__ == '__main__':
    unittest.main()
