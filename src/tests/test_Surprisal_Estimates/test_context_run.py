import unittest
from src.Surprisal_Estimates.debug_surp_val import get_surp_on_text
from src.Surprisal_Estimates.context_run import run
from src.constants import SurpVariantConfig

class TestContextRun(unittest.TestCase):
    def setUp(self):
        # path params
        self.et_data_path = (
            "/data/home/shared/onestop/processed/ia_data_enriched_360_05052024.csv"
        )
        self.debug_mode = True
        
    def test_P_context_with_gpt2(self):
        surp_model_name = 'gpt2'
        surp_cfg_list = [
            # P
            SurpVariantConfig(
                target_col = "paragraph",
                ordered_prefix_cols=[], 
                is_paragraph_level=True, 
                reread="both",
            )
        ]
        
        # Call the function
        eye_df_with_p_surp = run(
            et_data_path=self.et_data_path,
            surp_cfg_list=surp_cfg_list, 
            surp_model_names=[surp_model_name], 
            device="cuda:1", # or cude:0
            debug_mode=self.debug_mode
            )
        
        # check results
        p_df = eye_df_with_p_surp[['paragraph', 'new_unique_p_id']].drop_duplicates().reset_index(drop=True)
        # sample 10 examples and compare
        samples = p_df.sample(n=10)
        for _, row in samples.iterrows():
            p_text = row['paragraph']
            p_id = row['new_unique_p_id']
            expected_surp_vals = get_surp_on_text(p_text, surp_model_name)
            result_surp_vals = eye_df_with_p_surp[eye_df_with_p_surp['new_unique_p_id'] == p_id][['unique_paragraph_id', 'new_unique_p_id', 'IA_ID', 'IA_LABEL', 'gpt2_Surprisal_P']].drop_duplicates().reset_index(drop=True)
            self.assertTrue(all(result_surp_vals['gpt2_Surprisal_P'] == expected_surp_vals['Surprisal']))
        
    def test_textFillP_P_with_gpt2(self):
        surp_model_name = 'gpt2'
        surp_cfg_list = [
            # textFillP-P
            SurpVariantConfig(
                target_col = "paragraph",
                ordered_prefix_cols=["textFillP"], 
                is_paragraph_level=True, 
                reread=1, 
            )
        ]
        
        # Call the function
        eye_df_with_p_surp = run(
            et_data_path=self.et_data_path,
            surp_cfg_list=surp_cfg_list, 
            surp_model_names=[surp_model_name], 
            device="cuda:1", # or cude:0
            debug_mode=self.debug_mode
            )
            
        # check results
        eye_df_with_p_surp['context_with_P'] = eye_df_with_p_surp['textFillP'] + " " + eye_df_with_p_surp['paragraph']
        p_df = eye_df_with_p_surp[eye_df_with_p_surp["reread"] == 1][['context_with_P', 'new_unique_p_id', 'unique_paragraph_id', 'textFillP', 'paragraph']].drop_duplicates().reset_index(drop=True)
        # sample 10 examples and compare
        samples = p_df.sample(n=10)
        for _, row in samples.iterrows():
            p_text = row['context_with_P']
            p_id = row['new_unique_p_id']
            result_surp_vals = (eye_df_with_p_surp[
                (eye_df_with_p_surp['new_unique_p_id'] == p_id) & (eye_df_with_p_surp['reread'] == 1)
                ][['reread', 'unique_paragraph_id', 'new_unique_p_id', 'IA_ID', 'IA_LABEL', 'gpt2_Surprisal_textFillP-P']]
                .drop_duplicates()
                .reset_index(drop=True)
            )
            expected_surp_vals = get_surp_on_text(p_text, surp_model_name).iloc[-len(result_surp_vals):].reset_index(drop=True)

            self.assertTrue(all(result_surp_vals['gpt2_Surprisal_textFillP-P'] == expected_surp_vals['Surprisal']))

