"""
This script estimates surprisal values based on different context configurations and model variants.
It reads the eye-tracking data, computes surprisal values, and saves the results with optional analysis columns.
The script utilizes multiple models and context configurations as defined in the argument inputs.
"""

import argparse
from typing import List

import pandas as pd
from loguru import logger

from src.Surprisal_Estimates.context_funcs import (
    add_columns_by_surp_variants,
    add_custom_surp_col,
    filter_et_data_by_surp_variant,
    get_surp_variant_name,
    get_unique_text_df,
    load_et_data,
    merge_curr_eye_df_with_surp_column,
    merge_eye_df_with_surp_cols,
    update_surp_cfg,
    validate_surp_cfg,
    validate_surp_col,
)

# import torch
from src.Surprisal_Estimates.context_params import get_surp_cfgs, get_surp_model_names
from src.utils import add_col_not_num_or_punc, validate_n_rows_in_two_dfs

logger.add("context_run.log")


def run(
    et_data_path: str,
    surp_cfg_list: List[str],
    surp_model_names: List[str],
    device: str,
    debug_mode: bool,
    context_overlap: int,
    hf_access_token: str = "",
) -> pd.DataFrame:
    # get et_df
    original_et_df = load_et_data(et_data_path, debug_mode)
    et_df = add_columns_by_surp_variants(original_et_df, surp_cfg_list)

    text_dfs_enriched_lst = []
    for curr_surp_variant in surp_cfg_list:
        # surp varaint params
        surp_cfg = update_surp_cfg(curr_surp_variant)
        surp_variant_name = get_surp_variant_name(surp_cfg)
        validate_surp_cfg(surp_cfg, surp_variant_name)
        logger.info(f"Run| {surp_variant_name}")

        # curr et df and text df
        curr_et_df = filter_et_data_by_surp_variant(surp_cfg, et_df)
        curr_text_df = get_unique_text_df(curr_et_df, surp_cfg.text_key_cols)
        logger.info(f"n rows curr_text_df: {len(curr_text_df)}")

        # try:
        # Run the Language Model to add the surprisal estimates
        text_df_with_surp = add_custom_surp_col(
            text_df=curr_text_df,
            surp_variant_cfg=surp_cfg,
            surprisal_extraction_model_names=surp_model_names,
            surp_col_name_suffix=surp_variant_name,
            surp_model_names=surp_model_names,
            device=device,
            context_overlap=context_overlap,
            hf_access_token=hf_access_token,
        )
        validate_surp_col(text_df_with_surp)
        # Merge to curr_et_df
        et_data_w_surp_variant = merge_curr_eye_df_with_surp_column(
            et_data=curr_et_df,
            text_df_with_surp=text_df_with_surp,
            surp_variant_cfg=surp_cfg,
            text_key_cols=surp_cfg.text_key_cols,
        )
        text_dfs_enriched_lst.append(et_data_w_surp_variant)

        # except Exception as e:
        #     logger.error(f"Error in {surp_variant_name}: {e}")

    logger.info("merge all eye_df_with_surp_cols...")
    eye_df_with_p_surp = merge_eye_df_with_surp_cols(
        et_df, text_dfs_enriched_lst, surp_cfg.target_col_id
    )
    validate_n_rows_in_two_dfs(eye_df_with_p_surp, original_et_df)
    return eye_df_with_p_surp


def parse_arguments():
    """
    Parses command-line arguments for script configuration.

    Returns:
    - Namespace: A namespace containing the parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Process context configuration flags.")
    parser.add_argument(
        "--article_context",
        action="store_true",
        help="Enable article context processing.",
    )
    parser.add_argument(
        "--regular_articles", action="store_true", help="Process regular articles."
    )
    parser.add_argument(
        "--fillers", action="store_true", help="Include fillers in processing."
    )

    parser.add_argument("--models", nargs="+", help="List of model names to process.")
    parser.add_argument("--config_indices", nargs="+", help="List of config indices.", type=int)
    # add debug mode
    parser.add_argument(
        "--debug_mode", action="store_true", help="Enable debug mode for processing."
    )

    return parser.parse_args()


if __name__ == "__main__":
    # Parse input arguments
    args = parse_arguments()

    # Access input arguments and define key parameters
    ARTICLE_CONTEXT = args.article_context
    REGULAR_ARTICLES = args.regular_articles
    FILLERS = args.fillers

    # Context overlap value used for context calculations
    CONTEXT_OVERLAP = 1567

    # File path configurations
    hf_access_token = "missing_token"  # TODO: Replace with your token for accessing Hugging Face models
    base_path = "ln_shared_data/onestop/processed/" # TODO: Replace with your data path
    et_data_path = base_path + "ia_data_enriched_360_05052024.csv"  # Input eye-tracking data path

    # Debugging and saving options
    save_with_analysis_cols = False
    debug_mode = args.debug_mode

    # Fetch model names to use for surprisal calculations
    surp_model_names = get_surp_model_names()
    surp_model_names = args.models  # Use models specified in the arguments
    logger.info(surp_model_names)

    # Create surprisal configuration list based on context and input arguments
    surp_cfg_list = get_surp_cfgs(
        article_context=ARTICLE_CONTEXT,
        fillers=FILLERS,
        regular_articles=REGULAR_ARTICLES,
    )

    # Update configurations and log skipped configurations if not in indices
    for index, surp_cfg in enumerate(surp_cfg_list):
        surp_cfg = update_surp_cfg(surp_cfg)
        surp_variant_name = get_surp_variant_name(surp_cfg)
        logger.info(f"Surp variant {index}: {surp_variant_name} {'skipped' if index not in args.config_indices else ''}")

    # Loop over configurations and models to estimate surprisal
    for index, surp_cfg in enumerate(surp_cfg_list):
        if index not in args.config_indices:    
            continue  # Skip configurations not included in indices
        for surp_model_name in surp_model_names:
            # Update and name configuration variant
            surp_cfg = update_surp_cfg(surp_cfg)
            surp_variant_name = get_surp_variant_name(surp_cfg)
            model_name = surp_model_name.replace("/", "-")  # Replace '/' for compatibility in filenames

            # Define result file name and output path
            result_file_name = f"et_20240505_{model_name}_AR={ARTICLE_CONTEXT}_RA={REGULAR_ARTICLES}_F={FILLERS}_{surp_variant_name}_20240625"
            if debug_mode:
                result_file_name += "_debug"
            output_path = base_path + f"{result_file_name}.csv"

            logger.info(f"Start running {surp_model_name}...")
            logger.info(f"Output path: {output_path}")

            # Run the surprisal estimation process
            eye_df_with_p_surp = run(
                et_data_path,
                [surp_cfg],
                [surp_model_name],
                device="cuda",  # Specify GPU usage
                debug_mode=debug_mode,
                context_overlap=CONTEXT_OVERLAP,
                hf_access_token=hf_access_token,
            )

            # Define key columns for merging and analysis
            key_cols = [
                "subject_id",
                "reread",
                "IA_ID",
                "IA_LABEL",
                "unique_paragraph_id",
                "new_unique_p_id",
            ]
            surprisal_cols = [
                col for col in eye_df_with_p_surp.columns if "Surprisal" in col
            ]

            # Optionally include additional analysis columns
            if save_with_analysis_cols:
                et_df = pd.read_csv(et_data_path)
                et_df = add_col_not_num_or_punc(et_df)  # Pre-process data
                full_key_cols = [
                    "subject_id",
                    "reread",
                    "IA_ID",
                    "unique_paragraph_id",
                    "has_preview",
                    "list",
                ]
                # Columns for deeper analysis and exploration
                analysis_cols = [
                    "has_preview",
                    "is_in_aspan",
                    "article_ind",
                    "normalized_ID",
                    "Wordfreq_Frequency",
                    "prev_Wordfreq_Frequency",
                    "Length",
                    "prev_Length",
                    "IA_FIRST_FIXATION_DURATION",
                    "IA_FIRST_RUN_DWELL_TIME",
                    "IA_DWELL_TIME",
                    "IA_FIRST_FIX_PROGRESSIVE",
                    "IA_REGRESSION_PATH_DURATION",
                    "not_num_or_punc",
                ]

                # Merge the computed surprisal data with analysis columns
                et_df = et_df[analysis_cols + full_key_cols]
                output_df = eye_df_with_p_surp.merge(
                    et_df, on=full_key_cols, how="left"
                )
                output_df = output_df[key_cols + surprisal_cols + analysis_cols]
            else:
                output_df = eye_df_with_p_surp[key_cols + surprisal_cols]

            # Save the results to the output path
            output_df.to_csv(
                output_path,
                index=False,
            )
            logger.info(f"Finished. Saved to {output_path}")
