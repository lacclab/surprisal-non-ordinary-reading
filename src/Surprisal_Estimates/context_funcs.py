from functools import partial
from typing import List
import pandas as pd
from src.Data_Tools.eye_df_funcs import (
    add_answers_to_et_df,
    add_ids_to_et_df_if_not_exist,
    add_p_q_to_et_df_if_not_exist,
    add_filler_columns_to_et_df,
)
from src.Data_Tools.P_Q_df_funcs import split_unique_paragraph_id
from src.constants import SurpVariantConfig
from src.Data_Tools.eye_df_funcs import add_q_fr_to_et_df_if_not_exist
from src.Surprisal_Estimates.context_instructions import add_instructions_cols_to_et_df
from src.utils import validate_n_rows_in_two_dfs
from loguru import logger
from text_metrics.merge_metrics_with_eye_movements import (
    extract_metrics_for_text_df_multiple_hf_models,
)
# if missing text metrics repo - install using https://github.com/lacclab/text-metrics.git
from src.constants import UNIQUE_ID_COL, P_Q_NAMES
from src.Surprisal_Estimates.context_text_filler_funcs import add_textual_filler_q_p
from src.Surprisal_Estimates.A_context_cols import add_article_context_cols


def load_et_data(et_data_path: str, debug_mode: bool):
    logger.info("Preprocess| Loading ET data...")
    original_et_df = pd.read_csv(et_data_path, engine="pyarrow")
    logger.info(f"Preprocess| n rows et_df: {len(original_et_df)}")
    if debug_mode:
        logger.debug("Debug| Sample data by 5% of the subject IDs")
        # Sample 10% of the subject IDs
        sampled_subject_ids = (
            original_et_df["subject_id"]
            .drop_duplicates()
            .sample(frac=0.05, random_state=42)
        )
        # Filter the DataFrame by the selected subject IDs
        original_et_df = original_et_df[
            original_et_df["subject_id"].isin(sampled_subject_ids)
        ].reset_index(drop=True)

    et_df = filter_cols_et_df(original_et_df)
    logger.info("Preprocess| Split unique_paragraph_id...")
    et_df = split_unique_paragraph_id(et_df)
    logger.info("Preprocess| Add ids cols...")
    et_df = add_ids_to_et_df_if_not_exist(et_df)

    logger.info("Preprocess| Add p, q, ans cols...")
    et_df = add_p_q_to_et_df_if_not_exist(et_df)
    et_df = add_q_fr_to_et_df_if_not_exist(et_df)
    et_df = add_answers_to_et_df(et_df)

    logger.info("Preprocess| Add instructions cols...")
    et_df = add_instructions_cols_to_et_df(et_df)

    validate_n_rows_in_two_dfs(et_df, original_et_df)
    return et_df


def filter_cols_et_df(
    et_df: pd.DataFrame,
) -> pd.DataFrame:
    select_cols = [
        "subject_id",
        "has_preview",
        "reread",
        "list",
        "unique_paragraph_id",
        "q_ind",
        "IA_LABEL",
        "IA_ID",
        "article_ind",
    ]
    return et_df[select_cols]


def filter_et_data_by_surp_variant(s_cfg: SurpVariantConfig, et_df: pd.DataFrame):
    if s_cfg.reread == 1:
        logger.debug("Run| filter reread == 1")
        return et_df.query("reread == 1")
    elif s_cfg.reread == 0:
        logger.debug("Run| filter reread == 0")
        return et_df.query("reread == 0")
    elif s_cfg.reread == "both":
        logger.debug("Run| no filter on reread")
        return et_df
    else:
        raise ValueError("s_cfg.reread with invalid value")


def add_columns_by_surp_variants(
    et_df: pd.DataFrame, surp_cfg_list: List[SurpVariantConfig]
):
    logger.info("Preprocess| add text filler cols...")
    et_df = add_textual_filler_q_p(et_df)
    # filler_cols = get_filler_cols_by_cfg_list(surp_cfg_list)
    logger.info("Preprocess| add filler cols...")
    filler_cols = ["fillerP", "fillerQ", "fillerQfr"]
    et_df = add_filler_columns_to_et_df(et_df, filler_cols)
    article_cols = get_article_context_cols(surp_cfg_list)# noqa: F821
    if len(article_cols) > 0:
        logger.info("Preprocess| add article context cols...")
        et_df = add_article_context_cols(et_df, article_cols)

    return et_df


def get_filler_cols_by_cfg_list(surp_cfg_list: List[SurpVariantConfig]):
    return list(
        set(
            [
                col
                for surp_variant_cfg in surp_cfg_list
                for col in surp_variant_cfg.ordered_prefix_cols
                if "filler" in col
            ]
        )
    )


def get_article_context_cols(surp_cfg_list: List[SurpVariantConfig]):
    return list(
        set(
            [
                col
                for surp_variant_cfg in surp_cfg_list
                if not surp_variant_cfg.is_paragraph_level  # article level
                for col in surp_variant_cfg.ordered_prefix_cols  # all columns we need
            ]
        )
    )


def get_surp_variant_name(cfg: SurpVariantConfig) -> str:
    name = (
        f"{'-'.join(cfg.ordered_prefix_cols)}-{cfg.target_col}"
        if cfg.ordered_prefix_cols
        else cfg.target_col
    )
    for s, replacement in P_Q_NAMES.items():
        name = name.replace(s, replacement)
    return name

def validate_surp_col(text_df_with_surp):
    surp_col = [col for col in text_df_with_surp.columns if 'Surprisal' in col][0]
    if all(text_df_with_surp[surp_col].isna()):
        logger.warning(f"all surp vals are None! surp_col: {surp_col}")

def validate_surp_cfg(surp_cfg: SurpVariantConfig, surp_variant_name: str):
    assert (
        surp_cfg.target_col not in surp_cfg.ordered_prefix_cols or surp_cfg.reread
    ), f"target_col {surp_cfg.target_col} is in provided as prefix to the main text {surp_cfg.ordered_prefix_cols} but mode is not reread_only_surp"


def get_unique_text_df(
    et_df: pd.DataFrame,
    text_key_cols: List[str],
) -> pd.DataFrame:
    return et_df[text_key_cols].drop_duplicates().reset_index(drop=True)


def update_surp_cfg(surp_cfg: SurpVariantConfig) -> SurpVariantConfig:
    surp_cfg.target_col_id = UNIQUE_ID_COL[surp_cfg.target_col]
    surp_cfg.text_key_cols = get_text_key_cols(surp_cfg)
    return surp_cfg


def get_text_key_cols(surp_cfg: SurpVariantConfig):
    return list(
        set(
            [x for x in surp_cfg.ordered_prefix_cols]
            + [surp_cfg.target_col, surp_cfg.target_col_id]
        )
    )


def add_custom_surp_col(
    text_df: pd.DataFrame,
    surp_variant_cfg: SurpVariantConfig,
    surprisal_extraction_model_names: List[str],
    surp_col_name_suffix: str,
    surp_model_names: List[str],
    device: str,
    context_overlap: int,
    hf_access_token: str,
):
    extract_metrics_function = get_extract_surp_partial_func(
        surp_variant_cfg, surp_model_names, device, hf_access_token
    )

    extract_metrics_for_text_df_kwargs = dict(
        get_metrics_kwargs=dict(context_stride=context_overlap)
    )
    extract_metrics_for_text_df_kwargs["ordered_prefix_col_names"] = (
        surp_variant_cfg.ordered_prefix_cols
    )
    extract_metrics_for_text_df_kwargs["keep_prefix_metrics"] = False
    extract_metrics_for_text_df_kwargs["rebase_index_in_main_text"] = True

    enriched_text_df = extract_metrics_function(
        text_df=text_df,
        extract_metrics_for_text_df_kwargs=extract_metrics_for_text_df_kwargs,
    )

    # rename the columns of enriched_text_df such that for all models in surprisal_extraction_model_names add surp_col_name_prefix
    for model_name in surprisal_extraction_model_names:
        enriched_text_df[f"{model_name}_Surprisal_{surp_col_name_suffix}"] = (
            enriched_text_df[f"{model_name}_Surprisal"]
        )
        enriched_text_df.drop(columns=[f"{model_name}_Surprisal"], inplace=True)

    # rename 'Word' column to 'IA_LABEL' to match the original text_df
    enriched_text_df.rename(columns={"Word": "IA_LABEL"}, inplace=True)
    # rename 'index' to 'IA_ID' to match the original text_df
    enriched_text_df.rename(columns={"index": "IA_ID"}, inplace=True)
    return enriched_text_df


def get_extract_surp_partial_func(
    surp_variant_cfg: SurpVariantConfig,
    surp_model_names: List[str],
    device: str,
    hf_access_token: str,
):
    # prepare the function that calls the Language Model
    return partial(
        extract_metrics_for_text_df_multiple_hf_models,
        surprisal_extraction_model_names=surp_model_names,
        text_key_cols=surp_variant_cfg.text_key_cols,
        text_col_name=surp_variant_cfg.target_col,
        model_target_device=device,
        add_parsing_features=False,
        hf_access_token=hf_access_token,
    )


def merge_curr_eye_df_with_surp_column(
    surp_variant_cfg: SurpVariantConfig,
    et_data: pd.DataFrame,
    text_df_with_surp: pd.DataFrame,
    text_key_cols: List[str],
) -> pd.DataFrame:
    """
    # text_df_with_surp_variant = text df + surp
    # current_et_data

    # merge current_et_data with text_df_with_surp_variant

    Args:
        merge_with_et_on_cols (List[str]): _description_
        surp_variant_cfg (SurpVariantConfig): _description_
        et_data (pd.DataFrame): _description_

    Returns:
        pd.DataFrame: _description_
    """
    merge_with_et_on_cols = ["IA_ID"] + [
        x for x in text_key_cols if x != surp_variant_cfg.target_col
    ]
    # leave only the columns that are required for the merge with ET and the surprisal columns
    text_df_with_surp = text_df_with_surp[
        merge_with_et_on_cols
        + [col for col in text_df_with_surp.columns if "Surprisal" in col]
    ]
    if surp_variant_cfg.reread != "both":
        # if the mode is repeated reading, then we want to merge only for these rows
        text_df_with_surp = text_df_with_surp.assign(reread=surp_variant_cfg.reread)
        merge_with_et_on_cols.append("reread")
    # remember! subject_id, unique_paragraph_id and reread define a trial
    # Thus, if we keep those 3 in et_data, we make sure that our merge is lossless (assuming left merge)
    return et_data[list(set(merge_with_et_on_cols + ["subject_id", "reread"]))].merge(
        text_df_with_surp, on=merge_with_et_on_cols, how="left"
    )


def merge_eye_df_with_surp_cols(
    et_df: pd.DataFrame, text_dfs_enriched_lst: List[pd.DataFrame], target_col_id: str
) -> pd.DataFrame:
    """
    merges a few et_data files

    Args:
        full_et_data (_type_): _description_
        text_dfs_enriched_lst (_type_): _description_
        main_text_identifier_col_name (_type_): _description_

    Returns:
        _type_: _description_
    """
    # Enriched_text_df is a merged dataframe of all dataframes in text_dfs_enriched_lst on common columns. Use reduce
    word_in_trial_key_cols = [
        "subject_id",
        "reread",
        "IA_ID",
        target_col_id,
    ]
    all_surp_df = union_text_dfs_enriched_lst(
        text_dfs_enriched_lst, word_in_trial_key_cols
    )

    return et_df.merge(
        all_surp_df,
        on=word_in_trial_key_cols,
        how="left",
    )


def union_text_dfs_enriched_lst(text_dfs_enriched_lst, word_in_trial_key_cols):
    # Enriched_text_df is a dataframe of union all dataframes in text_dfs_enriched_lst
    # look at the unit test for exaple

    # Initialize the result DataFrame with the first DataFrame in the list
    result_df = text_dfs_enriched_lst[0]

    # Iterate over the remaining DataFrames and merge them with the result DataFrame
    for df in text_dfs_enriched_lst[1:]:
        result_df = pd.merge(
            result_df, df, on=word_in_trial_key_cols, how="outer", suffixes=("_x", "_y")
        )

        # Combine the columns with suffixes and remove duplicates
        for col in result_df.columns:
            if col.endswith("_x") or col.endswith("_y"):
                base_col = col[:-2]
                if base_col in result_df.columns:
                    result_df[base_col] = result_df[base_col].combine_first(
                        result_df[col]
                    )
                else:
                    result_df[base_col] = result_df[col]
                result_df.drop(columns=[col], inplace=True)

    # keep only columns word_in_trial_key_cols and surprisal columns
    surprisal_cols = [col for col in result_df.columns if "Surprisal" in col]
    return result_df[word_in_trial_key_cols + surprisal_cols]
