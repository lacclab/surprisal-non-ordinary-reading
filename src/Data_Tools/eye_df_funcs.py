import pandas as pd
import numpy as np
import ast
from pathlib import Path
import re
from typing import List, Optional
from src.constants import FILLER_COLS_DICT
from loguru import logger
from src.Data_Tools.P_Q_df_funcs import get_subjects_df, get_paragraphs_questions_df

def add_p_q_to_et_df_if_not_exist(et_df: pd.DataFrame)->pd.DataFrame:
    if ("paragraph" in et_df.columns) and ("question" in et_df.columns):
        return et_df
    
    p_df, q_df = get_paragraphs_questions_df()
    if "paragraph" not in et_df.columns:
        et_df = et_df.merge(
            p_df[[
                "unique_paragraph_id", "has_preview", "list", 
                "paragraph", "new_unique_p_id", "p_version"]], 
            on=["unique_paragraph_id", "has_preview", "list"],
            how="left")
    if "question" in et_df.columns:
        et_df = et_df.drop(columns=["question"])
        # there is a bug in the current question col in et_df
    et_df = et_df.merge(
        q_df[["unique_q_id", "question"]], 
        on="unique_q_id",
        how="left")
    return et_df

def add_ids_to_et_df_if_not_exist(et_df: pd.DataFrame)->pd.DataFrame:
    if "unique_paragraph_id" not in et_df.columns:
        et_df = et_df.assign(
            unique_paragraph_id=(
                et_df["batch"].astype("str")
                + "_"
                + et_df["article_id"].astype("str")
                + "_"
                + et_df["level"]
                + "_"
                + et_df["paragraph_id"].astype("str")
            )
        )
    if "unique_q_id" not in et_df.columns:
        et_df = et_df.assign(
            unique_q_id = (
                et_df["batch"].astype("str")
                + "_"
                + et_df["article_id"].astype("str")
                + "_"
                + et_df["paragraph_id"].astype("str")
                + "_"
                + et_df["q_ind"].astype("str")
            )
        )
    if ("unique_Qfr_id" not in et_df.columns) and ("q_ind_fr" in et_df.columns):
        et_df = et_df.assign(
            unique_Qfr_id = (np.where(
                pd.notna(et_df["q_ind_fr"]),
                et_df["batch"].astype("str")
                + "_"
                + et_df["article_id"].astype("str")
                + "_"
                + et_df["paragraph_id"].astype("str")
                + "_"
                + et_df["q_ind_fr"].astype("str"),
                np.nan)
            )
        )
    return et_df

def add_q_fr_to_et_df_if_not_exist(
    et_df: pd.DataFrame, subjects_q_df: Optional[pd.DataFrame]=None, debug_mode: bool=False
)->pd.DataFrame:
    "For each record in et_data (IA report) add the question the first reading"
    if "Qfr" in et_df.columns:
        return et_df
    logger.info("Preprocess | Add first question to et data...")

    if subjects_q_df is None:
        subjects_q_df = get_subjects_df(et_df)
    
    # for each subject_id and unique_paragraph_id 
    #   get q_ind for first reading (0: q_ind_fr) and repeated reding (1: q_ind_rr)
    #   table: unique_paragraph_id	| subject_id | q_ind_fr	| q_ind_rr
    subjects_q_inds_df = (
        et_df[["unique_paragraph_id", "subject_id", "reread", "q_ind"]]
        .drop_duplicates()
        .pivot(
            index=["unique_paragraph_id", "subject_id"],
            columns="reread",
            values="q_ind",
        )
        .reset_index()
        .dropna() # keep only repeated reading
        .rename(columns={0: "q_ind_fr", 1: "_q_ind_rr"})
    )
    subjects_q_inds_df["q_ind_fr"] = subjects_q_inds_df["q_ind_fr"].astype('Int64')
    subjects_q_inds_df["_q_ind_rr"] = subjects_q_inds_df["_q_ind_rr"].astype('Int64')
    

    # add first reading question from q_text_df
    subjects_Qfr_df = (
        subjects_q_inds_df.merge(
            subjects_q_df,
            left_on=["subject_id", "unique_paragraph_id", "q_ind_fr"],
            right_on=["subject_id", "unique_paragraph_id", "q_ind"],
            how="left",
        )
        .drop(columns=["q_ind"])
        .rename(columns={"question": "Qfr"})
    )
    subjects_Qfr_df["reread"] = 1

    if not debug_mode: # drop helper cols
        subjects_Qfr_df.drop(columns=[col for col in subjects_Qfr_df.columns if col.startswith('_')], inplace=True)
    
    # merge back to et
    return et_df.merge(
        subjects_Qfr_df,
        on=["unique_paragraph_id", "subject_id", "reread"],
        how="left",
    ).fillna(pd.NA)
    
def add_filler_columns_to_et_df(et_df: pd.DataFrame, filler_cols: List[str])->pd.DataFrame:
    """Add to df a filler columns.
        filler column is '.' as the number of words in the original col"""
    for filler_col in filler_cols:
        original_col_filler = FILLER_COLS_DICT[filler_col]
        et_df[filler_col] = et_df[original_col_filler].apply(
            lambda text: " ".join(["." for _ in text.split()]) if pd.notna(text) else ""
        )
    return et_df

def add_answers_to_et_df(
    et_df: pd.DataFrame, 
    all_dat_files_path: Path = "ln_shared_data/onestop/processed/all_dat_files_merged.tsv", 
    inplace=False
) -> pd.DataFrame | None:
    """This function adds the answers to the IA report DataFrame

    Args:
        et_df (pd.DataFrame):Eye tracking df assumed to have the following columns:
            batch, has_preview, list, reread, article_id, paragraph_id, q_ind, answer_order
        all_dat_files_path (str): Path to the all_dat_files.csv file
        inplace (bool, optional): Whether to modify the input DataFrame or return a new one. Defaults to False.

    Returns:
        pd.DataFrame | None: The IA report DataFrame with the answers added if inplace is False
    """
    all_dat_files_df = pd.read_csv(
        all_dat_files_path,
        sep="\t",
    )

    # Assuming all_dat_files_df is your DataFrame
    answers = all_dat_files_df[["a", "b", "c", "d"]].values.tolist()
    answers_orders = (
        all_dat_files_df["answers_order"]
        .apply(lambda x: ast.literal_eval(x.replace(" ", ", ")))
        .tolist()
    )

    ordered_answers = []

    for i in range(len(answers)):
        ordered_answers.append([answers[i][index] for index in answers_orders[i]])

    all_dat_files_df["answers"] = ordered_answers

    all_dat_files_df = all_dat_files_df.query("practice == 0")[
        [
            "batch",
            "has_preview",
            "list",
            "reread",
            "article_id",
            "paragraph_id",
            "q_ind",
            "answers_order",
            "answers",
        ]
    ]
    all_dat_files_df["answers"] = all_dat_files_df["answers"].apply(
        lambda x: ["a) " + x[0], "b) " + x[1], "c) " + x[2], "d) " + x[3]]
    )
    all_dat_files_df["answers"] = all_dat_files_df["answers"].apply(
        lambda x: ". ".join(x)
    )
    all_dat_files_df["has_preview"] = all_dat_files_df["has_preview"].apply(
        lambda x: "Hunting" if x == 1 else "Gathering"
    )

    if "answers_order" in et_df.columns:
        et_df = et_df.drop(columns=["answers_order"])
    
    if inplace:
        ia_rep_merged = et_df
    else:
        ia_rep_merged = et_df.copy()

        ia_rep_merged = ia_rep_merged.merge(
            all_dat_files_df,
            on=[
                "batch",
                "has_preview",
                "list",
                "reread",
                "article_id",
                "paragraph_id",
                "q_ind",
            ],
            how="left",
            validate="m:1",
        )
    if not inplace:
        return ia_rep_merged
    # Now ordered_answers should contain the ordered answers

def exclude_IAs(
    df: pd.DataFrame,
    remove_start_end_of_line: bool = True,
    remove_non_all_letters_words: bool = True,
) -> pd.DataFrame:
    et_data_enriched = df.copy()
    # ? Remove first and last words in each paragraph
    # For every unique_paragraph_id, subject_id, reread triplet, find the maximal and minimal IA_IDs
    # and remove the records with the minimal and maximal IA_ID
    min_IA_IDs = (
        et_data_enriched.groupby(["unique_paragraph_id", "subject_id", "reread"])[
            "IA_ID"
        ]
        .min()
        .reset_index()
    )
    max_IA_IDs = (
        et_data_enriched.groupby(["unique_paragraph_id", "subject_id", "reread"])[
            "IA_ID"
        ]
        .max()
        .reset_index()
    )

    # remove from et_data_enriched the records with ['unique_paragraph_id', 'subject_id', 'reread', 'IA_ID'] in min_IA_IDs
    et_data_enriched = et_data_enriched.merge(
        min_IA_IDs,
        on=["unique_paragraph_id", "subject_id", "reread", "IA_ID"],
        how="left",
        indicator=True,
    )
    et_data_enriched = et_data_enriched[et_data_enriched["_merge"] == "left_only"]
    et_data_enriched = et_data_enriched.drop(columns=["_merge"])

    # remove from et_data_enriched the records with ['unique_paragraph_id', 'subject_id', 'reread', 'IA_ID'] in max_IA_IDs
    et_data_enriched = et_data_enriched.merge(
        max_IA_IDs,
        on=["unique_paragraph_id", "subject_id", "reread", "IA_ID"],
        how="left",
        indicator=True,
    )
    et_data_enriched = et_data_enriched[et_data_enriched["_merge"] == "left_only"]
    et_data_enriched = et_data_enriched.drop(columns=["_merge"])

    # ? Remove words that are not all letters (contains numbers or symbols inclusind punctuation)
    et_data_enriched = et_data_enriched.loc[
        et_data_enriched["IA_LABEL"].apply(lambda x: bool(re.match("^[a-zA-Z ]*$", x)))
    ]

    if remove_start_end_of_line:
        # if 'end of line' column is in the dataframe, remove all rows where 'end of line' == 1
        if "end_of_line" in et_data_enriched.columns:
            et_data_enriched = et_data_enriched.query("end_of_line != True")

        if "start_of_line" in et_data_enriched.columns:
            et_data_enriched = et_data_enriched.query("start_of_line != True")

    return et_data_enriched
