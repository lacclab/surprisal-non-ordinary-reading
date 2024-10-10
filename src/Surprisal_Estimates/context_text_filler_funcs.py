from typing import Literal, Optional
import pandas as pd
from src.Data_Tools.eye_df_funcs import (
    add_q_fr_to_et_df_if_not_exist,
    add_p_q_to_et_df_if_not_exist,
    add_ids_to_et_df_if_not_exist
)
from src.Data_Tools.P_Q_df_funcs import (
    get_paragraphs_questions_df,
)
from src.constants import UNIQUE_ID_COL
from loguru import logger


def add_textual_filler_q_p(et_df: pd.DataFrame, subjects_q_df: Optional[pd.DataFrame]=None):
    paragraphs_df, questions_df = get_paragraphs_questions_df()
    paragraphs_df = paragraphs_df[['unique_paragraph_id', 'paragraph', 'new_unique_p_id']].drop_duplicates()
    paragraphs_df = paragraphs_df.assign(
        article_id = paragraphs_df["unique_paragraph_id"].str.split("_").str[1].astype(int)
    )
    et_df = add_ids_to_et_df_if_not_exist(et_df)
    et_df = add_q_fr_to_et_df_if_not_exist(
        et_df, subjects_q_df
    )
    et_df = add_p_q_to_et_df_if_not_exist(et_df)
    et_df = _add_text_fillers(et_df, paragraphs_df, questions_df)
    return et_df


def _add_text_fillers(
    eye_df: pd.DataFrame, paragraphs_df: pd.DataFrame, questions_df: pd.DataFrame
):
    print("_add_text_fillers")
    paragraphs_df = _add_len_col(paragraphs_df, col="P")
    questions_df = _add_len_col(questions_df, col="Q")
    eye_df = _add_text_filler_by_col(eye_df=eye_df, on="P", paragraphs_df=paragraphs_df)
    eye_df = _add_text_filler_by_col(eye_df=eye_df, on="Q", questions_df=questions_df)
    eye_df = _add_text_filler_by_col(eye_df=eye_df, on="Qfr", questions_df=questions_df)
    return eye_df


def _add_len_col(df: pd.DataFrame, col: Literal["P", "Q"]):
    if col == "P":
        df[f"len_{col}"] = df["paragraph"].str.split(" ").str.len()
    else:
        df[f"len_{col}"] = df["question"].str.split(" ").str.len()
    return df


def _add_text_filler_by_col(
    eye_df: pd.DataFrame,
    on: Literal["P", "Q", "Qfr"],
    paragraphs_df: pd.DataFrame = None,
    questions_df: pd.DataFrame = None,
    debug_mode: bool = False
) -> pd.DataFrame:
    if debug_mode:
        logger.debug("_add_text_filler_by_col ", on)
    if on == "P" and paragraphs_df is None:
        raise ValueError("paragraphs_df must be provided when col is 'P'")
    elif (on == "Q" or on == "Qfr") and questions_df is None:
        raise ValueError("questions_df must be provided when col is 'Q' or 'Qfr'")

    if on == "P":
        db_df = paragraphs_df.copy()
        text_col = "paragraph"
        p_or_q = "P"
        merge_on = UNIQUE_ID_COL["paragraph"]
    elif on == "Q":
        db_df = questions_df.copy()
        text_col = "question"
        p_or_q = "Q"
        merge_on = UNIQUE_ID_COL["question"]
    elif on == "Qfr":
        db_df = questions_df.copy()
        text_col = "question"
        p_or_q = "Q" # for searching in q db
        merge_on = UNIQUE_ID_COL["Qfr"]
    
    # Add a dummy column for the cartesian product
    db_df["key"] = 1
    # Perform the self-join
    merged_df = db_df.merge(db_df, on="key", suffixes=("_x", "_y")).drop(columns="key")

    # Filter out the same article_id pairs
    merged_df = merged_df[
        merged_df["article_id_x"] != merged_df["article_id_y"]
    ].reset_index(drop=True)
    # Calculate the length difference
    merged_df["len_diff"] = (
        merged_df[f"len_{p_or_q}_x"] - merged_df[f"len_{p_or_q}_y"]
    ).abs()
    # Find the minimum length difference for each paragraph
    text_col_id = UNIQUE_ID_COL[text_col]
    min_len_diff_idx = merged_df.groupby(f"{text_col_id}_x")["len_diff"].idxmin()

    # Extract the chosen ids and texts
    text_fillers = merged_df.loc[
        min_len_diff_idx, [f"{text_col_id}_x", f"{text_col_id}_y", f"{text_col}_y"]
    ]
    # Rename columns to match the desired output
    text_fillers.columns = [
        merge_on,
        f"textFill{on}_id",
        f"textFill{on}",
    ]
    if not debug_mode:
        text_fillers = text_fillers.drop(columns=[f"textFill{on}_id"])

    return eye_df.merge(text_fillers, on=merge_on, how="left")
