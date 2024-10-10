"""This module contains functions to add context columns to the DataFrame used for Surprisal Estimates."""

import pandas as pd


def add_article_context_cols(
    et_df: pd.DataFrame, article_cols: list[str]
) -> pd.DataFrame:
    """
    Adds various context columns to the DataFrame based on article content.

    This function sequentially applies several transformations to add new columns
    that provide context about the articles in the DataFrame. These transformations
    include gathering and hunting related columns, dot fillers, and text fillers.
    Finally, it validates that the lengths of the newly added columns match the
    specified article columns.

    Parameters:
    - et_df (pd.DataFrame): The DataFrame containing article data.
    - article_cols (list[str]): A list of column names that uniquely identify an article.

    Returns:
    - pd.DataFrame: The modified DataFrame with new context columns added.
    """
    et_df = add_Gathering_Article_cols(et_df)
    et_df = add_Hunting_Article_cols(et_df)
    et_df = add_article_dot_fillers(et_df)
    et_df = add_article_text_fillers(et_df)
    validate_cols(et_df, article_cols)
    
    return et_df

def validate_cols(et_df: pd.DataFrame, article_cols: list[str]) -> pd.DataFrame:
    """
    Validates that the lengths of the newly added columns match the specified article columns.

    This function validates that the lengths of the newly added columns match the specified
    article columns. If the lengths do not match, it raises an error.

    Parameters:
    - et_df (pd.DataFrame): The DataFrame containing article data.
    - article_cols (list[str]): A list of column names that uniquely identify an article.

    Returns:
    - pd.DataFrame: The modified DataFrame with new context columns added.
    """
    for col in article_cols:
        if col not in et_df.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame.")
    return et_df

def add_prefixes(
    et_df: pd.DataFrame,
    cols: list[str],
    new_col_name: str,
    only_paragraphs_so_far: bool,
    reread_only: bool,
    add_suffix_space: bool,
    all_articles_so_far: bool,
) -> pd.DataFrame:
    """
    Adds a new column to the DataFrame by concatenating specified columns with optional spacing.

    This function creates a new column in the DataFrame by concatenating the contents of specified
    columns for each row. It supports adding the concatenated content for all rows up to the current
    one within the same article (only_paragraphs_so_far=True) or for all rows in the article.
    It also allows for adding a space after each concatenated set of columns.

    Parameters:
    - et_df (pd.DataFrame): The DataFrame to modify.
    - cols (list[str]): The list of column names to concatenate.
    - new_col_name (str): The name of the new column to be added to the DataFrame.
    - only_paragraphs_so_far (bool): If True, concatenate columns for all previous rows within the
        same article up to the current row. If False, concat columns for all rows in the article.
    - reread_only (bool): If True, replaces first reading with empty string.

    Returns:
    - pd.DataFrame: The modified DataFrame with the new column added.
    """
    def concatenate_with_spaces(series):
        return series.cumsum().shift(fill_value="").str.strip()

    if all_articles_so_far:
        deduplicated = et_df.drop_duplicates(
            subset=["subject_id", "article_ind"]
        ).copy()

        # Ensure paragraphs are ordered (should be the case already, but just in case)
        deduplicated = deduplicated.sort_values(
            by=["subject_id", "article_ind"]
        )

    else:
        deduplicated = et_df.drop_duplicates(
            subset=["subject_id", "reread", "unique_paragraph_id"]
        )

        # Ensure paragraphs are ordered (should be the case already, but just in case)
        deduplicated = deduplicated.sort_values(
            by=["subject_id", "reread", "article_ind", "paragraph_id"]
        )

    deduplicated[cols] = deduplicated[cols].fillna("")
    deduplicated["joined_columns"] = deduplicated[cols].apply(" ".join, axis=1)
    if add_suffix_space:
        deduplicated["joined_columns"] += " "

    if all_articles_so_far:
        deduplicated[new_col_name] = (
            deduplicated.groupby(["subject_id"])["joined_columns"]
            .apply(concatenate_with_spaces)
            .reset_index()["joined_columns"]
            .values
        )
        et_df = pd.merge(
            et_df,
            deduplicated[["subject_id", "article_ind", new_col_name]],
            on=["subject_id", "article_ind"],
            how="left",
        )

    elif only_paragraphs_so_far:
        deduplicated[new_col_name] = (
            deduplicated.groupby(["subject_id", "reread", "article_ind"])[
                "joined_columns"
            ]
            .apply(concatenate_with_spaces)
            .reset_index()["joined_columns"]
            .values
        )
        et_df = pd.merge(
            et_df,
            deduplicated[
                ["subject_id", "article_ind", "unique_paragraph_id", new_col_name]
            ],
            on=["subject_id", "article_ind", "unique_paragraph_id"],
            how="left",
        )
    else:
        articles = (
            deduplicated.groupby(["subject_id", "reread", "article_ind"])[
                "joined_columns"
            ]
            .apply(" ".join)
            .str.strip()
            .reset_index()
        )
        articles = articles.rename(columns={"joined_columns": new_col_name})
        et_df = pd.merge(
            et_df,
            articles,
            on=["subject_id", "reread", "article_ind"],
            how="left",
        )

    et_df[new_col_name] = et_df[new_col_name].fillna("").str.strip()
    if reread_only:
        et_df.loc[et_df["reread"] == 0, new_col_name] = ""
    return et_df


def add_prefixes_based_on_ops(
    et_df: pd.DataFrame,
    operations: list[dict],
) -> pd.DataFrame:
    """
    Applies a series of operations to add new columns to the DataFrame based on specified prefixes.

    This function iterates over a list of operations, where each op is a dictionary specifying
    the columns to concatenate, the name of the new column, and additional options. It uses these
    operations to add new columns to the DataFrame by concatenating specified columns for each row,
    optionally adding spaces and considering only paragraphs up to the current one.

    Parameters:
    - et_df (pd.DataFrame): The DataFrame to modify.
    - operations (list[dict]): A list of dictionaries, where each dictionary specifies an operation
        to apply. Each operation should include keys for 'cols' (list of column names to concat),
        'new_col_name' (name of new column), 'only_paragraphs_so_far' (bool indicating whether to
        concat columns only up to the current row within the same article), and 'add_space_after'
        (boolean indicating whether to add a space after each set of concatenated columns).

    Returns:
    - pd.DataFrame: The modified DataFrame with new columns added based on the specified operations.
    """
    for op in operations:
        if not all(col in et_df.columns for col in op["cols"]):
            continue
        et_df = add_prefixes(
            et_df,
            cols=op["cols"],
            new_col_name=op["new_col_name"],
            only_paragraphs_so_far=op["only_paragraphs_so_far"],
            reread_only=op["reread_only"],
            add_suffix_space=op.get("add_suffix_space", True),
            all_articles_so_far=op.get("all_articles_so_far", False),
        )
    return et_df

def add_Gathering_Article_cols(et_df: pd.DataFrame) -> pd.DataFrame:
    # et_df already contains cols: paragraph, question, Qfr, and all key cols

    # pastP
    # add column of all previous paragraphs untill current P (defined in col "new_unique_p_id")
    # all P of the current article (which is defined by columns: batch + article_ind)

    # Article
    # add column of all paragraphs of the current article (defined by columns: batch + article_ind)

    operations = [
        {
            "new_col_name": "pastP",
            "cols": ["paragraph"],
            "only_paragraphs_so_far": True,
            "reread_only": False,
        },
        {
            "new_col_name": "Article",
            "cols": ["paragraph"],
            "only_paragraphs_so_far": False,
            "reread_only": False,
            "add_suffix_space": False
        }
    ]
    
    et_df = add_prefixes_based_on_ops(et_df, operations)

    return et_df

def add_Hunting_Article_cols(et_df: pd.DataFrame) -> pd.DataFrame:
    # et_df already contains cols: paragraph, question, Qfr, and all key cols

    # pastQP
    # add column of all previous questions paragraphs untill current P
    # (defined in col "new_unique_p_id")
    # all P of the current article (which is defined by columns: batch + article_ind)
    # Qfr1 P1

    # ArticleQfrP
    # add column of all Qfr (question of first reading) and paragraphs of the current article
    # (defined by columns: batch + article_ind)
    # Qfr1 P1 Qfr2 P2 ... Qfr5 P5

    # ArticleQP
    # add column of all Q and paragraphs of all articles untill the current article
    # Q P Q P Q P ...

    operations = [
        {
            "new_col_name": "pastQP",
            "cols": ["question", "paragraph"],
            "only_paragraphs_so_far": True,
            "reread_only": False,
        },
        {
            "new_col_name": "ArticleQfrP",
            "cols": ["Qfr", "paragraph"],
            "only_paragraphs_so_far": False,
            "reread_only": True,
            "add_suffix_space": False,
        },
        {
            "new_col_name": "ArticleQP",
            "cols": ["question", "paragraph"],
            "only_paragraphs_so_far": False,
            "reread_only": False,
            "add_suffix_space": False,
        },
    ]

    et_df = add_prefixes_based_on_ops(et_df, operations)

    return et_df


def add_article_dot_fillers(et_df: pd.DataFrame) -> pd.DataFrame:
    operations = [
        {
            "new_col_name": "fillPastP",
            "cols": ["fillerP"],
            "only_paragraphs_so_far": True,
            "reread_only": False,
            "add_suffix_space": True
        },
        {
            "new_col_name": "fillPastQP",
            "cols": ["fillerQ", "fillerP"],
            "only_paragraphs_so_far": True,
            "reread_only": False,
            "add_suffix_space": True
        },
        {
            "new_col_name": "fillArticle",
            "cols": ["fillerP"],
            "only_paragraphs_so_far": False,
            "reread_only": False,
            "add_suffix_space": False
        },
        {
            "new_col_name": "fillArticleQfrP",
            "cols": ["fillerQfr", "fillerP"],
            "only_paragraphs_so_far": False,
            "reread_only": True,
            "add_suffix_space": False
        },
        {
            "new_col_name": "fillArticleQP",
            "cols": ["fillerQ", "fillerP"],
            "only_paragraphs_so_far": False,
            "reread_only": False,
            "add_suffix_space": False
        },
    ]

    et_df = add_prefixes_based_on_ops(et_df, operations)

    return et_df


def add_article_text_fillers(et_df: pd.DataFrame) -> pd.DataFrame:
    operations = [
        {
            "new_col_name": "textFillPastP",
            "cols": ["textFillP"],
            "only_paragraphs_so_far": True,
            "reread_only": False,
            "add_suffix_space": True
        },
        {
            "new_col_name": "textFillPastQP",
            "cols": ["textFillQ", "textFillP"],
            "only_paragraphs_so_far": True,
            "reread_only": False,
            "add_suffix_space": True
        },
        {
            "new_col_name": "textFillArticle",
            "cols": ["textFillP"],
            "only_paragraphs_so_far": False,
            "reread_only": False,
            "add_suffix_space": False
        },
        {
            "new_col_name": "textFillArticleQfrP",
            "cols": ["textFillQfr", "textFillP"],
            "only_paragraphs_so_far": False,
            "reread_only": True,
            "add_suffix_space": False
        },
        {
            "new_col_name": "textFillArticleQP",
            "cols": ["textFillQ", "textFillP"],
            "only_paragraphs_so_far": False,
            "reread_only": False,
            "add_suffix_space": False
        },
    ]

    et_df = add_prefixes_based_on_ops(et_df, operations)

    return et_df