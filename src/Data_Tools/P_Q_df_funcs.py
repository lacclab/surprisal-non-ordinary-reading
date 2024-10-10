import pandas as pd
from loguru import logger
from tqdm import tqdm

def get_paragraphs_questions_df(
    all_dat_path = "ln_shared_data/onestop/processed/all_dat_files_merged.tsv",
    p_df_path: str = "ln_shared_data/onestop/processed/p_from_et_20240624.csv"
):
    all_dat_files = pd.read_csv(all_dat_path, sep="\t").query("article_id > 0")

    # make all numeric columns numeric
    all_dat_files = all_dat_files.apply(pd.to_numeric, errors="ignore")

    # new key cols: unique_paragraph_id, list, has_preview
    p_df = pd.read_csv(p_df_path)

    # Columns: question keys: ["batch", "article_id", "paragraph_id", "q_ind", "question"], question: "question"
    q_df = all_dat_files[
        ["batch", "article_id", "paragraph_id", "q_ind", "question"]
    ].drop_duplicates()
    q_df = add_unique_q_id(q_df)
    return p_df, q_df

def add_unique_paragraph_id(df: pd.DataFrame):
    logger.warning("This is not really a unique id! use new_uniqe_p_id")
    df["unique_paragraph_id"] = (
        df["batch"].astype("str")
        + "_"
        + df["article_id"].astype("str")
        + "_"
        + df["level"]
        + "_"
        + df["paragraph_id"].astype("str")
    )
    return df
    
def add_unique_q_id(df: pd.DataFrame):
    df["unique_q_id"] = (
        df["batch"].astype("str")
        + "_"
        + df["article_id"].astype("str")
        + "_"
        + df["paragraph_id"].astype("str")
        + "_"
        + df["q_ind"].astype("str")
    )
    return df

def get_subjects_df(full_et_df):
    key_cols = ["subject_id", "unique_paragraph_id", "q_ind", "question"]
    return full_et_df[key_cols].drop_duplicates()

def split_unique_paragraph_id(df: pd.DataFrame) -> pd.DataFrame | None:
    """This function splits the unique_paragraph_id column into batch, article_id, level, and paragraph_id columns.

    Args:
        df (pd.DataFrame): The DataFrame to split the unique_paragraph_id column.

    Returns:
        _type_: _description_
    """
    split_columns = df["unique_paragraph_id"].str.split("_", expand=True)

    # Assign the resulting columns to the DataFrame
    return df.assign(
        batch=split_columns[0].astype(int),
        article_id=split_columns[1].astype(int),
        level=split_columns[2],
        paragraph_id=split_columns[3].astype(int)
    )
    

def fix_p_df_and_save(
    save_path: str,
    dat_files_path: str ="/data/home/shared/onestop/processed/all_dat_files_merged.tsv",
    corrections_path: str = "src/Data_Tools/consecutive_same_values_unique_2.csv"
    ):
    all_dat_files = pd.read_csv(dat_files_path, sep="\t").query("article_id > 0")
    all_dat_files = all_dat_files.apply(pd.to_numeric, errors="ignore")
    all_dat_files = add_unique_paragraph_id(all_dat_files)
    
    # corrections df
    c_df = pd.read_csv(corrections_path)
    c_df = add_unique_paragraph_id(c_df)

    fixed_dat_files = _get_fixed_dat_files(all_dat_files, c_df)
    _sanity_check_dat_files(all_dat_files, fixed_dat_files)
    
    no_duplicates_p = (
        fixed_dat_files[['unique_paragraph_id', 'paragraph']]
        .drop_duplicates().reset_index(drop=True)
        .sort_values(by=['unique_paragraph_id']))
    
    no_dup_p_with_new_id = _create_new_unique_p_id(no_duplicates_p)
    
    fixed_dat_files = fixed_dat_files.merge(no_dup_p_with_new_id, on=['unique_paragraph_id', 'paragraph'], how="left")
    fixed_dat_files.to_csv(save_path) 

def validate_fixed_p(
    fixed_p_df: pd.DataFrame,
    et_data_path: str = (
        "/data/home/shared/onestop/processed/ia_data_enriched_360_05052024.csv"
    )
    ):
    et_df = pd.read_csv(et_data_path)
    et_df_no_dup = et_df[['has_preview', 'unique_paragraph_id', 'list', 'IA_ID', "IA_LABEL"]].drop_duplicates()
    n_df = et_df_no_dup.groupby(['has_preview', 'unique_paragraph_id', 'list'])['IA_ID'].count().reset_index().rename(columns={"IA_ID": "n_IA"})
    n_df_dat_files = fixed_p_df[['has_preview', 'unique_paragraph_id', 'list', 'len_P']].drop_duplicates()
    n_df_dat_files["has_preview"] = n_df_dat_files["has_preview"].apply(lambda v: "Gathering" if v==0 else "Hunting")
    check_n_df = n_df.merge(n_df_dat_files, on=['has_preview', 'unique_paragraph_id', 'list'])
    check_n_df["check"] = (check_n_df["n_IA"] == check_n_df["len_P"])
    check_n_df[~check_n_df["check"]]
    check_n_df[~check_n_df["check"]][['unique_paragraph_id', "n_IA", "len_P"]].drop_duplicates()
    
def get_p_from_et(
    et_data_path: str = (
        "/data/home/shared/onestop/processed/ia_data_enriched_360_05052024.csv"
    )):
    et_df = pd.read_csv(et_data_path)
    p_from_et = (et_df[["unique_paragraph_id", "IA_LABEL", "IA_ID", "has_preview", "list"]]
        .drop_duplicates()
        .sort_values(by=["has_preview", "list", "unique_paragraph_id", "IA_ID"])
        .groupby(["unique_paragraph_id", "has_preview", "list"])["IA_LABEL"].apply(list))
    p_from_et = p_from_et.apply(lambda text: " ".join(text)).reset_index().rename(columns={"IA_LABEL": "paragraph"})
    p_from_et["len_et_p"] = p_from_et["paragraph"].str.split().str.len()
    
    no_duplicates_p = p_from_et[["unique_paragraph_id", "paragraph"]].drop_duplicates()
    no_duplicates_p = _create_new_unique_p_id(no_duplicates_p)
    
    return p_from_et.merge(no_duplicates_p, on=["unique_paragraph_id", "paragraph"])

def _create_new_unique_p_id(no_duplicates_p):
    no_duplicates_p['p_version'] = no_duplicates_p.groupby(['unique_paragraph_id']).cumcount() + 1
    no_duplicates_p['new_unique_p_id'] = (
        no_duplicates_p['unique_paragraph_id'] + "_" + 
        no_duplicates_p['p_version'].astype(str))
    return no_duplicates_p

def _sanity_check_dat_files(all_dat_files: pd.DataFrame, fixed_dat_files: pd.DataFrame):
    # example for p_id which need to be fixed
    p_id = '1_6_Adv_4'
    
    all_dat_files['len_P'] = all_dat_files['paragraph'].str.split().str.len()
    fixed_dat_files['len_P'] = fixed_dat_files['paragraph'].str.split().str.len()
    len1 = all_dat_files[all_dat_files['unique_paragraph_id']==p_id]['len_P'].drop_duplicates().item()
    len2 = fixed_dat_files[fixed_dat_files['unique_paragraph_id']==p_id]['len_P'].drop_duplicates().item()
    assert(len1 != len2)
    
def _get_fixed_dat_files(all_dat_files: pd.DataFrame, c_df: pd.DataFrame):
    new_p_list = []
    for example in tqdm(
        all_dat_files.itertuples(),
        total=len(all_dat_files),
        desc="Tokenizing",
    ):
        # TODO consider using merge instead of iterating through examples to find matches
        matches = c_df[
            (c_df["paragraph_id"] == example.paragraph_id)
            & (c_df["list"] == example.list)
            & (c_df["level"] == example.level)
            & (c_df["batch"] == example.batch)
            & (c_df["has_preview"] == example.has_preview)
            & (c_df["article_id"] == example.article_id)
        ]
        paragraph = example.paragraph
        if len(matches) > 0:
            for match_ in matches.itertuples():
                paragraph = _fix_paragraph(
                    paragraph,  # type: ignore
                    match_.IA_LABEL,  # type: ignore
                    match_.IA_ID,  # type: ignore
                )
        
        new_p_list.append(pd.DataFrame([{
            "paragraph": paragraph,
            "paragraph_id": example.paragraph_id,
            "unique_paragraph_id": example.unique_paragraph_id,
            "list": example.list,
            "level": example.level,
            "batch": example.batch,
            "has_preview": example.has_preview,
            "article_id": example.article_id,
        }]))
    
    return pd.concat(new_p_list, ignore_index=True)
    
def _fix_paragraph(paragraph: str, word: str, ia_id: int) -> str:
    # Split the paragraph into words
    words = paragraph.split()

    # Check if the word at the given index is the word to replace
    for i in range(ia_id - 3, ia_id + 1):
        if i < 0 or i >= len(words):
            continue
        if words[i] == word:
            # Replace the word
            if word == "6.30am;":  # TODO remove hardcoding
                words[i] = "6.30 am;"
            words[i] = words[i].replace("-", "- ", 1)
            break

    # Join the words back into a paragraph
    new_paragraph = " ".join(words)

    return new_paragraph

if __name__ == "__main__":
    # fix_p_df_and_save(save_path="/data/home/shared/onestop/processed/fixed_dat_files_20240624.csv")
    # fixed_dat_files = pd.read_csv("/data/home/shared/onestop/processed/fixed_dat_files_20240624.csv")
    # validate_fixed_p(fixed_dat_files)
    p_df = get_p_from_et()
    p_df.to_csv("/data/home/shared/onestop/processed/p_from_et_20240624.csv")