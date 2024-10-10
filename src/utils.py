import pandas as pd
import re

def validate_n_rows_in_two_dfs(df1: pd.DataFrame, df2: pd.DataFrame):
    n_rows_df1 = len(df1)
    n_rows_df2 = len(df2)
    
    if n_rows_df1 != n_rows_df2:
        raise ValueError(
            f"Row count mismatch: df1 has {n_rows_df1} rows, "
            f"while df2 has {n_rows_df2} rows."
        )
        
def add_col_not_num_or_punc(df: pd.DataFrame):
    df["not_num_or_punc"] = df["IA_LABEL"].apply(lambda x: bool(re.match("^[a-zA-Z ]*$", x)))
    return df
