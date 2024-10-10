import pandas as pd

def add_instructions_cols_to_et_df(et_df: pd.DataFrame):
    instructions_df = pd.read_csv("src/Surprisal_Estimates/context_instructions_df.csv")
    
    # Merge the instructions with et_df
    return et_df.merge(
        instructions_df, how="left", on=["has_preview"]
    )
