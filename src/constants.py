# create an enum surp_presentation_num with the value RR and FR
from enum import Enum
from typing import List, Optional, Literal
from dataclasses import dataclass


class surp_mode(Enum):
    RR = 1
    FR = 2


@dataclass
class SurpVariantConfig:
    target_col: str # the name of column for which we will extract surprisal values.
    ordered_prefix_cols: List[str] # what are the columns that are entered as context
    is_paragraph_level: bool # paragraph or article level
    reread: Literal[1, 0, "both"] # run only on repeated reading or not
    ordinary_reading_instructions: Optional[bool] = None # gathering (ordinary) or hunting (infromation seeking) instructions
    target_col_id: Optional[str] = None
    text_key_cols: Optional[List[str]] = None

P_Q_NAMES = {
    "paragraph": "P",
    "question": "Q",
}

UNIQUE_ID_COL = {
    "paragraph": "new_unique_p_id",
    "question": "unique_q_id",
    "Qfr": "unique_Qfr_id"
}

FILLER_COLS_DICT = {
    "fillerP": "paragraph",
    "fillerQ": "question",
    "fillerQfr": "Qfr",
    "fillerAns": "answers",
    "fillPastP": "pastP",
    "fillPastQP": "PastQP",
    "fillArticle": "Article",
    "fillArticleQfrP": "ArticleQfrP",
}

TEXT_FILLER_COLS_DICT = {
    "textFillP": "paragraph",
    "textFillQ": "question",
    "textFillQfr": "Qfr",
    "textFillAns": "answers",
    "textFillPastP": "pastP",
    "textFillPastQP": "PastQP",
    "textFillArticle": "Article",
    "textFillArticleQfrP": "ArticleQfrP",
}