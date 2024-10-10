from src.constants import SurpVariantConfig


def get_surp_model_names():
    """
    Get the list of model names for which surprisal estimates are to be computed.
    Options:
        * Llama - 5 models
        * Mistral - 3 models
        * GPT2 - 4 models
        * GPT-J - 1 model
        * GPT-Neo - 3 models
        * OPT - 4 models
        * Pythia - 7 models
        * Gemma - 3 models
    In total, 30 models.
    
    Returns:
        _type_: _description_
    """
    return [
        # "meta-llama/Llama-2-7b-hf",
        # "meta-llama/Llama-2-7b-chat-hf"
        # "meta-llama/Llama-2-13b-hf",
        # "meta-llama/Llama-2-13b-chat-hf"
        # "meta-llama/Llama-2-70b-hf"

        # "mistralai/Mistral-7B-v0.1",
        # "mistralai/Mistral-7B-v0.3",
        # "mistralai/Mistral-7B-Instruct-v0.3",
        
        # "gpt2",
        # "gpt2-medium",
        # "gpt2-large",
        # "gpt2-xl",
        
        # "EleutherAI/gpt-j-6B",
        
        # "EleutherAI/gpt-neo-125M",
        # "EleutherAI/gpt-neo-1.3B",
        # "EleutherAI/gpt-neo-2.7B",
        
        # "facebook/opt-350m",
        # "facebook/opt-1.3b",
        # "facebook/opt-2.7b",
        # "facebook/opt-6.7b",
        
        "EleutherAI/pythia-70m",
        # "EleutherAI/pythia-160m",
        # "EleutherAI/pythia-410m",
        # "EleutherAI/pythia-1b",
        # "EleutherAI/pythia-1.4b",
        # "EleutherAI/pythia-2.8b",
        # "EleutherAI/pythia-6.9b",
        
        # "google/gemma-7b"
        # "google/gemma-2-9b",  # Requires transformers >=4.42.3
        # "google/recurrentgemma-9b"
    ]


def get_surp_cfgs(
    article_context: bool, fillers: bool, regular_articles: bool
):
    # ---- Article context ----
    if article_context:
        configs = []

        if regular_articles:
            configs += [
                # Gathering: current article first reading
                # pastP-P
                SurpVariantConfig(
                    target_col="paragraph",
                    ordered_prefix_cols=["pastP"],
                    is_paragraph_level=False,
                    reread="both",
                ),
                # Hunting: current article first reading
                # pastQP-Q-P
                SurpVariantConfig(
                    target_col="paragraph",
                    ordered_prefix_cols=["pastQP", "question"],
                    is_paragraph_level=False,
                    reread=0,
                ),
                # Gathering: current article first and repeated reading
                # Article-pastP-P
                SurpVariantConfig(
                    target_col="paragraph",
                    ordered_prefix_cols=["Article", "pastP"],
                    is_paragraph_level=False,
                    reread=1,
                ),
                # Hunting: current article first and repeated reading
                # ArticleQfrP-pastQP-Q-P
                SurpVariantConfig(
                    target_col="paragraph",
                    ordered_prefix_cols=["ArticleQfrP", "pastQP", "question"],
                    is_paragraph_level=False,
                    reread=1,
                ),
                # ---- instructions ----
                # Gathering: current article first reading
                # instFirstReadA-pastP-P
                SurpVariantConfig(
                    target_col="paragraph",
                    ordered_prefix_cols=["instFirstReadA", "pastP"],
                    is_paragraph_level=False,
                    reread="both",
                ),
                # Hunting: current article first reading
                # instFirstReadA-pastQP-P
                SurpVariantConfig(
                    target_col="paragraph",
                    ordered_prefix_cols=["instFirstReadA", "pastQP", "question"],
                    is_paragraph_level=False,
                    reread=0,
                ),
                # Gathering: current article first and repeated reading
                # instFirstReadA-Article-instReReadA-pastP-P
                SurpVariantConfig(
                    target_col="paragraph",
                    ordered_prefix_cols=[
                        "instFirstReadA",
                        "Article",
                        "instReReadA",
                        "pastP",
                    ],
                    is_paragraph_level=False,
                    reread=1,
                ),
                # Hunting: current article first and repeated reading
                # instFirstReadA-ArticleQfrP-instReReadA-pastQP-Q-P
                SurpVariantConfig(
                    target_col="paragraph",
                    ordered_prefix_cols=[
                        "instFirstReadA",
                        "ArticleQfrP",
                        "instReReadA",
                        "pastQP",
                        "question",
                    ],
                    is_paragraph_level=False,
                    reread=1,
                ),
            ]

        if fillers:
            configs += [
                # ---- fillers ----
                # Gathering: current article first reading
                # fillerpastP-P
                SurpVariantConfig(
                    target_col="paragraph",
                    ordered_prefix_cols=["fillPastP"],
                    is_paragraph_level=False,
                    reread=0,
                ),
                # Hunting: current article first reading
                # fillerPastQP-fillerQ-P
                SurpVariantConfig(
                    target_col="paragraph",
                    ordered_prefix_cols=["fillPastQP", "fillerQ"],
                    is_paragraph_level=False,
                    reread=0,
                ),
                # Gathering: current article first and repeated reading
                # fillArticle-fillPastP-P
                SurpVariantConfig(
                    target_col="paragraph",
                    ordered_prefix_cols=["fillArticle", "fillPastP"],
                    is_paragraph_level=False,
                    reread=1,
                ),
                # Hunting: current article first and repeated reading
                # fillerArticleQfrP-fillPastQP-fillerQ-P
                SurpVariantConfig(
                    target_col="paragraph",
                    ordered_prefix_cols=["fillArticleQfrP", "fillPastQP", "fillerQ"],
                    is_paragraph_level=False,
                    reread=1,
                ),
                # ---- text fillers ----
                # Gathering: current article first reading
                # textFillPastP-P
                SurpVariantConfig(
                    target_col="paragraph",
                    ordered_prefix_cols=["textFillPastP"],
                    is_paragraph_level=False,
                    reread=0,
                ),
                # Hunting: current article first reading
                # textFillPastQP-textFillQ-P
                SurpVariantConfig(
                    target_col="paragraph",
                    ordered_prefix_cols=["textFillPastQP", "textFillQ"],
                    is_paragraph_level=False,
                    reread=0,
                ),
                # Gathering: current article first and repeated reading
                # textFillArticle-textFillPastP-P
                SurpVariantConfig(
                    target_col="paragraph",
                    ordered_prefix_cols=["textFillArticle", "textFillPastP"],
                    is_paragraph_level=False,
                    reread=1,
                ),
                # Hunting: current article first and repeated reading
                # textFillArticleQfrP-textFillPastQP-textFillQ-P
                SurpVariantConfig(
                    target_col="paragraph",
                    ordered_prefix_cols=[
                        "textFillArticleQfrP",
                        "textFillPastQP",
                        "textFillQ",
                    ],
                    is_paragraph_level=False,
                    reread=1,
                ),
            ]

        return configs
    # ---- P context ----
    else:
        return [
            # P
            SurpVariantConfig(
                target_col="paragraph",
                ordered_prefix_cols=[],
                is_paragraph_level=True,
                reread="both",
            ),
            # Q-P
            SurpVariantConfig(
                target_col="paragraph",
                ordered_prefix_cols=["question"],
                is_paragraph_level=True,
                reread=0,
            ),
            # P-P
            SurpVariantConfig(
                target_col="paragraph",
                ordered_prefix_cols=["paragraph"],
                is_paragraph_level=True,
                reread=1,
            ),
            # Q'-P-Q-P
            SurpVariantConfig(
                target_col="paragraph",
                ordered_prefix_cols=["Qfr", "paragraph", "question"],
                is_paragraph_level=True,
                reread=1,
            ),
            # ---- fillers ----
            # fillerP-P
            SurpVariantConfig(
                target_col="paragraph",
                ordered_prefix_cols=["fillerP"],
                is_paragraph_level=True,
                reread=1,
            ),
            # fillerQ-P
            SurpVariantConfig(
                target_col="paragraph",
                ordered_prefix_cols=["fillerQ"],
                is_paragraph_level=True,
                reread=0,
            ),
            # fillerQfr-fillerP-fillerQ-P
            SurpVariantConfig(
                target_col="paragraph",
                ordered_prefix_cols=["fillerQfr", "fillerP", "fillerQ"],
                is_paragraph_level=True,
                reread=1,
            ),
            # ---- text fillers ----
            # textFillP-P
            SurpVariantConfig(
                target_col="paragraph",
                ordered_prefix_cols=["textFillP"],
                is_paragraph_level=True,
                reread=1,
            ),
            # textFillQ-P
            SurpVariantConfig(
                target_col="paragraph",
                ordered_prefix_cols=["textFillQ"],
                is_paragraph_level=True,
                reread=0,
            ),
            # textFillQfr-textFillP-textFillQ-P
            SurpVariantConfig(
                target_col="paragraph",
                ordered_prefix_cols=["textFillQfr", "textFillP", "textFillQ"],
                is_paragraph_level=True,
                reread=1,
            ),
            # ---- instructions ----
            # instFirstReadP-P
            SurpVariantConfig(
                target_col="paragraph",
                ordered_prefix_cols=["instFirstReadP"],
                is_paragraph_level=True,
                reread="both",
            ),
            # instFirstReadP-Q-P
            SurpVariantConfig(
                target_col="paragraph",
                ordered_prefix_cols=["instFirstReadP", "question"],
                is_paragraph_level=True,
                reread=0,
            ),
            # instFirstReadP-P-instReReadP-P
            SurpVariantConfig(
                target_col="paragraph",
                ordered_prefix_cols=["instFirstReadP", "paragraph", "instReReadP"],
                is_paragraph_level=True,
                reread=1,
            ),
            # instFirstReadP-Q'-P-instReReadP-Q-P
            SurpVariantConfig(
                target_col="paragraph",
                ordered_prefix_cols=[
                    "instFirstReadP",
                    "Qfr",
                    "paragraph",
                    "instReReadP",
                    "question",
                ],
                is_paragraph_level=True,
                reread=1,
            ),
            # # ---- instruction for fillers ---
            # # instFirstReadP-fillerP-instReReadP-P
            # SurpVariantConfig(
            #     target_col = "paragraph",
            #     ordered_prefix_cols=["instFirstReadP", "fillerP", "instReReadP"],
            #     is_paragraph_level=True,
            #     reread=1,
            # ),
            # # instFirstReadP-fillerQ-P
            # SurpVariantConfig(
            #     target_col = "paragraph",
            #     ordered_prefix_cols=["instFirstReadP", "fillerQ"],
            #     is_paragraph_level=True,
            #     reread=0,
            # ),
            # # instFirstReadP-fillerQfr-fillerP-instReReadP-fillerQ-P
            # SurpVariantConfig(
            #     target_col = "paragraph",
            #     ordered_prefix_cols=["instFirstReadP", "fillerQfr", "fillerP", "instReReadP", "fillerQ"],
            #     is_paragraph_level=True,
            #     reread=1,
            # ),
        ]
