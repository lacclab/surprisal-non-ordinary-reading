from typing import List, Union
import torch
from text_metrics.utils import clean_text
# if missing text metrics repo - install using https://github.com/lacclab/text-metrics.git

from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GPT2TokenizerFast,
    GPTNeoForCausalLM,
    GPTNeoXForCausalLM,
    GPTNeoXTokenizerFast,
)


def sentence_level_ppl(
    model: Union[AutoModelForCausalLM, GPTNeoXForCausalLM, GPTNeoForCausalLM],
    tok: Union[AutoTokenizer, GPTNeoXTokenizerFast, GPT2TokenizerFast],
    corpus: List[str],
    device: str,
) -> float:
    """This function calculates the sentence level perplexity of a given model on a given corpus.

    Args:
        model (Union[AutoModelForCausalLM, GPTNeoXForCausalLM]): The model to calculate perplexity for
        tok (Union[AutoTokenizer, GPTNeoXTokenizerFast]): The tokenizer for the model
        corpus (List[str]): A list of sentences to calculate perplexity for
        device (str): The device to run the model on

    Returns:
        float: The mean sentence level perplexity of the model on the given corpus
    """
    corpus = [clean_text(sentence) for sentence in corpus]
    nlls = []
    for sentence in tqdm(corpus):
        encodings = tok(
            sentence,
            return_tensors="pt",
        )

        with torch.no_grad():
            # the labels parameter is defined as
            outputs = model(
                encodings["input_ids"].to(device),
                attention_mask=encodings["attention_mask"].to(device),
                return_dict=True,
                labels=encodings["input_ids"].to(device),
                # dont return the hidden states
                output_hidden_states=False,
            )

            # loss is calculated using CrossEntropyLoss which averages over valid labels
            # N.B. the model only calculates loss over trg_len - 1 labels, because it internally shifts the labels
            # to the left by 1.
            neg_log_likelihood = outputs.loss

        nlls.append(neg_log_likelihood)

    paragraph_level_ppl = torch.exp(torch.stack(nlls).mean())

    return paragraph_level_ppl.item()
