from text_metrics.utils import clean_text, get_surprisal, init_tok_n_model
# if missing text metrics repo - install using https://github.com/lacclab/text-metrics.git

import pandas as pd

def get_surp_on_text(text: str, model_name: str)-> pd.DataFrame:

    text_reformatted = clean_text(text)

    tokenizer, model = init_tok_n_model(
        model_name=model_name,
        device="cuda:1",
        hf_access_token="missing_token", # TODO: replace with your own token
    )

    surprisal = get_surprisal(
        text=text_reformatted,
        tokenizer=tokenizer,
        model=model,
        model_name=model_name,
        context_stride=512,
    )

    return(surprisal)

if __name__ == "__main__":
    # Text to analyze
    P_1_6_Adv_4 = ("On top of the scientiﬁc impact, artists might have to rethink their drawings of the people. "
            "“You see a lot of reconstructions of these people hunting and gathering and they look like modern Europeans with light skin. "
            "You never see a reconstruction of a Mesolithic hunter-gatherer with dark skin and blue eye color,” Lalueza-Fox said. "
            "The Spanish team went on to compare the genome of the hunter-gatherer to those of modern Europeans from different regions to see how they might be related. "
            "They found that the ancient DNA most closely matched the genetic makeup of people living in northern Europe, in particular Sweden and Finland.")

    P_1_10_Adv_1 = "Agios Efstratios is so remote, so forgotten by the banks, the government and most of the modern world that the mobile phone network can't process data and there isn't a single ATM or credit-card machine on the island. Before Greece was plunged into financial chaos, residents of this tranquil outpost in the northern Aegean managed quite well. They did their banking at the post office and the few dozen rooms to rent were booked out every summer with people who had heard – by word of mouth – of its spectacular empty beaches, clear seas and fresh seafood."

    surprisal = get_surp_on_text(P_1_6_Adv_4, model_name = 'gpt2')
    surprisal.to_csv("debug_surp_vals.csv")
