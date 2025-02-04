# Code for the paper: The Effect of Surprisal on Reading Times in Information Seeking and Repeated Reading

[![Ruff](https://github.com/lacclab/surprisal-non-ordinary-reading/actions/workflows/ruff.yml/badge.svg?branch=main)](https://github.com/lacclab/surprisal-non-ordinary-reading/actions/workflows/ruff.yml)

## Quick Start

1. `conda env create -f surp_py_env.yml`
   - `conda activate surp_py_env` for running python scripts
2. `conda env create -f surp_r_env.yml`
   - `conda activate surp_r_env` for running R scripts

### Main Analysis Steps

1. Produce surprisals using `src/Surprisal_Estimates/context_run.py`
2. Merge them into a single "all surprisals" CSV
3. Put the path of the merged CSV in `GAM_run_config.R` (set `basic_file_path`)
4. For the following steps (5-8) - Choose the analysis type and model name
   - Analysis types:
     - `"Basic_Analysis"` (Fig 1)
     - `"Different_Surprisal_Context"` and `"Different_Surprisal_Context_zoom_in"` (Fig 2)
   - Example model name: `"EleutherAI-pythia-70m"`
     - Full model names can be found in: `/src/GAM/perplexity/models_names_with_ppl.csv`
5. Run `src/GAM/GAM_run_analysis.R` - Fit models and calculate DLLs
6. Run `src/GAM/GAM_context_plots.R` - Context Plots (Fig 2)
7. Run `src/GAM/plot_funcs/GAM_plot_grid_run.ipynb` - Add more notations to the plots
8. Run `src/GAM/context_plots/stats_tests/GAM_context_dll_tests.R` - Conduct permutation tests for comparing DLLs (Reproduce Section 6.2 p vals)

### Reproduce the Results in Appendix

1. Run `src/GAM/perplexity/perplexity_plot.R` (Fig A1)
2. Additional analysis types:
   - `"Different_Surprisal_Estimates"` (Fig A2)
   - `"Basic_Analysis"` (Fig A3)
   - `"Consecutive_Repeated_reading"` and `"Critical_Span"` (Fig A4)
   - `"Different_Surprisal_Context"` and `"Different_Surprisal_Context_zoom_in"` (Figures A5-A8)

## Credits

- We have adopted some scripts and code snippets from the [xlang-processing repository](https://github.com/wilcoxeg/xlang-processing). We acknowledge and thank the authors for their contributions.

## Citation

If you use this repository, please consider citing the following work:

```bibtex
@inproceedings{gruteke-klein-etal-2024-effect,
    title = "The Effect of Surprisal on Reading Times in Information Seeking and Repeated Reading",
    author = "Gruteke Klein, Keren  and
      Meiri, Yoav  and
      Shubi, Omer  and
      Berzak, Yevgeni",
    editor = "Barak, Libby  and
      Alikhani, Malihe",
    booktitle = "Proceedings of the 28th Conference on Computational Natural Language Learning",
    month = nov,
    year = "2024",
    address = "Miami, FL, USA",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.conll-1.17/",
    doi = "10.18653/v1/2024.conll-1.17",
    pages = "219--230",
    abstract = "The effect of surprisal on processing difficulty has been a central topic of investigation in psycholinguistics. Here, we use eyetracking data to examine three language processing regimes that are common in daily life but have not been addressed with respect to this question: information seeking, repeated processing, and the combination of the two. Using standard regime-agnostic surprisal estimates we find that the prediction of surprisal theory regarding the presence of a linear effect of surprisal on processing times, extends to these regimes. However, when using surprisal estimates from regime-specific contexts that match the contexts and tasks given to humans, we find that in information seeking, such estimates do not improve the predictive power of processing times compared to standard surprisals. Further, regime-specific contexts yield near zero surprisal estimates with no predictive power for processing times in repeated reading. These findings point to misalignments of task and memory representations between humans and current language models, and question the extent to which such models can be used for estimating cognitively relevant quantities. We further discuss theoretical challenges posed by these results."
}
```
