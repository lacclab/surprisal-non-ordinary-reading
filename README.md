# surprisal-non-ordinary-reading

[![Ruff](https://github.com/lacclab/surprisal-non-ordinary-reading/actions/workflows/ruff.yml/badge.svg?branch=main)](https://github.com/lacclab/surprisal-non-ordinary-reading/actions/workflows/ruff.yml)

## Quick Start

1. `conda env create -f surp_py_env.yml`
    - `conda activate surp_py_env` for running python scripts 
2. `conda env create -f surp_r_env.yml`
    - `conda activate surp_r_env` for running R scripts 

### Surprisal Estimation

    parser = argparse.ArgumentParser(description="Process context configuration flags.")
    parser.add_argument(
        "--article_context",
        action="store_true",
        help="Enable article context processing.",
    )
    parser.add_argument(
        "--regular_articles", action="store_true", help="Process regular articles."
    )
    parser.add_argument(
        "--fillers", action="store_true", help="Include fillers in processing."
    )
    parser.add_argument("--models", nargs="+", help="List of model names to process.")
    parser.add_argument(
        "--config_indices", nargs="+", help="List of config indices.", type=int
    )
    # add surp extractor that can be one of
    surp_extractor_types = SurpExtractorType._member_names_
    parser.add_argument(
        "--surp_extractor",
        type=str,
        choices=surp_extractor_types,
        help="Surp extractor type.",
    )
    # add debug mode
    parser.add_argument(
        "--debug_mode", action="store_true", help="Enable debug mode for processing."
    )

1. Run `src/Surprisal_Estimates/context_run.py`. Parameters:
   - `--article_context`: Enable article context processing.
   - `--fillers`: Include fillers in processing.
   - `--models`: List of model names to process.
   - `--config_indices`: List of config indices where configs are listed in `src/Surprisal_Estimates/context_params.py`.
   - `--surp_extractor`: Surp extractor type. One of `text_metrics.surprisal_extractors.extractor_switch.SurpExtractorType._member_names_`.
   - `--debug_mode`: Enable debug mode for processing.
2. Merge the output CSVs into one CSV which also contains eye-tracking measures using `src/merge_surp_yoav_250924.ipynb`.

### Main Analysis Steps

1. Procduce surprisals using `src/Surprisal_Estimates/context_run.py`
2. Merge them into a single "all surprisals" CSV
3. Put the path of the merged CSV in `GAM_run_config.R` (set `basic_file_path`)
4. Choose the analysis type and model name
    - Possible analysis types:
        - "Basic_Analysis" (Fig 1)
        - "Different_Surprisal_Context" and "Different_Surprisal_Context_zoom_in" (Fig 2)
    - Example model name: "EleutherAI-pythia-70m"
        - Full model names can be found in: /src/GAM/perplexity/models_names_with_ppl.csv
5. Run `src/GAM/GAM_run_analysis.R` - Fit models and calculate DLLs
6. Run `src/GAM/GAM_context_plots.R` - Context Plots (Fig 2)
7. Run `src/GAM/plot_funcs/GAM_plot_grid_run.ipynb` - Add more notations to the plots
8. Run `src/GAM/context_plots/stats_tests/GAM_context_dll_tests.R` - Conduct permutation tests for comparing DLLs (Reproduce Section 6.2 p vals)

### Reproduce the Results in Appendix
1. Run `src/GAM/perplexity/perplexity_plot.R` - Perplexity Plot (Reproduce Figure A1 in Appendix)
2. Additional analysis types:
    - "Different_Surprisal_Estimates" (Fig A2)
    - "Basic_Analysis" (Fig A3)
    - "Consecutive_Repeated_reading" and "Critical_Span" (Fig A4)
    - "Different_Surprisal_Context" and "Different_Surprisal_Context_zoom_in" (Figures A5-A8)
