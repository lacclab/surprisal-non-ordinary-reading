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
    parser.add_argument(
        "--session_context",
        action="store_true",
        help="Enable session context processing.",
    )
    parser.add_argument(
        "--session_context_fillers",
        action="store_true",
        help="Include session context fillers in processing.",
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
   - `--regular_articles`: ?
   - `--fillers`: Include fillers in processing ('.' instead of each token?).
   - `--session_context`: Enable session context processing.
   - `--session_context_fillers`: Include session context fillers in processing.
   - `--models`: List of model names to process.
   - `--config_indices`: List of config indices where configs are listed in `src/Surprisal_Estimates/context_params.py`.
   - `--surp_extractor`: Surp extractor type. One of `text_metrics.surprisal_extractors.extractor_switch.SurpExtractorType._member_names_`.
   - `--debug_mode`: Enable debug mode for processing.
2. Merge the output CSVs into one CSV which also contains eye-tracking measures using `src/merge_surp_yoav_250924.ipynb`.

### Main Analysis Steps

1. Procduce surprisals using `src/Surprisal_Estimates/context_run.py`
2. Merge them into a single "all surprisals for configuration" CSV using `src/merge_surp_yoav_250924.ipynb`
3. in `GAM_run_config.R` set `basic_file_path` to the path of the merged CSV
4. Create a new folder in `src/GAM/results 0<RT<3000 firstpassNA` with the name of the configuration
5. Run `src/GAM/GAM_run_analysis.R`
6. Run `src/GAM/GAM_context_plots.R`
