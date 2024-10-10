# surprisal-non-ordinary-reading

[![Ruff](https://github.com/lacclab/surprisal-non-ordinary-reading/actions/workflows/ruff.yml/badge.svg?branch=main)](https://github.com/lacclab/surprisal-non-ordinary-reading/actions/workflows/ruff.yml)

## Quick Start

1. `conda env create -f surp_py_env.yml`
    - `conda activate surp_py_env` for running python scripts 
2. `conda env create -f surp_r_env.yml`
    - `conda activate surp_r_env` for running R scripts 

### Main Analysis Steps

1. Procduce surprisals using `src/Surprisal_Estimates/context_run.py`
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

1. Run `src/GAM/perplexity/perplexity_plot.R` - Perplexity Plot (Reproduce Figure A1 in Appendix)
2. Additional analysis types:
    - `"Different_Surprisal_Estimates"` (Fig A2)
    - `"Basic_Analysis"` (Fig A3)
    - `"Consecutive_Repeated_reading"` and `"Critical_Span"` (Fig A4)
    - `"Different_Surprisal_Context"` and `"Different_Surprisal_Context_zoom_in"` (Figures A5-A8)
