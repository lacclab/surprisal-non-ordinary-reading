# -----------------------------------------------------------------------------
# This is the main script for fitting Generalized Additive Models (GAMs) to the 
# eye-tracking data and obtaining the analysis results. The script imports necessary 
# dependencies and configuration files, iterates over multiple analysis types and 
# surprisal models, and performs density calculations, linear and non-linear model 
# comparisons, as well as GAM smooth fitting and plotting. Results are saved based 
# on the defined configurations and conditions.
# -----------------------------------------------------------------------------

shhh <- suppressPackageStartupMessages
shhh(library( mgcv ))
shhh(library(dplyr))
shhh(library(ggplot2))
shhh(library(lme4))
shhh(library(tidymv))
shhh(library(gamlss))
shhh(library(gsubfn))
shhh(library(lmerTest))
shhh(library(tidyverse))
shhh(library(boot))
shhh(library(rsample))
shhh(library(plotrix))
shhh(library(ggrepel))
shhh(library(mgcv))
shhh(library(tidyr))
shhh(library(CIPerm))
library(jmuOutlier)
library(here)
theme_set(theme_bw())
options(digits=4)
options(dplyr.summarise.inform = FALSE)

## Import files
source(here("src", "GAM", "preprocess_df.R"))
source(here("src", "GAM", "GAM_analysis_funcs.R"))
source(here("src", "GAM", "density", "GAM_density.R"))
source(here("src", "GAM", "GAM_dll_main.R"))
source(here("src", "GAM", "plot_funcs", "GAM_plot_funcs.R"))
source(here("src", "GAM", "GAM_run_config.R"))
source(here("src", "GAM", "GAM_smooths_main.R"))

# Configs
set.seed(12)

# ----------------- Configs -----------------
run_dll = TRUE
# run_dll = FALSE
run_smooths = TRUE
# run_smooths = FALSE

# Flags for running only specific parts
# only_plot = TRUE
only_plot = FALSE
# only_density = TRUE
only_density = FALSE
# only_dll_tests = TRUE
only_dll_tests = FALSE
# plot_and_density = TRUE
plot_and_density = FALSE

# TODO: choose the analysis type and model name
analysis_type_list = c("Consecutive_Repeated_reading", "Critical_Span", "Basic_Analysis",
                    "Different_Surprisal_Context", "Different_Surprisal_Context_zoom_in")
model_name = "EleutherAI-pythia-70m"

# ----------------- Run ---------------------
for (analysis_type in analysis_type_list) {
    config = get_configs(analysis_type, model_name)
    # Unpack the configuration into the current environment
    list2env(config, .GlobalEnv) # move config to global env
    print_configs(analysis_type, x_condition_name, y_condition_name, surp_list, RT_col_list)
    
    for (surp_column in surp_list){
        cat("------------", surp_column, "\n")
        if (only_plot == FALSE  | only_density == TRUE | plot_and_density == TRUE){
            original_eye_df <- get_eye_data(path, surp_column=surp_column)
        }
        
        for (RT_col in RT_col_list){
            if (only_plot == FALSE  | only_density == TRUE  | plot_and_density == TRUE){
                eye_df <- clean_eye_df(original_eye_df, RT_col, result_dir)
            }
            
            new_config = update_configs_by_surp_col(surp_column)
            list2env(new_config, .GlobalEnv) # move config to global env

            if (only_plot == FALSE  | only_density == TRUE  | plot_and_density == TRUE){
                # filter by config
                eye_df = filter_eye_df_by_config(eye_df)
                # save filter params
                zoom_in_threshold = get_zoom_in_threshold(eye_df)
                list2env(list(zoom_in_threshold=zoom_in_threshold), .GlobalEnv) # move config to global env
                save_filter_params(zoom_in, zoom_in_threshold, reread, reread_10_11, filter_hunting)
                # calc density
                density_path = here(results_path, paste0(RT_col, " - ", surp_column, "_density_data.csv"))
                eye_df = calc_density_data(eye_df, density_path, zoom_in_threshold)
                # filter zoom in
                eye_df = filter_eye_df_by_zoom_in(eye_df, zoom_in_threshold)
                # filter surp vals
                print_surp_quantiles(eye_df)
                eye_df = filter_eye_df_by_surp(eye_df)
                print_surp_quantiles(eye_df)
                # df after filters
                glimpse(eye_df)
            }

            ## Compare Linear and Non-Linear GAMs
            if (run_dll == TRUE){
                run_dll_analysis(eye_df)
            }

            ## Fit GAM and plot smooths
            if (run_smooths == TRUE){
                run_smooths_analysis(eye_df)
            }
        }
    }
}