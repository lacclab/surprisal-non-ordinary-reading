# Conduct permutation tests for comparing DLLs (Reproduce Section 6.2 p vals)

# -----------------------------------------------------------------------------
# This script conducts context-based DLL (Delta Log-Likelihood) comparisons for 
# various models using permutation tests. It loads DLL data, filters it based 
# on conditions, and performs paired comparisons between different contexts to 
# evaluate their statistical significance. Outputs include p-values and summary 
# statistics saved as CSV files. 
# -----------------------------------------------------------------------------

shhh <- suppressPackageStartupMessages # It's a library, so shhh!
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
library(tidyr)
library(jmuOutlier)
library(purrr)
library(CIPerm)
theme_set(theme_bw())
library(here)
source(here("src", "GAM", "GAM_analysis_funcs.R"))
source(here("src", "GAM", "plot_funcs", "GAM_plot_funcs.R"))

options(digits=15)
options(dplyr.summarise.inform = FALSE)


get_path_by_surp <- function(surp_column, zoom_in, results_path, results_path_zoom_in) {
    if (zoom_in==1){
        print(surp_column)
        print(results_path_zoom_in)
        path = here(results_path_zoom_in, paste0(RT_col, " - ", surp_column, "_re=", RE, "_useCV=", use_CV, "_", "dll_raw_df_stats.csv"))
    } else{
        path = here(results_path, paste0(RT_col, " - ", surp_column, "_re=", RE, "_useCV=", use_CV, "_", "dll_raw_df_stats.csv"))
    }
    return(path)
    }

print_n_rows <- function(df1, df2, x_condition_val, y_condition_val) {
    cat("Test | ", x_condition_name, " =", x_condition_val, 
    " | ",y_condition_name, " =", y_condition_val, 
    "| ", length(df1), "rows ---------- \n")
    cat("Test | ", x_condition_name, " =", x_condition_val, 
    " | ",y_condition_name, " =", y_condition_val, 
    "| ", length(df2), "rows ---------- \n")
}

# permutation test between two cols
get_p_df <- function(
    context_name_1, context_name_2, 
    dll_1, dll_2, 
    zoom_in_1, zoom_in_2,
    context_type_1, context_type_2,
    linear, 
    x_condition_val, y_condition_val,
    surp_prefix,
    results_path, results_path_zoom_in
    ) {
    # load data
    cat("loading data...", context_type_1, context_name_1, context_type_2, context_name_2, "\n")
    raw_dll_1 = read.csv(get_path_by_surp(paste0(surp_prefix, context_name_1), zoom_in_1, results_path, results_path_zoom_in))
    raw_dll_2 = read.csv(get_path_by_surp(paste0(surp_prefix, context_name_2), zoom_in_2, results_path, results_path_zoom_in))
    # filter by x_condition_val, y_condition_val, linear and create dll column
    dll_1 = get_dll_df_for_permu_test(raw_dll_1, x_condition_val, y_condition_val, linear=linear)
    dll_2 = get_dll_df_for_permu_test(raw_dll_2, x_condition_val, y_condition_val, linear=linear)
    print_n_rows(dll_1, dll_2, x_condition_val, y_condition_val)
    # conduct test
    n_sim = 1000
    dset_result = dset(dll_1, dll_2, nmc=n_sim)
    CIPerm_p_val = pval(dset_result, tail = c("Two"))
    if (is.na(CIPerm_p_val) || CIPerm_p_val == 0){
        n_sim = 20000
        dset_result = dset(dll_1, dll_2, nmc=n_sim)
        CIPerm_p_val = pval(dset_result, tail = c("Two"))
        print(CIPerm_p_val)
    }
    # save results to df
    p_df = setNames(
        data.frame(
            x_condition_name = x_condition_val, 
            y_condition_name = y_condition_val,
            "context_name_1" = context_name_1,
            "context_type_1" = context_type_1,
            "context_name_2" = context_name_2,
            "context_type_2" = context_type_2,
            linear = linear,
            CIPerm_p_val = CIPerm_p_val,
            n_sim = n_sim
        ),
        c(x_condition_name, y_condition_name, 
        "context_name_1", "context_type_1", 
        "context_name_2", "context_type_2", 
        "linear", "CIPerm_p_val", "n_sim")
    )
    print(p_df)
    return(p_df)
}

# permutation tests
run_dll_tests <- function(
    compare_plan, 
    surp_prefix,
    file_name,
    results_path,
    results_path_zoom_in
    ) {
    all_p_df = data.frame()
    x_condition_vals = compare_plan$x_condition_vals
    y_condition_vals = compare_plan$y_condition_vals
    context_name_1_vals = compare_plan$context_name_1_vals
    context_name_2_vals = compare_plan$context_name_2_vals
    context_type_1_vals = compare_plan$context_type_1
    context_type_2_vals = compare_plan$context_type_2
    zoom_in_1_vals = compare_plan$zoom_in_1_vals
    zoom_in_2_vals = compare_plan$zoom_in_2_vals
    for (i in seq_along(x_condition_vals)) {
        x_condition_val <- x_condition_vals[i]
        y_condition_val <- y_condition_vals[i]
        context_name_1 <- context_name_1_vals[i]
        context_name_2 <- context_name_2_vals[i]
        zoom_in_1 <- zoom_in_1_vals[i]
        zoom_in_2 <- zoom_in_2_vals[i]
        context_type_1 <- context_type_1_vals[i]
        context_type_2 <- context_type_2_vals[i]
        for (linear in c("linear", "nonlinear")) {
            # get p values
            p_df = get_p_df(
                context_name_1=context_name_1, 
                context_name_2=context_name_2, 
                zoom_in_1=zoom_in_1,
                zoom_in_2=zoom_in_2,
                context_type_1=context_type_1,
                context_type_2=context_type_2,
                linear=linear,  
                x_condition_val=x_condition_val, 
                y_condition_val=y_condition_val,
                surp_prefix=surp_prefix,
                results_path=results_path,
                results_path_zoom_in=results_path_zoom_in
            )
            all_p_df = rbind(all_p_df, p_df)
        }
    }
    all_p_df$p_val_symbol <- sapply(all_p_df$CIPerm_p_val, p2stars)

    # save results to csv
    write.csv(all_results_df, here(results_path, paste0(
        "context_dll_tests", "_", surp_prefix, RT_col, ".csv"))
    )
}

run_context_dll_compare <- function(model_name, file_name) {
    cat("run_context_dll_compare \n")
    surp_prefix = paste0(model_name, "-Surprisal-Context-")
    results_path <- paste0("/src/GAM/results 0<RT<3000 firstpassNA/results_context/", file_name)
    results_path_zoom_in <- paste0("/src/GAM/results 0<RT<3000 firstpassNA/results_context/", file_name, "/zoom_in")
    run_dll_tests(
        compare_plan=compare_plan,
        surp_prefix=surp_prefix,
        file_name=file_name,
        results_path=results_path,
        results_path_zoom_in=results_path_zoom_in
    )
}

# params
RE = F
use_CV = T
additive_model = T
RT_col = "FirstPassGD"
x_condition_name="has_preview_condition"
y_condition_name="reread_condition"

# Example
file_name = "et_20240505_with_all_surp20240624" # Define the file_name for results
compare_plan = read.csv("/src/GAM/context_plots/stats_tests/GAM_dll_compare_plan.csv") # Load the comparison plan from a CSV file
models_list = c("EleutherAI-pythia-70m") # Specify the list of models to test

for (model_name in models_list){
    cat("Context DLL Test| ", model_name, " --------- \n")
    run_context_dll_compare(model_name, file_name)
}

