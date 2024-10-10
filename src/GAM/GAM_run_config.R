# ----------------------------------------------------------------------
# This R script defines functions for configuring and executing different analysis types 
# on eye-tracking data with various surprisal models. The script includes configurations 
# for baseline, linear, nonlinear, and context-specific models, with options to modify 
# conditions like rereading behavior and critical span inclusion. Results are saved 
# based on the chosen configuration and model type, and the script can handle different 
# input file paths and context-specific data columns.
# ----------------------------------------------------------------------

# ------------------------------ Data paths ----------------------------
# TODO: Put your data files here
basic_file_path = "/src/data/et_20240505_with_all_surp20240624.csv"
large_models_cotext_path = "/src/data/et_20240505_large_models_context_cols_20240625.csv"
large_models_part_2_path = "/src/data/et_20240505_large_models_part_2_context_cols_20240625.csv"
large_models_part_3_path = "/src/data/et_20240505_large_models_part_3_context_cols_20240625.csv"
pythia70m_article_path = "/src/data/et_20240505_EleutherAI-pythia-70m_article.csv"
gemma2_9b_article_path = "/src/data/et_20240505_gemma-2-9b_article.csv"

# ------------------------------ Results directory ---------------------
# This defined the preprocessing type (which filter of RT to use) and the results directory
result_dir = "results 0<RT<3000 firstpassNA"

# ------------------------------ Debugging ------------------------------
# Set to TRUE to use a smaller data file for debugging
debug_on_mini = FALSE
debug_path = here("src", "GAM", "tests", "generated_data.csv")

# ------------------------------ Functions ------------------------------

get_file_name <- function(path) {
    print(sprintf("!! data path: %s", path))
    file_name <- basename(path)
    file_name = substr(file_name, 1, nchar(file_name) - 4)
    print(sprintf("!! file_name: %s", file_name))
    return(file_name)
}

get_models_names <- function(){
    models_names = c(
        "baseline", 
        "linear",
        "nonlinear")
    return(models_names)
}

get_configs <- function(analysis_type, model_name) {
    # default configs
    RE=F
    additive_list = c(T) # T F
    CV_list = c(T) # T, F
    num_folds = 10
    mode = "surp"

    reread = "both"
    reread_10_11 = FALSE
    filter_hunting = FALSE
    zoom_in = FALSE
    surp_df = "missing surp_df"

    models_names = get_models_names()
    cat("Models | ", models_names, "\n")

    if (analysis_type == "Basic_Analysis") {
        path = basic_file_path
        file_name = get_file_name(path)
        results_path = here("src", "GAM", result_dir, "results", file_name)
        
        surp_list = paste0(model_name, "-Surprisal")
        
        RT_col_list = c("FirstPassGD", "GD", "FirstPassFF", "FF", "TF")
        
        x_condition_name="has_preview_condition"
        x_condition_labels=c("Ordinary Reading", "Information Seeking")
        x_condition_vals=c("Gathering", "Hunting")
        
        y_condition_name="reread_condition"
        y_condition_labels=c("First Reading", "Repeated Reading")
        y_condition_vals=c(0, 1)

    } else if (analysis_type == "Consecutive_Repeated_reading"){
        path = basic_file_path
        file_name = get_file_name(path)
        results_path = here("src", "GAM", result_dir, "results_reread_split", file_name)
        
        surp_list = paste0(model_name, "-Surprisal")

        RT_col_list = c("FirstPassGD", "GD", "FirstPassFF", "FF", "TF")

        x_condition_name="has_preview_condition"
        x_condition_labels=c("Ordinary Reading", "Information Seeking")
        x_condition_vals=c("Gathering", "Hunting")
        
        y_condition_name="reread_consecutive_condition"
        y_condition_labels=c("Consecutive\nRepeated Reading", "Non-Consecutive\nRepeated Reading")
        y_condition_vals=c(11, 12)

        reread = 1

    } else if (analysis_type == "Critical_Span"){
        path = basic_file_path
        file_name = get_file_name(path)
        results_path = here("src", "GAM", result_dir, "results_critical_span", file_name)
        
        surp_list = paste0(model_name, "-Surprisal")

        RT_col_list = c("FirstPassGD", "GD", "FirstPassFF", "FF", "TF")

        x_condition_name = "critical_span_condition"
        x_condition_labels=c("Outside CS", "Inside CS")
        x_condition_vals=c(0, 1)
        
        y_condition_name="reread_condition"
        y_condition_labels=c("First Reading", "Repeated Reading")
        y_condition_vals=c(0, 1)

        filter_hunting = TRUE

    } else if (analysis_type == "Different_Surprisal_Estimates"){
        path = basic_file_path
        file_name = get_file_name(path)
        results_path = here("src", "GAM", result_dir, "results_different_surp_estimates", file_name)
        surp_list = read.csv(here("src", "GAM", "cols_lists", "P_context_surp_cols_all_models.csv"))$surp_col
        # surp_list = surp_list[7:length(surp_list)]
        # surp_list = c(
        #     "google-recurrentgemma-9b-Surprisal-Context-P"
        # )

        RT_col_list = c("FirstPassGD")

        x_condition_name="has_preview_condition"
        x_condition_labels=c("Ordinary Reading", "Information Seeking")
        x_condition_vals=c("Gathering", "Hunting")
        
        y_condition_name="reread_condition"
        y_condition_labels=c("First Reading", "Repeated Reading")
        y_condition_vals=c(0, 1)

    } else if (analysis_type == "Different_Surprisal_Context"){
        if (model_name == "large_models"){
            path = large_models_cotext_path
        } else if (model_name == "large_models_part_2") {
            path = large_models_part_2_path
        } else if (model_name == "large_models_part_3") {
            path = large_models_part_3_path
        } else if (model_name == "pythia70m_article") {
            path = pythia70m_article_path
            reread_10_11 = TRUE
        } else if (model_name == "gemma2_9b_article") {
            path = gemma2_9b_article_path
            reread_10_11 = TRUE
        } else {
            path = basic_file_path
        }
        
        # file name
        file_name = get_file_name(path)
        # results path
        results_path = here("src", "GAM", result_dir, "results_context", file_name)
        
        if (model_name == "large_models"){
            surp_df = read.csv(here("src", "GAM", "cols_lists", "surp_large_models-context_cols.csv"))
        } else if (model_name == "large_models_part_2") {
            surp_df = read.csv(here("src", "GAM", "cols_lists", "surp_large_models_part_2-context_cols.csv"))
        } else if (model_name == "large_models_part_3") {
            surp_df = read.csv(here("src", "GAM", "cols_lists", "surp_large_models_part_3-context_cols.csv"))
        } else if (model_name == "pythia70m_article") {
            surp_df = read.csv(here("src", "GAM", "cols_lists", "surp_pythia70m_article_cols.csv"))
        } else if (model_name == "gemma2_9b_article") {
            surp_df = read.csv(here("src", "GAM", "cols_lists", "surp_gemma-2-9b_article_cols.csv"))
        } else {
            surp_df = read.csv(here("src", "GAM", "cols_lists", paste0("surp_", model_name, "-context_cols.csv")))
        }
        surp_list = surp_df$surp_col
        # surp_list = surp_list[91:length(surp_list)]
        
        RT_col_list = c("FirstPassGD")

        x_condition_name="has_preview_condition"
        x_condition_labels=c("Ordinary Reading", "Information Seeking")
        x_condition_vals=c("Gathering", "Hunting")
        
        y_condition_name="reread_condition"
        y_condition_labels=c("First Reading", "Repeated Reading")
        y_condition_vals=c(0, 1)

    } else if (analysis_type == "Different_Surprisal_Context_zoom_in"){
        if (model_name == "large_models"){
            path = large_models_cotext_path
        } else if (model_name == "large_models_part_2") {
            path = large_models_part_2_path
        } else if (model_name == "large_models_part_3") {
            path = large_models_part_3_path
        } else if (model_name == "pythia70m_article") {
            path = pythia70m_article_path
            reread_10_11 = TRUE
        } else if (model_name == "gemma2_9b_article") {
            path = gemma2_9b_article_path
            reread_10_11 = TRUE
        } else {
            path = basic_file_path
        }
        
        # file name
        file_name = get_file_name(path)
        # results path
        results_path = here("src", "GAM", result_dir, "results_context", file_name, "zoom_in")
        
        if (model_name == "large_models"){
            surp_list = read.csv(here("src", "GAM", "cols_lists", "surp_large_models-context_zoom_in_cols.csv"))$surp_col
        } else if (model_name == "large_models_part_2") {
            surp_list = read.csv(here("src", "GAM", "cols_lists", "surp_large_models_part_2-context_zoom_in_cols.csv"))$surp_col
        } else if (model_name == "large_models_part_3") {
            surp_list = read.csv(here("src", "GAM", "cols_lists", "surp_large_models_part_3-context_zoom_in_cols.csv"))$surp_col
        } else if (model_name == "pythia70m_article") {
            surp_list = read.csv(here("src", "GAM", "cols_lists", "surp_pythia70m_article_zoom_in_cols.csv"))$surp_col
        } else if (model_name == "gemma2_9b_article") {
            surp_list = read.csv(here("src", "GAM", "cols_lists", "surp_gemma-2-9b_article_zoom_in_cols.csv"))$surp_col
        } else {
        surp_list = c(
            paste0(model_name, "-Surprisal-Context-P-P"), 
            paste0(model_name, "-Surprisal-Context-Qfr-P-Q-P"),
            paste0(model_name, "-Surprisal-Context-instFirstReadP-P-instReReadP-P"),
            paste0(model_name, "-Surprisal-Context-instFirstReadP-Qfr-P-instReReadP-Q-P")
            )
        }
        
        RT_col_list = c("FirstPassGD")

        x_condition_name="has_preview_condition"
        x_condition_labels=c("Ordinary Reading", "Information Seeking")
        x_condition_vals=c("Gathering", "Hunting")
        
        y_condition_name="reread_condition"
        y_condition_labels=c("Repeated Reading")
        y_condition_vals=c(1)
        
        reread = 1
        zoom_in = TRUE

    } else if (analysis_type == "Context_ReRead"){
        if (model_name == "pythia70m_article") {
            path = pythia70m_article_path
        } else {
            path = "missing path"
        }

        file_name = get_file_name(path)
        results_path = here("src", "GAM", result_dir, "results_context_reread_split", file_name)
        
        if (model_name == "pythia70m_article") {
            surp_list = read.csv(here("src", "GAM", "cols_lists", "surp_pythia70m_article_cols.csv"))$surp_col
        } else {
            surp_list = c("missing_surp_list")
        }

        RT_col_list = c("FirstPassGD", "GD", "FirstPassFF", "FF", "TF")

        x_condition_name="has_preview_condition"
        x_condition_labels=c("Ordinary Reading", "Information Seeking")
        x_condition_vals=c("Gathering", "Hunting")
        
        y_condition_name="reread_consecutive_condition"
        y_condition_labels=c("Consecutive\nRepeated Reading", "Non-Consecutive\nRepeated Reading")
        y_condition_vals=c(11, 12)

        reread = 1

    } else if (analysis_type == "Context_ReRead_zoom_in"){
        if (model_name == "pythia70m_article") {
            path = pythia70m_article_path
        } else {
            path = "missing path"
        }

        file_name = get_file_name(path)
        results_path = here("src", "GAM", result_dir, "results_context_reread_split", file_name, "zoom_in")
        
        if (model_name == "pythia70m_article") {
            surp_list = read.csv(here("src", "GAM", "cols_lists", "surp_pythia70m_article_zoom_in_col.csv"))$surp_col
        } else {
            surp_list = c("missing_surp_list")
        }

        RT_col_list = c("FirstPassGD", "GD", "FirstPassFF", "FF", "TF")

        x_condition_name="has_preview_condition"
        x_condition_labels=c("Ordinary Reading", "Information Seeking")
        x_condition_vals=c("Gathering", "Hunting")
        
        y_condition_name="reread_consecutive_condition"
        y_condition_labels=c("Consecutive\nRepeated Reading", "Non-Consecutive\nRepeated Reading")
        y_condition_vals=c(11, 12)

        reread = 1
        zoom_in = TRUE

    } else {
        "Undefined analysis_type"
    }

    if (debug_on_mini == TRUE){
        path=debug_path
        file_name = get_file_name(path)
    }

    return(list(
        models_names = models_names,
        RE = RE,
        additive_list = additive_list,
        CV_list = CV_list,
        num_folds = num_folds,
        mode = mode,
        reread = reread,
        reread_10_11 = reread_10_11,
        filter_hunting = filter_hunting,
        zoom_in = zoom_in,
        path = path,
        file_name = file_name,
        results_path = results_path,
        surp_list = surp_list,
        surp_df = surp_df,
        RT_col_list = RT_col_list,
        x_condition_name = x_condition_name,
        x_condition_labels = x_condition_labels,
        x_condition_vals = x_condition_vals,
        y_condition_name = y_condition_name,
        y_condition_labels = y_condition_labels,
        y_condition_vals = y_condition_vals
    ))
}

print_configs <- function(
    analysis_type, x_condition_name, y_condition_name, surp_list, RT_col_list) {
    cat(analysis_type, " ", x_condition_name, " ", y_condition_name, "\n")
    cat(length(surp_list), " surp cols\n")
    cat(surp_list, "\n")
    cat(RT_col_list, "\n")
}