# -----------------------------------------------------------------------------
# run_dll_analysis
# This function performs delta log-likelihood (DLL) analysis on the eye-tracking dataset
# using different model configurations. It supports fitting models with cross-validation 
# (CV) and calculating DLL statistics. The results are saved as CSV files and can be used 
# for model comparison through permutation tests.
# -----------------------------------------------------------------------------

run_dll_analysis <- function(
    eye_df
) {
    for (use_CV in CV_list){
        prefix_comp_results = paste0(RT_col, " - ", surp_column, "_re=", RE, "_useCV=", use_CV, "_")

        if (only_plot == FALSE & only_density == FALSE) {
            if (only_dll_tests == FALSE) {
                #Fit
                dll_raw_df = fit_models_and_get_dll_raw_df(eye_df, models_names, RT_col, use_CV, num_folds)
                write.csv(dll_raw_df, here(results_path, paste0(prefix_comp_results, "dll_raw_df_stats.csv")))
                dll_comp_stats_df = get_dll_comp_stats_df(dll_raw_df, models_names)
                write.csv(dll_comp_stats_df, here(results_path, paste0(prefix_comp_results, "dll_comp_stats_df.csv")))
                }
            else {
                dll_raw_df = read.csv(here(results_path, paste0(prefix_comp_results, "dll_raw_df_stats.csv"))) # load_csv
            }
            comp_dfs = get_linear_comp_df_using_permutation_test(dll_raw_df, models_names)
            linear_comp_df <- comp_dfs$linear_comp_df
            p_vals_df <- comp_dfs$p_vals_df
            write.csv(linear_comp_df, here(results_path, paste0(prefix_comp_results, "linear_comp_df.csv")))
            write.csv(p_vals_df, here(results_path, paste0(prefix_comp_results, "dll_p_vals_df.csv")))
        }

        if (only_plot == TRUE){
            # Load
            linear_comp_df = read.csv(here(results_path, paste0(prefix_comp_results, "linear_comp_df.csv"))) # load csv
        }
        
        if (zoom_in == FALSE  & only_density == FALSE & analysis_type != "Different_Surprisal_Context"){
            # plot 
            plot_dll_by_model(
                linear_comp_df=linear_comp_df, 
                path=here(results_path, paste0(prefix_comp_results, "linear_comp_df.pdf")),
                x_condition_name=x_condition_name,
                y_condition_name=y_condition_name,
                x_condition_labels=x_condition_labels,
                y_condition_labels=y_condition_labels,
                RT_col=RT_col,
                use_CV=use_CV
            )
        }
    }
}