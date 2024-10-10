
run_smooths_analysis <- function(
    eye_df
) {
    for (additive_model in additive_list){
        prefix_smooths_results = paste0(RT_col, " - ", surp_column, "_re=", RE, "_additive=", additive_model, "_")
        
        if (only_plot == FALSE & only_density == FALSE & only_dll_tests == FALSE & plot_and_density == FALSE) {
            # Fit
            smooths = get_gam_smooths_including_bootstrap(eye_df, additive_model)
            linear_smooths <- smooths$linear_smooths
            nonlinear_smooths <- smooths$nonlinear_smooths
            smooths_p_vals <- smooths$all_p_vals
            # save to csv
            write.csv(linear_smooths, here(results_path, paste0(prefix_smooths_results, "linear_smooths.csv")))
            write.csv(nonlinear_smooths, here(results_path, paste0(prefix_smooths_results, "nonlinear_smooths.csv")))
            write.csv(smooths_p_vals, here(results_path, paste0(prefix_smooths_results, "smooths_p_vals.csv")))
        }

        if ((only_plot == TRUE & only_density == FALSE & only_dll_tests == FALSE) | plot_and_density == TRUE){
            # Load from csv
            linear_smooths = read.csv(here(results_path, paste0(prefix_smooths_results, "linear_smooths.csv"))) 
            nonlinear_smooths = read.csv(here(results_path, paste0(prefix_smooths_results, "nonlinear_smooths.csv"))) 
        }

        if (only_density == FALSE & only_dll_tests == FALSE){
            density_data = read.csv(density_path) 
            
            # Plot Surprisal / RT relationship
            plot_smooths_RT_by_surprisal(
                linear_smooths=linear_smooths,
                nonlinear_smooths=nonlinear_smooths,
                density_data=density_data,
                path=here(results_path, paste0(prefix_smooths_results, "surp_link.pdf")),
                x_condition_name=x_condition_name,
                y_condition_name=y_condition_name,
                x_condition_labels=x_condition_labels,
                y_condition_labels=y_condition_labels,
                RT_col=RT_col,
                additive_model=additive_model
            )
        }
    }
}