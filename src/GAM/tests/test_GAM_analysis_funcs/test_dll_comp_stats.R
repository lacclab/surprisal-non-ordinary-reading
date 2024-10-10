library(testthat)
library(dplyr)
library(here)
## Import files
source(here("src", "GAM", "GAM_analysis_funcs.R"))

# -----------------------------------------------------------------
# Test Function update_dll_comp_stats_df
# -----------------------------------------------------------------

test_that("update_dll_comp_stats_df correctly appends rows", {
    # Initialize an empty data frame
    dll_comp_stats_df <- data.frame()

    # Example input data
    models_dll_list <- list(
        dll_linear = c(0.85, 0.88, 0.87),
        dll_nonlinear = c(0.95, 0.97, 0.92)
    )
    
    x_name = "x1"
    y_name = "y1"
    x_condition_name <- x_name
    y_condition_name <- y_name
    list2env(list(x_condition_name=x_condition_name, y_condition_name=y_condition_name), .GlobalEnv)
    
    x_condition_val <- "ConditionX"
    y_condition_val <- "ConditionY"
    
    # Call the function
    updated_df <- update_dll_comp_stats_df(
        dll_comp_stats_df, 
        models_dll_list, 
        x_condition_val, 
        y_condition_val
    )
    
    # Expected output
    expected_df <- data.frame(
        dll_linear = c(0.85, 0.88, 0.87),
        dll_nonlinear = c(0.95, 0.97, 0.92),
        x1 = rep("ConditionX", 3),
        y1 = rep("ConditionY", 3)
    )
    
    # Check if the data frames are equal
    expect_equal(updated_df, expected_df)
})
