library(testthat)
library(dplyr)
library(here)
## Import files
source(here("src", "GAM", "GAM_analysis_funcs.R"))

library(testthat)
library(here)

# Define a test for the save_filter_params function
test_that("save_filter_params correctly appends data to the CSV file", {
  # Define dummy parameters
  zoom_in <- TRUE
  zoom_in_threshold <- 0.75
  reread <- TRUE
  reread_10_11 <- 10.11
  filter_hunting <- "Hunting"

  results_path <- here("src","GAM","results 0<RT<3000 firstpassNA","results_context","et_20240505_large_models_context_cols_20240625","zoom_in")
  RT_col <- "FirstPassGD"
  surp_column <- "test_col"
  
  # Expected file path
  new_path <- here(results_path, paste0(RT_col, " - ", surp_column, "_log_filter_params.csv"))

  # Remove the file if it exists from previous tests
  if (file.exists(new_path)) {
    file.remove(new_path)
  }

  list2env(list(
        results_path = results_path,
        RT_col = RT_col,
        surp_column = surp_column), .GlobalEnv)

  # Call the function to save parameters
  save_filter_params(zoom_in, zoom_in_threshold, reread, reread_10_11, filter_hunting)

  # Check if the file is created
  expect_true(file.exists(new_path))

  # Read the contents of the file
  saved_data <- read.csv(new_path)

  # Check that the data was saved correctly
  expect_equal(nrow(saved_data), 1)  # Should be one row
  expect_equal(saved_data$surp_column, surp_column)
  expect_equal(saved_data$zoom_in, zoom_in)
  expect_equal(saved_data$zoom_in_threshold, zoom_in_threshold)
  expect_equal(saved_data$reread, reread)
  expect_equal(saved_data$reread_10_11, reread_10_11)
  expect_equal(saved_data$filter_hunting, filter_hunting)

  # Clean up: remove the test file
  file.remove(new_path)
})

