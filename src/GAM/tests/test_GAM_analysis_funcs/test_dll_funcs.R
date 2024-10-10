library(testthat)
library(dplyr)
library(here)
## Import files
source(here("src", "GAM", "GAM_analysis_funcs.R"))

# -----------------------------------------------------------------
# Test DLL functions: 
#   - get_target_model_dll_df
#   - get_baseline_model_dll_df
#   - get_dll_vec_by_params
# -----------------------------------------------------------------

# Set the seed for reproducibility
set.seed(123)

# Sample data frame
dll_raw_df <- data.frame(
  model_name = rep(c("baseline", "linear", "nonlinear"), each = 40),
  subject_id = factor(rep(1:20, 10)),
  unique_paragraph_id = factor(rep(paste0("Para_", 1:20), 10)),
  RT_true = sample(100:300, 200, replace = TRUE),
  RT_pred = sample(100:300, 200, replace = TRUE),
  logliks = rnorm(200, mean = -5, sd = 2)
)
x_condition_name="has_preview_condition"
y_condition_name="reread_condition"

# Calculate squared_error
dll_raw_df$squared_error <- (dll_raw_df$RT_pred - dll_raw_df$RT_true)^2

# Add has_preview_condition column with "Gathering" or "Hunting"
dll_raw_df$has_preview_condition <- rep(c("Gathering", "Hunting"), each = 20, length.out = nrow(dll_raw_df))

# Add reread_condition column with 0 or 1
dll_raw_df$reread_condition <- rep(c(0, 1), each = 10, length.out = nrow(dll_raw_df))

# Shuffle the rows to ensure random distribution
dll_raw_df <- dll_raw_df[sample(nrow(dll_raw_df)), ]

# Reset row names after shuffling
rownames(dll_raw_df) <- NULL

# Print the first few rows to check the data
head(dll_raw_df)

# get_target_model_dll_df test
test_that("get_target_model_dll_df returns correct dataframe", {
  x_condition_val <- "Hunting"  # or "Gathering"
  y_condition_val <- 1
  model_name <- "linear"
  
  result <- get_target_model_dll_df(dll_raw_df, x_condition_val, y_condition_val, model_name)
  
  expect_true(is.data.frame(result))
  expect_true(all(result$model_name == model_name))
  expect_true(all(result$has_preview_condition == x_condition_val))
  expect_true(all(result$reread_condition == y_condition_val))
})

# get_baseline_model_dll_df test
test_that("get_baseline_model_dll_df returns correct dataframe", {
  x_condition_val <- "Gathering"  # or "Hunting"
  y_condition_val <- 0
  
  result <- get_baseline_model_dll_df(dll_raw_df, x_condition_val, y_condition_val)
  
  expect_true(is.data.frame(result))
  expect_true(all(result$model_name == "baseline"))
  expect_true(all(result$has_preview_condition == x_condition_val))
  expect_true(all(result$reread_condition == y_condition_val))
})

# get_dll_vec_by_params test
test_that("get_dll_vec_by_params returns correct delta log likelihood vector", {
  x_condition_val <- "Hunting"  # or "Gathering"
  y_condition_val <- 1
  model_name <- "linear"
  
  dll_vec <- get_dll_vec_by_params(dll_raw_df, x_condition_val, y_condition_val, model_name)
  
  expect_true(is.numeric(dll_vec))
  expect_equal(length(dll_vec), nrow(dll_raw_df[dll_raw_df$model_name == model_name &
                                                dll_raw_df$has_preview_condition == x_condition_val &
                                                dll_raw_df$reread_condition == y_condition_val, ]))
})

# get_delta_SE_vec_by_params test
test_that("get_delta_SE_vec_by_params returns correct delta squared error vector", {
  x_condition_val <- "Gathering"  # or "Hunting"
  y_condition_val <- 0
  model_name <- "linear"
  
  delta_SE_vec <- get_delta_SE_vec_by_params(dll_raw_df, x_condition_val, y_condition_val, model_name)
  
  expect_true(is.numeric(delta_SE_vec))
  expect_equal(length(delta_SE_vec), nrow(dll_raw_df[dll_raw_df$model_name == model_name &
                                                     dll_raw_df$has_preview_condition == x_condition_val &
                                                     dll_raw_df$reread_condition == y_condition_val, ]))
})
