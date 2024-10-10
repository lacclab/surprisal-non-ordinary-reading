library(testthat)
library(dplyr)
library(here)
## Import files
source(here("src", "GAM", "GAM_analysis_funcs.R"))

# -----------------------------------------------------------------
# Test Function get_models_forms
# -----------------------------------------------------------------

# Test for RE = TRUE
test_that("get_models_forms generates correct formulas with random effects", {
    # Define the inputs
    models_names <- c("baseline", "linear", "nonlinear")
    RT_col <- "RT"
    RE <- TRUE
    
    # Call the function
    model_forms <- get_models_forms(models_names, RE, RT_col)
    
    # Expected outputs (converted to formulas)
    expected_forms <- list(
        baseline = paste0(
            RT_col, " ~ te(freq, len, bs = 'cr') + ",
            "te(prev_freq, prev_len, bs = 'cr') + ",
            "s(subject_id, bs='re') + te(subject_id, freq, len, bs='re') + ",
            "s(unique_paragraph_id, bs='re') + te(unique_paragraph_id, freq, len, bs='re')"
        ),
        linear = paste0(
            RT_col, " ~ surp + prev_surp + te(freq, len, bs = 'cr') + ",
            "te(prev_freq, prev_len, bs = 'cr') + ",
            "s(subject_id, bs='re') + s(subject_id, surp, bs='re') + ",
            "te(subject_id, freq, len, bs='re') + s(unique_paragraph_id, bs='re') + ",
            "s(unique_paragraph_id, surp, bs='re') + te(unique_paragraph_id, freq, len, bs='re')"
        ),
        nonlinear = paste0(
            RT_col, " ~ s(surp, bs = 'cr', k = 6) + s(prev_surp, bs = 'cr', k = 6) + ",
            "te(freq, len, bs = 'cr') + te(prev_freq, prev_len, bs = 'cr') + ",
            "s(subject_id, bs='re') + s(subject_id, surp, bs='re') + ",
            "te(subject_id, freq, len, bs='re') + s(unique_paragraph_id, bs='re') + ",
            "s(unique_paragraph_id, surp, bs='re') + te(unique_paragraph_id, freq, len, bs='re')"
        ),
    )
    
    # Test if the generated formulas match the expected formulas
    expect_equal(model_forms, expected_forms)
})

# Test for RE = FALSE
test_that("get_models_forms generates correct formulas without random effects", {
    # Define the inputs
    models_names <- c("baseline", "linear", "nonlinear")
    RT_col <- "RT"
    RE <- FALSE
    
    # Call the function
    model_forms <- get_models_forms(models_names, RE, RT_col)
    
    # Expected outputs (converted to formulas)
    expected_forms <- list(
        baseline = paste0(
            RT_col, " ~ te(freq, len, bs = 'cr') + ",
            "te(prev_freq, prev_len, bs = 'cr')"
        ),
        linear = paste0(
            RT_col, " ~ surp + prev_surp + te(freq, len, bs = 'cr') + ",
            "te(prev_freq, prev_len, bs = 'cr')"
        ),
        nonlinear = paste0(
            RT_col, " ~ s(surp, bs = 'cr', k = 6) + s(prev_surp, bs = 'cr', k = 6) + ",
            "te(freq, len, bs = 'cr') + te(prev_freq, prev_len, bs = 'cr')"
        ),
    )
    
    # Test if the generated formulas match the expected formulas
    expect_equal(model_forms, expected_forms)
})

# Test for RE = FALSE, partial models list
test_that("get_models_forms generates correct formulas without random effects", {
    # Define the inputs
    models_names <- c("baseline", "linear", "nonlinear")
    RT_col <- "RT"
    RE <- FALSE
    
    # Call the function
    model_forms <- get_models_forms(models_names, RE, RT_col)
    
    # Expected outputs (converted to formulas)
    expected_forms <- list(
        baseline = "RT ~ te(freq, len, bs = 'cr') + te(prev_freq, prev_len, bs = 'cr')",
        linear = "RT ~ surp + prev_surp + te(freq, len, bs = 'cr') + te(prev_freq, prev_len, bs = 'cr')",
        nonlinear = "RT ~ s(surp, bs = 'cr', k = 6) + s(prev_surp, bs = 'cr', k = 6) + te(freq, len, bs = 'cr') + te(prev_freq, prev_len, bs = 'cr')"
    )
    
    # Test if the generated formulas match the expected formulas
    expect_equal(model_forms, expected_forms)
})