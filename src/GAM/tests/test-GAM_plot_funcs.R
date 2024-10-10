library(testthat)
# install.packages("here")
library(here)

source(here("src", "GAM", "plot_funcs", "GAM_plot_funcs.R"))

test_that("get_ylim_vals_by_mode_and_outcome returns correct y-axis limits for different modes and outcomes", {
    expect_equal(get_ylim_vals_by_mode_and_outcome("surp", "IA_FIRST_RUN_DWELL_TIME"), c(-25, 35))
    expect_equal(get_ylim_vals_by_mode_and_outcome("freq", "IA_FIRST_RUN_DWELL_TIME"), c(-27, 60))
    expect_equal(get_ylim_vals_by_mode_and_outcome("len", "IA_FIRST_RUN_DWELL_TIME"), c(-80, 180))

    expect_equal(get_ylim_vals_by_mode_and_outcome("surp", "IA_DWELL_TIME"), c(-50, 100))
    expect_equal(get_ylim_vals_by_mode_and_outcome("freq", "IA_DWELL_TIME"), c(-75, 120))
    expect_equal(get_ylim_vals_by_mode_and_outcome("len", "IA_DWELL_TIME"), c(-150, 250))

    expect_equal(get_ylim_vals_by_mode_and_outcome("surp", "IA_FIRST_FIXATION_DURATION"), c(-10, 20))
    expect_equal(get_ylim_vals_by_mode_and_outcome("freq", "IA_FIRST_FIXATION_DURATION"), c(-30, 30))
    expect_equal(get_ylim_vals_by_mode_and_outcome("len", "IA_FIRST_FIXATION_DURATION"), c(-60, 60))

    # Test for unknown outcome
    expect_null(get_ylim_vals_by_mode_and_outcome("surp", "unknown_outcome"))
}
)
