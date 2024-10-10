library(testthat)
library(dplyr)
library(here)
## Import files
source(here("src", "GAM", "preprocess_df.R"))

# Specific dummy data for testing
test_data <- data.frame(
    GD = c(200, 150, 180, 220, 170),
    FirstFixProg = c(1, 2, 1, 1, 0),
    practice = c(0, 0, 1, 0, 0),
    normalized_ID = c(2, 0, 3, 1, 4),
    not_num_or_punc = c(FALSE, TRUE, FALSE, FALSE, TRUE),
    subject_id = c(1, 2, 3, 4, 5),
    word_id = c(1, 1, 1, 1, 1),
    word_str = c("a", "a", "a", "a", "a"),
    unique_paragraph_id = c(101, 102, 103, 104, 105),
    has_preview_condition = c(TRUE, TRUE, FALSE, TRUE, FALSE),
    reread_condition = c(TRUE, FALSE, TRUE, FALSE, TRUE),
    critical_span_condition = c(TRUE, FALSE, TRUE, FALSE, TRUE),
    reread_consecutive_condition = c(TRUE, FALSE, TRUE, FALSE, TRUE),
    surp = c(10, 22, 18, 5, 25),
    prev_surp = c(15, 20, 8, 10, 17),
    len = c(5, 10, 8, 12, 6),
    prev_len = c(7, 9, 10, 11, 5),
    freq = c(10, 20, 15, 25, 5),
    prev_freq = c(12, 22, 18, 21, 14),
    FF = c(300, 250, 280, 320, 270),
    TF = c(400, 350, 380, 420, 370),
    RegPD = c(400, 350, 380, 420, 370)
)

result_dir = "results 0<RT<3000 firstpassNA"

test_that("clean_eye_df function works correctly", {
    # Apply the function for each mode and ensure rows are filtered correctly
    cleaned_df <- clean_eye_df(test_data, RT_col = "surp", result_dir)

    # Ensure specific rows are filtered out by checking the number of rows remaining
    # 'practice' == 1, 'normalized_ID' == 0 or 1, and 'not_num_or_punc' == FALSE should be filtered out
    expect_false(any(cleaned_df$practice == 1))
    expect_false(any(cleaned_df$normalized_ID %in% c(0, 1)))
    expect_true(all(cleaned_df$not_num_or_punc))

    # Check if 'FirstPassGD' column is added and processed correctly
    expect_true(all(cleaned_df$FirstPassGD == ifelse(cleaned_df$FirstFixProg != 1, NA, cleaned_df$GD)))
    expect_true(all(cleaned_df$FirstPassFF == ifelse(cleaned_df$FirstFixProg != 1, NA, cleaned_df$FF)))
    expect_true(all(cleaned_df$GoPast == ifelse(cleaned_df$FirstFixProg != 1, NA, cleaned_df$RegPD)))

    # Print the final dimensions for verification
    print(sprintf("Final dimensions - surp mode: %d rows, %d columns", nrow(cleaned_df), ncol(cleaned_df)))
    })


# Define the test
test_that("get_eye_data creates prev_surp correctly", {
    # Create a sample dataframe
    sample_data <- tibble(
        subject_id = c(
            1, 1, 1, 
            2, 2, 2, 2, 
            3, 3, 3, 3),
        unique_paragraph_id = c(
            1, 1, 1, 
            2, 2, 2, 3, 
            1, 1, 2, 2),
        IA_ID = c(
            1, 1, 1, 
            2, 2, 2, 2, 
            3, 3, 3, 3),
        IA_LABEL = c(
            "a", "a", "a", 
            "b", "b", "b", "b", 
            "c", "c", "c", "c"),
        reread = c(
            0, 0, 1, 
            0, 0, 1, 1, 
            0, 1, 1, 1),
        surp_column = c(
            0.5, 0.6, 0.7, 
            0.8, 0.9, 1.0, 1.1, 
            0.4, 0.5, 0.6, 0.7),
        has_preview = c(
            0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0),
        is_in_aspan = c(
            1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1),
        article_ind = c(
            1, 1, 2, 2, 2, 3, 2, 2, 2, 3, 3),
        Wordfreq_Frequency = c(10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110),
        prev_Wordfreq_Frequency = c(5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55),
        Length = c(100, 200, 150, 250, 300, 350, 400, 450, 500, 550, 600),
        prev_Length = c(90, 190, 140, 240, 290, 340, 390, 440, 490, 540, 590),
        IA_FIRST_FIXATION_DURATION = c(300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800),
        IA_FIRST_RUN_DWELL_TIME = c(600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600),
        IA_DWELL_TIME = c(1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000, 2100, 2200),
        IA_FIRST_FIX_PROGRESSIVE = c(200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700),
        IA_REGRESSION_PATH_DURATION = c(400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900),
        not_num_or_punc = c(1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1),
        normalized_ID = c(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1)
    )

    # Updated expected_prev_surp values
    expected_prev_surp <- c(
        NA, 0.5, NA, 
        NA, 0.8, NA, NA, 
        NA, NA, NA, 0.6
    )

    # Run the function
    result <- get_eye_data(sample_data, "surp_column")

    # Check if prev_surp is created correctly
    expect_equal(result$prev_surp, expected_prev_surp)
})


# Sample data for testing
eye_df_sample <- data.frame(
    surp = c(10, 25, 5, 15, 30),
    prev_surp = c(15, 5, 20, 10, 35),
    freq = c(10, 5, 20, 30, 3),
    prev_freq = c(6, 10, 15, 3, 20),
    len = c(5, 2, 12, 20, 1),
    prev_len = c(1, 5, 15, 10, 2)
)

# Test for filter_eye_df_by_surp
test_that("filter_eye_df_by_surp filters correctly", {
    result <- filter_eye_df_by_surp(eye_df_sample)
    
    # Expected result
    expected <- eye_df_sample %>% filter(surp < 20 & prev_surp < 20)
    
    expect_equal(result, expected)
    expect_true(all(result$surp < 20))
    expect_true(all(result$prev_surp < 20))
})

# Test for filter_eye_df_by_freq
test_that("filter_eye_df_by_freq filters correctly", {
    result <- filter_eye_df_by_freq(eye_df_sample)
    
    # Expected result
    expected <- eye_df_sample %>% 
        filter(freq < 25 & freq >= 4 & prev_freq < 25 & prev_freq >= 4)
    
    expect_equal(result, expected)
    expect_true(all(result$freq < 25 & result$freq >= 4))
    expect_true(all(result$prev_freq < 25 & result$prev_freq >= 4))
})

# Test for filter_eye_df_by_len
test_that("filter_eye_df_by_len filters correctly", {
    result <- filter_eye_df_by_len(eye_df_sample)
    
    # Expected result
    expected <- eye_df_sample %>% 
        filter(len < 15 & len >= 1 & prev_len < 15 & prev_len >= 1)
    
    expect_equal(result, expected)
    expect_true(all(result$len < 15 & result$len >= 1))
    expect_true(all(result$prev_len < 15 & result$prev_len >= 1))
})

###
# get_avg_RT_per_word_df
###

# Test 1: Basic test with only one word instance
test_that("get_avg_RT_per_word_df works for a single instance of a word", {
    RT_col <- "GD"
    # Define the input data explicitly
    input_data <- data.frame(
        unique_paragraph_id = 1,
        subject_id = 1,
        word_id = 1,
        word_str = "The",
        has_preview_condition = TRUE,
        reread_condition = FALSE,
        surp = 0.5,
        prev_surp = 0.4,
        len = 3,
        prev_len = 2,
        freq = 0.8,
        prev_freq = 0.7,
        GD = 300,  # Setting RT to 300
        stringsAsFactors = FALSE  # Ensure no factors are created automatically
    )

    # Define the expected output explicitly
    expected_output <- data.frame(
        unique_paragraph_id = 1,
        word_id = 1,
        word_str = "The",
        has_preview_condition = "TRUE",  # Converting logical to character to match the function's output
        reread_condition = "FALSE",
        surp = 0.5,
        prev_surp = 0.4,
        len = 3,
        prev_len = 2,
        freq = 0.8,
        prev_freq = 0.7,
        GD = 300,  # The mean RT is set to 300
        stringsAsFactors = FALSE  # Ensure no factors are created automatically
    )

    # Apply the function to the input data
    result <- get_avg_RT_per_word_df(input_data, RT_col)

    # Print structures for debugging (optional)
    # print("Result Data Frame:")
    # print(result)
    # print("Expected Output Data Frame:")
    # print(expected_output)

    # Check if the result matches the expected output
    expect_equal(result, expected_output)
})


# Test 2: Test with repeated instances of the same word with different RT values
test_that("get_avg_RT_per_word_df calculates mean RT correctly for repeated instances", {
    # Define the input data frame explicitly
    input_data <- data.frame(
        unique_paragraph_id = c(1, 1, 1),
        subject_id = c(1, 2, 3),
        word_id = c(1, 1, 1),
        word_str = c("The", "The", "The"),
        has_preview_condition = c(TRUE, TRUE, TRUE),
        reread_condition = c(FALSE, FALSE, FALSE),
        surp = c(0.5, 0.5, 0.5),
        prev_surp = c(0.4, 0.4, 0.4),
        len = c(3, 3, 3),
        prev_len = c(2, 2, 2),
        freq = c(0.8, 0.8, 0.8),
        prev_freq = c(0.7, 0.7, 0.7),
        RT = c(300, 350, 400),  # Different RT values for the same word "The"
        stringsAsFactors = FALSE  # Ensure no factors are created automatically
    )

    # Define the expected output data frame explicitly
    expected_output <- data.frame(
        unique_paragraph_id = 1,
        word_id = 1,
        word_str = "The",
        has_preview_condition = "TRUE",  # Character type to match function output
        reread_condition = "FALSE",
        surp = 0.5,
        prev_surp = 0.4,
        len = 3,
        prev_len = 2,
        freq = 0.8,
        prev_freq = 0.7,
        RT = 350,  # The mean RT for "The" is (300 + 350 + 400) / 3 = 350
        stringsAsFactors = FALSE  # Ensure no factors are created automatically
    )

    # Apply the function to the input data
    result <- get_avg_RT_per_word_df(input_data, "RT")

    # Check if the result matches the expected output
    expect_equal(result, expected_output)
})


# Test 3: Test with multiple words and unique combinations
test_that("get_avg_RT_per_word_df works for multiple unique word combinations", {
    mock_data <- data.frame(
        unique_paragraph_id = c(1, 1, 1, 1),
        subject_id = c(1, 2, 1, 2),
        word_id = c(1, 1, 2, 2),
        word_str = c("The", "The", "cat", "cat"),
        has_preview_condition = c(TRUE, TRUE, TRUE, TRUE),
        reread_condition = c(FALSE, FALSE, TRUE, TRUE),
        surp = c(0.5, 0.5, 1.2, 1.2),
        prev_surp = c(0.4, 0.4, 0.3, 0.3),
        len = c(3, 3, 4, 4),
        prev_len = c(2, 2, 3, 3),
        freq = c(0.8, 0.8, 0.3, 0.3),
        prev_freq = c(0.7, 0.7, 0.6, 0.6),
        RT = c(300, 350, 400, 450)  # Different RT values for "The" and "cat"
    )

    # Expected output:
    # Mean RT of "The" = (300 + 350) / 2 = 325
    # Mean RT of "cat" = (400 + 450) / 2 = 425
    expected_output <- data.frame(
        unique_paragraph_id = c(1, 1),
        word_id = c(1, 2),
        word_str = c("The", "cat"),
        has_preview_condition = c(TRUE, TRUE),
        reread_condition = c(FALSE, TRUE),
        surp = c(0.5, 1.2),
        prev_surp = c(0.4, 0.3),
        len = c(3, 4),
        prev_len = c(2, 3),
        freq = c(0.8, 0.3),
        prev_freq = c(0.7, 0.6),
        RT = c(325, 425)
    )

        # Convert all logical columns to character
    expected_output[] <- lapply(expected_output, function(col) {
        if (is.logical(col)) as.character(col) else col
    })

    result <- get_avg_RT_per_word_df(mock_data, "RT")
    expect_equal(result, expected_output)
})

test_that("get_preprocess_cond works", {
    # Test cases
    res <- get_preprocess_cond("results 0<RT<3000 firstpassNA bla")
    expect_equal(res, list(fill_notFirstFixProg = NA, filter_RT_greater_than_0 = TRUE))

    res <- get_preprocess_cond("results 0<=RT<3000 firstpass0 bla")
    expect_equal(res, list(fill_notFirstFixProg = 0, filter_RT_greater_than_0 = FALSE))

    # Test for an invalid input that should raise an error
    expect_error(get_preprocess_cond("results bla"))
})