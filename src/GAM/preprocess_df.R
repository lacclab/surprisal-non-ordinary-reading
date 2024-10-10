# install.packages("tidyr")
library(tidyr)

add_prev_surp <- function(df) {
    df <- df %>%
    group_by(subject_id, unique_paragraph_id, reread_condition) %>%
    mutate(
        prev_surp = lag(surp)
    ) %>%
    ungroup()
    return(df)
}

get_eye_data <- function(input, surp_column) {
    cat("Getting data...","\n")

    if (is.character(input)) {
        df <- read_csv(input)
    } else {
        df <- input
    }

    cat("df dim: ", dim(df)[1]," rows, ", dim(df)[2], " columns", "\n")

    prev_surp_column <- paste0("prev_", surp_column)
    common_mutate <- . %>%
        mutate(
            subject_id = as.factor(subject_id),
            unique_paragraph_id = as.factor(unique_paragraph_id),
            word_id = as.factor(IA_ID),
            word_str = as.factor(IA_LABEL),
            question_id = as.factor(q_ind),
            has_preview_condition = as.factor(has_preview),
            reread_condition = as.factor(reread),
            critical_span_condition = as.numeric(is_in_aspan),
            reread_consecutive_condition = as.numeric(article_ind),
            surp = as.numeric(!!sym(surp_column)),
            freq = as.numeric(Wordfreq_Frequency),
            prev_freq = as.numeric(prev_Wordfreq_Frequency),
            len = as.numeric(Length),    
            prev_len = as.numeric(prev_Length),
            FF = as.numeric(IA_FIRST_FIXATION_DURATION),
            GD = as.numeric(IA_FIRST_RUN_DWELL_TIME),
            TF = as.numeric(IA_DWELL_TIME),
            FirstFixProg = as.numeric(IA_FIRST_FIX_PROGRESSIVE),
            RegPD = as.numeric(IA_REGRESSION_PATH_DURATION),
            not_num_or_punc = as.numeric(not_num_or_punc),
            normalized_ID = as.numeric(normalized_ID)
        )

    df = df %>% common_mutate

    if (!prev_surp_column %in% colnames(df)) {
        df = add_prev_surp(df)
    } else {
        df <- df %>%
            mutate(
                prev_surp = as.numeric(!!sym(prev_surp_column))
            )
    }

    return(df)
}

# Threshold parameters
EYE_MEASUREMENT_UPPER_THRESH <- 3000
EYE_MEASUREMENT_LOWER_THRESH <- 0
SURP_THRESH <- 20
FREQ_LOWER_THRESH <- 4
FREQ_UPPER_THRESH <- 25
LEN_LOWER_THRESH <- 1
LEN_UPPER_THRESH <- 15

filter_eye_df_by_surp <- function(eye_df){
    cat("Preprocess | filter surp vals","\n")
    eye_df <- eye_df %>% filter(surp < SURP_THRESH)
    eye_df <- eye_df %>% filter(prev_surp < SURP_THRESH)
    cat("Preprocess | eye_df dim: ", dim(eye_df)[1], " rows,", dim(eye_df)[2], " columns","\n")
    return(eye_df)
}

filter_eye_df_by_freq <- function(eye_df){
    cat("Preprocess | filter freq vals","\n")
    eye_df <- eye_df %>% filter(freq < FREQ_UPPER_THRESH)
    eye_df <- eye_df %>% filter(freq >= FREQ_LOWER_THRESH)
    eye_df <- eye_df %>% filter(prev_freq < FREQ_UPPER_THRESH)
    eye_df <- eye_df %>% filter(prev_freq >= FREQ_LOWER_THRESH)
    cat("Preprocess | eye_df dim: ", dim(eye_df)[1], " rows,", dim(eye_df)[2], " columns","\n")
    return(eye_df)
}

filter_eye_df_by_len <- function(eye_df){
    cat("Preprocess | filter len vals","\n")
    eye_df <- eye_df %>% filter(len < LEN_UPPER_THRESH)
    eye_df <- eye_df %>% filter(len >= LEN_LOWER_THRESH)
    eye_df <- eye_df %>% filter(prev_len < LEN_UPPER_THRESH)
    eye_df <- eye_df %>% filter(prev_len >= LEN_LOWER_THRESH)
    cat("Preprocess | eye_df dim: ", dim(eye_df)[1], " rows,", dim(eye_df)[2], " columns","\n")
    return(eye_df)
}

get_preprocess_cond <- function(result_dir) {
    cat("Preprocess | Preprocessing eye_df...\n")

    # Use grepl() to check for substrings
    if (grepl("results 0<RT<3000 firstpassNA", result_dir)) {
        fill_notFirstFixProg <- NA
        filter_RT_greater_than_0 <- TRUE
    } else if (grepl("results 0<=RT<3000 firstpass0", result_dir)) {
        fill_notFirstFixProg <- 0
        filter_RT_greater_than_0 <- FALSE
    } else {
        stop("ValueError: need to configure fill_notFirstFixProg in clean_eye_df func")
    }
    
    # Return the values as a list for checking
    return(list(fill_notFirstFixProg = fill_notFirstFixProg, filter_RT_greater_than_0 = filter_RT_greater_than_0))
}

clean_eye_df <- function(eye_df, RT_col, result_dir, surp_norm=FALSE){
    cat("Preprocess | Preprocessing eye_df...\n")
    cat("Preprocess | RT_col= ",RT_col, "\n")

    preprocess_cond = get_preprocess_cond(result_dir)
    fill_notFirstFixProg = preprocess_cond$fill_notFirstFixProg
    filter_RT_greater_than_0 = preprocess_cond$filter_RT_greater_than_0
    
    cat("Preprocess| result_dir = ", result_dir, "\n")
    cat("Preprocess| fill_notFirstFixProg = ", fill_notFirstFixProg, "\n")
    cat("Preprocess| filter_RT_greater_than_0 = ", filter_RT_greater_than_0, "\n")

    eye_df <- eye_df %>% 
    mutate(FirstPassGD = GD) %>% # add FirstPassGD column
    mutate(FirstPassGD = ifelse(FirstFixProg != 1, fill_notFirstFixProg, FirstPassGD))

    eye_df <- eye_df %>% 
    mutate(FirstPassFF = FF) %>% # add FirstPassFF column
    mutate(FirstPassFF = ifelse(FirstFixProg != 1, fill_notFirstFixProg, FirstPassFF))
    
    eye_df <- eye_df %>% 
    mutate(GoPast = RegPD) %>% # add GoPast column (Go-Past Duration)
    mutate(GoPast = ifelse(FirstFixProg != 1, fill_notFirstFixProg, GoPast))

    # filter out practice rows
    eye_df <- eye_df %>% filter(practice == 0)
    cat("Preprocess | Filter practice == 0 | eye_df rows: ", dim(eye_df)[1], "\n")

    # filter
    eye_df <- filter(eye_df, !!sym(RT_col) >= 0) # removes also nulls
    cat("Preprocess | Filter RT_col >= 0 (null values) | eye_df rows: ", dim(eye_df)[1],"\n")

    # filter
    eye_df <- filter(eye_df, !!sym(RT_col) <= EYE_MEASUREMENT_UPPER_THRESH) # removes also nulls
    cat("Preprocess | Filter RT_col <=" ,EYE_MEASUREMENT_UPPER_THRESH, " | eye_df rows: ", dim(eye_df)[1], "\n")

    # filters
    if (filter_RT_greater_than_0 == TRUE){
        eye_df <- filter(eye_df, !!sym(RT_col) > EYE_MEASUREMENT_LOWER_THRESH) # removes also nulls
        cat("Preprocess | Filter RT_col >" ,EYE_MEASUREMENT_LOWER_THRESH, " (so also nulls) | eye_df rows: ", dim(eye_df)[1], "\n")
    }
    
    # filter out start and end of paragrpah
    eye_df <- eye_df %>% filter(normalized_ID != 0 & normalized_ID != 1)
    cat("Preprocess | Filter out start and end of paragrpah | eye_df rows: ", dim(eye_df)[1], "\n")

    # filter out words with numbers or punctuation
    eye_df <- eye_df %>% filter(not_num_or_punc == 1)
    cat("Preprocess | Filter out num or punc | eye_df rows: ", dim(eye_df)[1], "\n")
    
    # Drop nulls from prev columns
    eye_df <- eye_df %>% drop_na(
        prev_surp,
        prev_len, 
        prev_freq)
    cat("Preprocess | Filter out nulls from prev cols | eye_df rows: ", dim(eye_df)[1], "\n")

    # select
    if (!"question_id" %in% colnames(eye_df)) {
        stop("'q_ind' column is missing in the input DataFrame.")
    }
    eye_df <- eye_df %>% select(
        subject_id, unique_paragraph_id,
        word_id, word_str, question_id,
        has_preview_condition, reread_condition, critical_span_condition, reread_consecutive_condition,
        surp, prev_surp,
        len, prev_len, 
        freq, prev_freq, 
        FF, GD, TF, FirstPassGD, FirstPassFF, GoPast)

    cat("Preprocess | clean eye_df dim: ", dim(eye_df)[1], " rows,", dim(eye_df)[2], " columns", "\n")

    round_p <- function(x) {format(round(x, 8), scientific = FALSE)}
    z_normalize <- function(x) {(x - mean(x)) / sd(x)}
    scale_range <- function(y, a, b) {
        y_min <- min(y)
        y_max <- max(y)
        scaled_y <- a + (y - y_min) * (b - a) / (y_max - y_min)
        return(scaled_y)
    }
    if (surp_norm == "z_norm")
    {
        cat("Preprocess | Performing Z-score normalization on surprisal | mean: " ,round_p(mean(eye_df$surp)), "std: ", round_p(sd(eye_df$surp)))
        cat(" | prev_surp stats | mean: " ,round_p(mean(eye_df$prev_surp)), "std: ", round_p(sd(eye_df$prev_surp)), "\n")
        eye_df$surp <- z_normalize(eye_df$surp)
        eye_df = add_prev_surp(eye_df)
        cat("Preprocess | After Normalization | mean: " ,mean(eye_df$surp), "std: ", sd(eye_df$surp))
        cat(" | prev_surp stats | mean: " ,round_p(mean(eye_df$prev_surp)), "std: ", round_p(sd(eye_df$prev_surp)), "\n")
    }
    if (surp_norm == "scale-0-20")
    {
        cat("Preprocess | Performing scale-0-20 normalization on surprisal | mean: " ,round_p(mean(eye_df$surp)), "std: ", round_p(sd(eye_df$surp)), "\n")
        cat(" | prev_surp stats | mean: " ,round_p(mean(eye_df$prev_surp)), "std: ", round_p(sd(eye_df$prev_surp)), "\n")
        eye_df$surp <- scale_range(eye_df$surp, a=0, b=20)
        eye_df = add_prev_surp(eye_df)
        cat("Preprocess | After Normalization | mean: " ,mean(eye_df$surp), "std: ", sd(eye_df$surp), "\n")
        cat(" | prev_surp stats | mean: " ,round_p(mean(eye_df$prev_surp, na.rm = TRUE)), "std: ", round_p(sd(eye_df$prev_surp, na.rm=TRUE)), "\n")
    }

    # add_surp_sqaured columns
    eye_df <- add_surp_sqaured(eye_df)

    if (grepl("avg_RT_per_word", result_dir)){
        eye_df = get_avg_RT_per_word_df(eye_df, RT_col)
        cat("Preprocess | Calculating avg RT per word | eye_df rows: ", dim(eye_df)[1], "\n")
    }

    return(eye_df)
}

add_surp_sqaured <- function(df){
    cat("Preprocess | add surp sqared...","\n")
    df$surp_squared <- df$surp^2
    df$prev_surp_squared <- df$prev_surp^2
    return(df)
}

get_avg_RT_per_word_df <- function(eye_df, RT_col){
    # Define the columns to group by
    text_cond_cols <- c(
        "unique_paragraph_id",
        "word_id", "word_str", "question_id",
        "has_preview_condition", "reread_condition",
        "surp", "prev_surp",
        "len", "prev_len",
        "freq", "prev_freq"
    )
    
    # Use dplyr for grouping and calculating mean
    avg_RT_df <- eye_df %>%
        group_by(across(all_of(text_cond_cols))) %>%
        summarise(mean_RT = mean(.data[[RT_col]], na.rm = TRUE), .groups = "drop") %>% 
        as.data.frame()

    # Convert all logical columns to character
    avg_RT_df[] <- lapply(avg_RT_df, function(col) {
        if (is.logical(col)) as.character(col) else col
    })

    eye_df <- avg_RT_df
    
    # # drop duplicates
    # drop_dup_by <- c(
    #     "unique_paragraph_id",
    #     "word_id", "word_str", "question_id",
    #     "has_preview_condition", "reread_condition",
    #     "surp", "prev_surp",
    #     "len", "prev_len",
    #     "freq", "prev_freq"
    # )
    # dedup_df <- eye_df %>% distinct(across(all_of(drop_dup_by)))
    # # merge dedup_df with avg_RT_df
    # eye_df <- merge(dedup_df, avg_RT_df, by=text_cond_cols, all.x=TRUE) # all.x=TRUE ensures a left join

    # Rename mean_RT to RT_col
    eye_df <- eye_df %>% rename(!!RT_col := mean_RT) 
    return(eye_df)
}