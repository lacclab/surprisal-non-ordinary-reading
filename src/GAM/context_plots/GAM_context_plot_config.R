
get_surp_cols_by_article_level <- function(article_level, rr) {
    if (rr == TRUE){
        if (article_level == "paragraph"){
            surp_cols = c(
                paste0(model_name, "-Surprisal-Context-P-P"), 
                paste0(model_name, "-Surprisal-Context-Qfr-P-Q-P"),
                paste0(model_name, "-Surprisal-Context-instFirstReadP-Qfr-P-instReReadP-Q-P"),
                paste0(model_name, "-Surprisal-Context-instFirstReadP-P-instReReadP-P")
            )
        } else if (article_level == "article") {
            surp_cols = c(
                paste0(model_name, "-Surprisal-Context-Article-pastP-P"), 
                paste0(model_name, "-Surprisal-Context-ArticleQfrP-pastQP-Q-P"),
                paste0(model_name, "-Surprisal-Context-instFirstReadA-ArticleQfrP-instReReadA-pastQP-Q-P"),
                paste0(model_name, "-Surprisal-Context-instFirstReadA-Article-instReReadA-pastP-P")
            )
        }
    }
    else {
        if (article_level == "paragraph"){
            gpt2_surp_list = read.csv(here("src", "GAM", "cols_lists", paste0("surp_gpt2-context_cols.csv")))$surp_col
            surp_cols = gsub("gpt2", model_name, gpt2_surp_list)
        } else if (article_level == "article") {
            pythia70m_article_surp_list = read.csv(here("src", "GAM", "cols_lists", paste0("surp_pythia70m_article_cols.csv")))$surp_col
            surp_cols = gsub("EleutherAI-pythia-70m", model_name, pythia70m_article_surp_list)
        }
    }
    return(surp_cols)
}

get_context_by_article_level <- function(article_level, rr) {
    if (rr == TRUE){
        if (article_level == "paragraph"){
            cat("context_cols by paragraph, rr \n")
            context_cols = c("P-P", "Qfr-P-Q-P", "instFirstReadP-Qfr-P-instReReadP-Q-P", "instFirstReadP-P-instReReadP-P")
        } else if (article_level == "article") {
            cat("context_cols by article, rr \n")
            context_cols =  c(
                "Article-pastP-P", "ArticleQfrP-pastQP-Q-P", 
                "instFirstReadA-ArticleQfrP-instReReadA-pastQP-Q-P", 
                "instFirstReadA-Article-instReReadA-pastP-P"
            )
        } else {
            cat("unknown article_level: ", article_level, " \n")
        }
    }
    else{
        if (article_level == "paragraph"){
            cat("context_cols by paragraph, fr \n")
            context_cols = read.csv(here("src", "GAM", "cols_lists", paste0("surp_gpt2-context_cols.csv")))$name
        } else if (article_level == "article") {
            cat("context_cols by article, fr \n")
            context_cols =  read.csv(here("src", "GAM", "cols_lists", paste0("surp_pythia70m_article_cols.csv")))$name
        } else {
            cat("unknown article_level: ", article_level, " \n")
        }
    }
    return(context_cols)
}

get_context_plot_configs <- function(
    context_plot_type,
    model_name,
    result_dir,
    file_name,
    plot_fillers,
    article_level_val
    ) {
    
    RE = F
    use_CV = T
    additive_model_name = T
    RT_col = "FirstPassGD"
    x_condition_name="has_preview_condition"
    x_condition_labels <- c("Ordinary Reading", "Information Seeking")
    x_condition_vals=c("Gathering", "Hunting")
    zoom_in_only_for_rr = FALSE
    article_level = FALSE
    results_path_zoom_in = "None"
    rr_surp_cols = "None"
    rr_context_cols = "None"

    if (context_plot_type == "zoom_in") {
        results_path = here("src", "GAM", result_dir, "results_context", file_name, "zoom_in")
        
        surp_list = get_surp_cols_by_article_level(article_level=article_level_val, rr=TRUE)
        context_list = get_context_by_article_level(article_level=article_level_val, rr=TRUE)
        
        y_condition_name="reread_condition"
        y_condition_labels=c("Repeated Reading")
        y_condition_vals=c(1)

    } else if (context_plot_type == "zoom_in_only_for_rr") {
        results_path = here("src", "GAM", result_dir, "results_context", file_name)
        results_path_zoom_in = here("src", "GAM", result_dir, "results_context", file_name, "zoom_in")
        
        surp_list = get_surp_cols_by_article_level(article_level=article_level_val, rr=FALSE)
        context_list = get_context_by_article_level(article_level=article_level_val, rr=FALSE)
        rr_surp_cols = get_surp_cols_by_article_level(article_level=article_level_val, rr=TRUE)
        rr_context_cols = get_context_by_article_level(article_level=article_level_val, rr=TRUE)
        
        y_condition_name="reread_condition"
        y_condition_labels <- c("First Reading", "Repeated Reading")
        y_condition_vals=c(0, 1)

        zoom_in_only_for_rr=TRUE

    } else { # regular
        results_path = here("src", "GAM", result_dir, "results_context", file_name)
        gpt2_surp_list = read.csv(here("src", "GAM", "cols_lists", paste0("surp_gpt2-context_cols.csv")))$surp_col
        surp_list = get_surp_cols_by_article_level(article_level=article_level_val, rr=FALSE)
        context_list = get_context_by_article_level(article_level=article_level_val, rr=FALSE)
        y_condition_name="reread_condition"
        y_condition_labels <- c("First Reading", "Repeated Reading")
        y_condition_vals=c(0, 1)
    }

    # paths
    if (article_level_val=="article"){
        dll_path = here(results_path, paste0("Article-Context-", RT_col, "-", model_name, "-", context_plot_type, "-fillers=", plot_fillers, "-dll.pdf"))
        surp_link_path = here(results_path, paste0("Article-Context-", RT_col, "-", model_name, "-", context_plot_type, "-fillers=", plot_fillers, "-surp_link.pdf"))
        prefix_context_results = paste0("Article-Context-", RT_col, "-", model_name, "-fillers=", plot_fillers, "-")
    } else {
        dll_path = here(results_path, paste0("Context-", RT_col, "-", model_name, "-", context_plot_type, "-fillers=", plot_fillers, "-dll.pdf"))
        surp_link_path = here(results_path, paste0("Context-", RT_col, "-", model_name, "-", context_plot_type, "-fillers=", plot_fillers, "-surp_link.pdf"))
        prefix_context_results = paste0("Context-", RT_col, "-", model_name, "-fillers=", plot_fillers, "-")
    }

    if (context_plot_type == "zoom_in") {
        # max_x <- c(3, 0.3, 0.05, 0.005)
        # quan <- c(99, 95, 90, "below_90")
        max_x <- c(0.0075)
        quan <- c("zoom_in")
        surp_link_path = here(results_path, paste0("Context-", RT_col, "-", model_name, "-", "-surp_link_plot-quantile", quantile_value, "_xMax=", max_value, ".pdf"))
    } else{
        max_x = 20
        quan = "None"
    }

    return(list(
        article_level = article_level,
        results_path = results_path,
        results_path_zoom_in = results_path_zoom_in,
        dll_path = dll_path,
        surp_link_path = surp_link_path,
        surp_list = surp_list,
        context_list = context_list,
        zoom_in_only_for_rr = zoom_in_only_for_rr,
        rr_surp_cols = rr_surp_cols,
        rr_context_cols = rr_context_cols,
        x_condition_name = x_condition_name,
        x_condition_labels = x_condition_labels,
        x_condition_vals = x_condition_vals,
        y_condition_name = y_condition_name,
        y_condition_labels = y_condition_labels,
        y_condition_vals = y_condition_vals,
        RE = RE,
        use_CV = use_CV,
        additive_model_name = additive_model_name,
        RT_col = RT_col,
        max_x = max_x,
        quan = quan,
        plot_fillers = plot_fillers,
        model_name = model_name,
        context_plot_type = context_plot_type,
        article_level_val = article_level_val,
        prefix_context_results = prefix_context_results
    ))
}