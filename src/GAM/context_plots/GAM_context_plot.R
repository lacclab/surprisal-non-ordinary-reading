# Functions for context plot generation and analysis

# get_context_plot_configs: Retrieves configuration settings for context plots
# get_context_dfs: Loads and processes context-related data frames
# plot_Context_dll: Generates Difference in Log-Likelihood (DLL) plots
# plot_Context_slowdown_RT_by_surprisal: Creates GAM plots for smoothed data

# These functions are used to analyze and visualize the effects of different context on dll and RT smooths.
# They handle various aspects of data processing, configuration, and plotting for different models and settings.

# An example of how to use these functions with the run_context_plots function is provided at the end of this file.


library(here)
library(ggplot2)
library(ggh4x)
#> Loading required package: ggplot2
library(scales)
source(here("src", "GAM", "GAM_analysis_funcs.R"))
source(here("src", "GAM", "plot_funcs", "GAM_plot_funcs.R"))
source(here("src", "GAM", "context_plots", "GAM_context_plot_config.R"))

add_context_type_col <- function(df, plot_fillers) {
    if (plot_fillers == TRUE){
        df <- df %>%
            mutate(
                context_type = case_when(
                    context == "P" ~ "standard_context",
                    context == "pastP-P" ~ "standard_context",
                    grepl("fill", context) ~ "dots_control_context",
                    grepl("text", context) ~ "text_control_context",
                    )
            )
        df$context_type = factor(df$context_type, levels=c(
            "standard_context", 
            "dots_control_context",
            "text_control_context"
            ))
    } else {
        df <- df %>%
            mutate(
                context_type = case_when(
                    context == "P" ~ "standard_context",
                    context == "pastP-P" ~ "standard_context",
                    grepl("inst", context) ~ "prompt_regime_context",
                    TRUE ~ "regime_context"
                    )
            )
        df$context_type = factor(df$context_type, levels=c(
            "standard_context", 
            "regime_context", 
            "prompt_regime_context"
            ))
    }
    return(df)
}

order_context_type <- function(df, plot_fillers) {
    context_order = c("standard_context", "dots_control_context", "text_control_context", "regime_context", "prompt_regime_context")
    df <- df[order(match(df$context_type, context_order)), ]

    if (plot_fillers == TRUE){
        df$context_type = factor(df$context_type, levels=c(
            "standard_context", 
            "dots_control_context",
            "text_control_context"
            ), ordered = TRUE)
    } else {
        df$context_type = factor(df$context_type, levels=c(
            "standard_context", 
            "regime_context", 
            "prompt_regime_context"
            ), ordered = TRUE)
    }
    return(df)
}

get_context_labeller <- function(plot_fillers) {
    if (plot_fillers == TRUE){
        context_labeller <- as_labeller(c(
            standard_context = "Standard Context",
            dots_control_context = "'.'  Control Context",
            text_control_context = "Text Control Context"
        ))
    } else {
        context_labeller <- as_labeller(c(
            standard_context = "Standard Context",
            regime_context = "Regime Context",
            prompt_regime_context = "Prompt + Regime Context"
        ))
    }
    return(context_labeller)
}

plot_Context_dll <- function(    
    all_dlls, 
    path, 
    x_condition_name,
    y_condition_name,
    x_condition_labels,
    y_condition_labels,
    plot_fillers,
    zoom_in_only_for_rr=FALSE
    )
    {
    all_dlls = order_context_type(all_dlls, plot_fillers)

    all_dlls <- add_symbol_by_p_val(all_dlls)
    all_dlls[[x_condition_name]] <- factor(all_dlls[[x_condition_name]], labels = x_condition_labels)
    all_dlls[[y_condition_name]] <- factor(all_dlls[[y_condition_name]], labels = y_condition_labels)
    # context labels
    context_labeller <- get_context_labeller(plot_fillers)
    
    # Calculate the jump size for y-axis ticks
    min_y <- min(all_dlls$lower)
    max_y <- max(all_dlls$upper)
    jump <- round((max_y - min_y) / 3,4)

    # Create breaks and labels
    breaks <- seq(0, max_y+jump, jump)
    labels <- sprintf("%0.4f", breaks)
    labels[1] <- "0"  # Set the label for 0 to "0.0000"

    y_max_plot = max(all_dlls$upper) + 1.2 * jump
    y_min_plot = min(all_dlls$lower)
    all_dlls <- all_dlls %>%
    mutate(
        vjust_value = -((3.8*5)/(y_max_plot-y_min_plot)*(upper-m)+0.8)
    )

    # Define the possible model names and their corresponding labels, colors, and shapes
    model_names_all <- c("linear", "nonlinear")
    labels_all <- c("Linear", "Non-linear")
    colors_all <- c("#005CAB", "#AF0038")
    shapes_all <- c(19, 17)

    # Get the distinct model names that actually appear in your data
    model_names_present <- all_dlls %>% distinct(linear) %>% pull(linear)

    # Filter the labels, colors, and shapes based on the present model names
    filtered_indices <- match(model_names_present, model_names_all)
    filtered_labels <- labels_all[filtered_indices]
    filtered_colors <- colors_all[filtered_indices]
    filtered_shapes <- shapes_all[filtered_indices]

    plot <- all_dlls %>%
        ggplot(aes(x = linear, y = m, color = linear, shape = linear)) +
        geom_point(position = position_dodge(width = 0.5)) +
        geom_errorbar(aes(ymin = lower, ymax = upper, width = 0.2), position = position_dodge(width = 0.5)) +
        geom_text(aes(label = p_val_symbol, vjust=vjust_value), size = 4.5, show.legend = FALSE) + # Add repelled labels for p val symbol
        ggh4x::facet_grid2(
                formula(paste(as.name(y_condition_name), "~", as.name(x_condition_name), "~ context_type")),
                labeller = labeller(context_type = context_labeller),
                # scales = "free",
                # independent = "all"
            ) +
        scale_color_manual(name = "Model Type", labels = filtered_labels, values = filtered_colors) +
        scale_shape_manual(name = "Model Type", labels = filtered_labels, values = filtered_shapes) +
        ylab("Delta Log Likelihood (per word)") +
        scale_y_continuous(labels = labels, breaks = breaks, minor_breaks = NULL) +
        expand_limits(y = 0) +
        expand_limits(y = y_max_plot) +
        theme(
            legend.position = "bottom",
            axis.title.x = element_blank(),
            axis.text.x = element_blank(),  # Remove x-axis text
            axis.ticks.x = element_blank(),  # Remove x-axis ticks
            text = element_text(family = "serif", size = 15),
            strip.text = element_text(size = 15)
        )

    # Save the plot
    ggsave(path, plot, device = "pdf", width = 9, height = 10)
}

## Plot Surprisal / RT Link
plot_Context_slowdown_RT_by_surprisal <- function(
    all_smooths,
    density_data, 
    path,
    x_condition_name,
    y_condition_name,
    x_condition_labels,
    y_condition_labels,
    RT_col,
    additive_model,
    plot_fillers,
    article_level_val,
    max_x=20,
    zoom_in_only_for_rr=FALSE
    ) {    
    all_smooths = order_context_type(all_smooths, plot_fillers)
    density_data = order_context_type(density_data, plot_fillers)

    # context labels
    context_labeller <- get_context_labeller(plot_fillers)

    if (max_x != 20){
        all_smooths = all_smooths %>% filter(surp<=max_x)
        density_data = density_data %>% filter(x<=max_x)
        all_smooths = all_smooths %>% filter(context_type=="regime_context" & reread_condition=="Repeated Reading")
        density_data = density_data %>% filter(context_type=="regime_context" & reread_condition=="Repeated Reading")
    } else{
        # cut_y
        y_cut = -40
        all_smooths <- all_smooths %>% mutate(y_lower = ifelse(y_lower < y_cut, y_cut, y_lower))
    }
    
    # filter only linear and nonlinear smooths
    linear_smooths <- all_smooths %>% filter(linear == "linear")
    nonlinear_smooths <- all_smooths %>% filter(linear == "nonlinear")

    calculate_density_range <- function(df_linear, df_nonlinear) {
        max_y <- max(max(df_linear$y_upper), max(df_nonlinear$y_upper))
        min_y_lower <- min(min(df_linear$y_lower), min(df_nonlinear$y_lower))
        
        jump <- (max_y - min_y_lower) * 0.3
        floor_y_min <- floor(min_y_lower / jump) * jump - 0.5 * jump
        y_min_dens_plot <- min(-jump, floor_y_min)
        y_max_dens_plot <- y_min_dens_plot + jump * 0.7
        
        if (max_x != 20) {
            y_min_dens_plot <- y_min_dens_plot / 3
            y_max_dens_plot <- y_max_dens_plot / 3
        }
        
        # cat("Calc| ", y_min_dens_plot, "\n")
        return(list(y_min_dens_plot = y_min_dens_plot, y_max_dens_plot = y_max_dens_plot))
    }

    # default limits for density data plot
    density_data <- density_data %>%
        mutate(
            y_min_dens_plot = -20,
            y_max_dens_plot = -8
        )
    
    # filter data by valid combinations
    valid_combinations <- read.csv(file = here("src", "GAM", "context_plots", "GAM_context_plot_valid_combinations.csv"), 
        sep = ",", 
        encoding = "UTF-8", 
        stringsAsFactors = FALSE)
    valid_rows <- valid_combinations %>% filter(article_level == article_level_val)

    # calc y_min_dens_plot and y_max_dens_plot
    for (i in seq_len(nrow(valid_rows))) {
        context_val <- valid_rows$context[i]
        context_type_val <- valid_rows$context_type[i]
        hp_val <- valid_rows$has_preview_condition[i]
        reread_val <- valid_rows$reread_condition[i]

        df_linear <- linear_smooths %>%
            filter(context == context_val, context_type == context_type_val, has_preview_condition == hp_val, reread_condition == reread_val)
        df_nonlinear <- nonlinear_smooths %>%
            filter(context == context_val, context_type == context_type_val, has_preview_condition == hp_val, reread_condition == reread_val)
        if (nrow(df_linear) > 0 & nrow(df_nonlinear) > 0) {
            ranges <- calculate_density_range(df_linear, df_nonlinear)
            density_data <- density_data %>%
                mutate(
                    y_min_dens_plot = ifelse(
                        (context == context_val) & (context_type == context_type_val) & (has_preview_condition == hp_val) & (reread_condition == reread_val),
                        ranges$y_min_dens_plot,
                        y_min_dens_plot
                    ),
                    y_max_dens_plot = ifelse(
                        (context == context_val) & (context_type == context_type_val) & (has_preview_condition == hp_val) & (reread_condition == reread_val),
                        ranges$y_max_dens_plot,
                        y_max_dens_plot
                    )
                )
        }
    }

    # sacle function
    scale_range <- function(y, a, b) {
        y_min <- 0
        y_max <- max(y)
        scaled_y <- a + (y - y_min) * (b - a) / (y_max - y_min)
        return(scaled_y)
    }

    # calc scaled_y using y_min_dens_plot and y_max_dens_plot
    density_data <- density_data %>%
    group_by(filter_type, context, context_type, !!sym(x_condition_name), !!sym(y_condition_name)) %>%
    mutate(scaled_y = scale_range(y, a = y_min_dens_plot, b = y_max_dens_plot)) %>%
    ungroup()

    # calc max_x density data
    density_data <- density_data %>%
    group_by(filter_type, context, context_type, !!sym(x_condition_name), !!sym(y_condition_name)) %>%
    mutate(max_x = max(x)) %>%
    ungroup()

    # calc max_x max_y smooths
    nonlinear_smooths <- nonlinear_smooths %>%
        group_by(filter_type, context, context_type, !!sym(x_condition_name), !!sym(y_condition_name)) %>%
        mutate(max_y_smooths = max(y_upper), max_x_smooths = max(surp)) %>%
        ungroup()

    # Filter density data by max_x_smooths 
    # (Filter Only After Scale and Not Before!)
    density_data <- density_data %>%
        left_join(nonlinear_smooths %>%
                    select(filter_type, context, context_type, !!sym(x_condition_name), !!sym(y_condition_name), max_x_smooths) %>% distinct() ,
                    by = c("filter_type", "context", "context_type", x_condition_name, y_condition_name)
                ) %>%
        filter(x <= max_x_smooths)
    
    # Set labels for x_condition_name, y_condition_name
    linear_smooths[[x_condition_name]] <- factor(linear_smooths[[x_condition_name]], labels = x_condition_labels)
    linear_smooths[[y_condition_name]] <- factor(linear_smooths[[y_condition_name]], labels = y_condition_labels)
    nonlinear_smooths[[x_condition_name]] <- factor(nonlinear_smooths[[x_condition_name]], labels = x_condition_labels)
    nonlinear_smooths[[y_condition_name]] <- factor(nonlinear_smooths[[y_condition_name]], labels = y_condition_labels)
    density_data[[x_condition_name]] <- factor(density_data[[x_condition_name]], labels = x_condition_labels)
    density_data[[y_condition_name]] <- factor(density_data[[y_condition_name]], labels = y_condition_labels)

    plot = ggplot() +
        geom_rect(data=density_data, aes(
            xmin=0, xmax=max_x_smooths, 
            ymin=y_min_dens_plot, ymax=y_max_dens_plot),
            fill="#f4f4f4", color="grey", alpha=1, size=0.2) +
        geom_line(data=linear_smooths, aes(x=surp, y=y, color=linear), size=0.5) +
        geom_line(data=nonlinear_smooths, aes(x=surp, y=y, color=linear), size=0.5) +
        geom_ribbon(data=linear_smooths, aes(x=surp, ymin=y_lower, ymax=y_upper, fill=linear), alpha=0.2, size=0.5) +
        geom_ribbon(data=nonlinear_smooths, aes(x=surp, ymin=y_lower, ymax=y_upper, fill=linear), alpha=0.2, size=0.5) +
        ggh4x::facet_grid2(
            formula(paste(as.name(y_condition_name), "~", as.name(x_condition_name), "~ context_type")),
            labeller = labeller(context_type = context_labeller),
            scales = "free",
            independent = "all"
        ) +
        ylab("Slowdown (ms)") +
        xlab("Surprisal (bits)") +
        scale_color_manual(name="Model Type", labels=c("Linear", "Non-linear"), values=c("#005CAB", "#AF0038")) +
        scale_fill_manual(name="Model Type", labels=c("Linear", "Non-linear"), values=c("#005CAB", "#AF0038")) +
        scale_linetype_manual(values=c("a", "b")) +
        geom_line(data=density_data, aes(x=x, y=scaled_y), color="#8E8991", size=0.8) +
        geom_point(data=nonlinear_smooths, aes(x=max_x_smooths, y=max_y_smooths*1.2), color="white") +
        theme(
            text=element_text(family="serif", size=15),
            legend.position="bottom",
            panel.grid.minor=element_blank(),
            strip.text=element_text(size=15),
            axis.title.y=element_text(margin=margin(t=0, r=10, b=0, l=0))
        )
    
    if (max_x == 20 & zoom_in_only_for_rr==FALSE){
        x_labels = c(0, 10, 20)
    }

    ggsave(path, device="pdf", width = 9, height = 10)
}

filter_df_by_context <- function(df, plot_fillers, filter_type="exclude_RegimeRepeatedReading") {
    valid_combinations <- read.csv(file = here("src", "GAM", "context_plots", "GAM_context_plot_valid_combinations.csv"), 
        sep = ",", 
        encoding = "UTF-8", 
        stringsAsFactors = FALSE)
    filter_expression <- as.symbol(filter_type)
    df$filter_type = filter_type
    if (plot_fillers == TRUE){
        df$filler_plot = 1
        df <- df %>% 
            inner_join(valid_combinations, by = c("has_preview_condition", "reread_condition", "context", "filler_plot", "article_level"),
            relationship="many-to-many") %>%
            filter(!!filter_expression == 1)
    } else {
        df$regime_plot = 1
        df <- df %>% 
            inner_join(valid_combinations, by = c("has_preview_condition", "reread_condition", "context", "regime_plot", "article_level"),
            relationship="many-to-many") %>%
            filter(!!filter_expression == 1)
    }
    return(df)
}

iterate_over_results_dir <- function(
    configs,
    results_path,
    surp_list,
    context_list,
    filter_type
) {
    RE = configs$RE
    use_CV = configs$use_CV
    additive_model = configs$additive_model
    RT_col = configs$RT_col
    zoom_in_only_for_rr = configs$zoom_in_only_for_rr
    plot_fillers = configs$plot_fillers
    article_level_val = configs$article_level_val

    n = length(context_list)

    cat(results_path, "\n")

    all_dlls = data.frame()
    all_densities = data.frame()
    all_smooths = data.frame()
    all_dll_p_vals = data.frame()
    all_smooths_p_vals = data.frame()
    for (i in c(1:n)){
        surp_column = surp_list[i]
        context_name = context_list[i]
        print(context_name)
        prefix_comp_results = paste0(RT_col, " - ", surp_column, "_re=", RE, "_useCV=", use_CV, "_")
        prefix_smooths_results = paste0(RT_col, " - ", surp_column, "_re=", RE, "_additive=", additive_model, "_")
        density_path = here(results_path, paste0(RT_col, " - ", surp_column, "_density_data.csv"))
        
        # load csv
        density_data = read.csv(density_path) %>% mutate(context = context_name, article_level = article_level_val)
        linear_comp_df = read.csv(here(results_path, paste0(prefix_comp_results, "linear_comp_df.csv"))) %>% mutate(context = context_name, article_level = article_level_val)
        dll_p_vals = read.csv(here(results_path, paste0(prefix_comp_results, "dll_p_vals_df.csv"))) %>% mutate(context = context_name, article_level = article_level_val)
        # rbind
        all_densities = rbind(all_densities, density_data)
        all_dlls = rbind(all_dlls, linear_comp_df)
        all_dll_p_vals = rbind(all_dll_p_vals, dll_p_vals)
        
        if (plot_smooths==TRUE){
            linear_smooths = read.csv(here(results_path, paste0(prefix_smooths_results, "linear_smooths.csv"))) %>% mutate(context = context_name, article_level = article_level_val)
            nonlinear_smooths = read.csv(here(results_path, paste0(prefix_smooths_results, "nonlinear_smooths.csv"))) %>% mutate(context = context_name, article_level = article_level_val)
            smooths_p_vals = read.csv(here(results_path, paste0(prefix_smooths_results, "smooths_p_vals.csv"))) %>% mutate(context = context_name, article_level = article_level_val)
            # rbind
            all_smooths = rbind(all_smooths, linear_smooths)
            all_smooths = rbind(all_smooths, nonlinear_smooths)
            if ("converged" %in% colnames(smooths_p_vals)) {smooths_p_vals <- smooths_p_vals[, !names(smooths_p_vals) %in% "converged"]}
            all_smooths_p_vals = rbind(all_smooths_p_vals, smooths_p_vals)
        }
    }

    # filter_df_by_context
    all_dlls <- filter_df_by_context(all_dlls, filter_type=filter_type, plot_fillers=plot_fillers)
    all_densities <- filter_df_by_context(all_densities, filter_type=filter_type, plot_fillers=plot_fillers)
    all_dll_p_vals <- filter_df_by_context(all_dll_p_vals, filter_type=filter_type, plot_fillers=plot_fillers)
    if (plot_smooths==TRUE){
        all_smooths <- filter_df_by_context(all_smooths, filter_type=filter_type, plot_fillers=plot_fillers)
        all_smooths_p_vals <- filter_df_by_context(all_smooths_p_vals, filter_type=filter_type, plot_fillers=plot_fillers)
    }
    
    return(list(
        all_dlls=all_dlls,
        all_smooths=all_smooths,
        all_dll_p_vals=all_dll_p_vals,
        all_smooths_p_vals=all_smooths_p_vals,
        all_densities=all_densities
        ))
}

get_context_dfs <- function(
    configs
) {
    if (zoom_in_only_for_rr==TRUE){
        filter_type="exclude_RegimeRepeatedReading"
    } else {filter_type="Regular"}

    # regular iteration
    context_dfs = iterate_over_results_dir(
        configs=configs,
        results_path=results_path,
        surp_list=surp_list,
        context_list=context_list,
        filter_type=filter_type
    )
    # unpack
    all_dlls = context_dfs$all_dlls
    all_smooths = context_dfs$all_smooths
    all_dll_p_vals = context_dfs$all_dll_p_vals
    all_smooths_p_vals = context_dfs$all_smooths_p_vals
    all_densities = context_dfs$all_densities
    
    prefix_context_results = configs$prefix_context_results
    
    if (zoom_in_only_for_rr==TRUE){
        # Regime Repeated Reading iteration
        rr_context_dfs = iterate_over_results_dir(
            configs=configs,
            results_path=results_path_zoom_in,
            surp_list=rr_surp_cols,
            context_list=rr_context_cols,
            filter_type="only_RegimeRepeatedReading"
        )

        # unpack
        rr_all_dlls = rr_context_dfs$all_dlls
        rr_all_smooths = rr_context_dfs$all_smooths
        rr_all_dll_p_vals = rr_context_dfs$all_dll_p_vals
        rr_all_smooths_p_vals = rr_context_dfs$all_smooths_p_vals
        rr_all_densities = rr_context_dfs$all_densities

        # union dfs
        all_dlls = rbind(all_dlls, rr_all_dlls)
        all_smooths = rbind(all_smooths, rr_all_smooths)
        all_dll_p_vals = rbind(all_dll_p_vals, rr_all_dll_p_vals)
        all_smooths_p_vals = rbind(all_smooths_p_vals, rr_all_smooths_p_vals)
        all_densities = rbind(all_densities, rr_all_densities)
    }    
    
    # save results
    write.csv(all_dll_p_vals, here(results_path, paste0(prefix_context_results, "dll_p_vals_df.csv")))
    write.csv(all_smooths_p_vals, here(results_path, paste0(prefix_context_results, "smooths_p_vals.csv")))

    return(list(
        all_dlls=all_dlls,
        all_smooths=all_smooths,
        all_densities=all_densities)
    )
}

run_context_plots <- function(
    context_plot_type, 
    model_name, 
    result_dir, 
    file_name, 
    plot_fillers,
    article_level_val
    ) {
    
    configs = get_context_plot_configs(    
        context_plot_type,
        model_name,
        result_dir,
        file_name,
        plot_fillers,
        article_level_val
    )
    # Unpack the configuration into the current environment
    list2env(configs, .GlobalEnv)

    context_dfs = get_context_dfs(configs)
    # dll
    all_dlls = context_dfs$all_dlls
    # density
    all_densities = context_dfs$all_densities

    # Plot context dll
    plot_Context_dll(all_dlls, dll_path, 
        x_condition_name, y_condition_name,
        x_condition_labels, y_condition_labels,
        plot_fillers, zoom_in_only_for_rr)
    
    # Plot GAM
    if (plot_smooths==TRUE){
        # smooths
        all_smooths = context_dfs$all_smooths

        plot_Context_slowdown_RT_by_surprisal(all_smooths, all_densities, surp_link_path,
            x_condition_name, y_condition_name,
            x_condition_labels, y_condition_labels,
            RT_col, additive_model, plot_fillers, article_level_val,
            max_x, zoom_in_only_for_rr)
    }
}

# Run
# Description of the run_context_plots function and its usage

# The run_context_plots function is a comprehensive tool for generating context-related plots
# for different models and configurations. It performs the following main tasks:

# 1. Retrieves configuration settings based on input parameters
# 2. Loads and processes context-related data frames
# 3. Generates Difference in Log-Likelihood (DLL) plots
# 4. Optionally creates GAM (Generalized Additive Model) plots for smoothed data

# The function takes several parameters:
# - context_plot_type: Determines the type of context plot to generate - use "zoom_in_only_for_rr" for results like in the paper
# - model_name: Specifies the name of the model being analyzed
# - result_dir: Directory where results of model are saved
# - file_name: Name of the input file containing the data
# - plot_fillers: Boolean flag to include or exclude filler plots
# - article_level_val: Specifies the level of analysis (e.g., "paragraph" or "article")

# The subsequent code demonstrates how to use this function for different models and configurations,
# allowing for comprehensive analysis of context effects on reading times and surprisal.

preprocess_type = "results 0<RT<3000 firstpassNA"
file_name = "et_20240505_with_all_surp20240624"
plot_smooths = TRUE
article_level_val = "paragraph"
models_list = c( # nlp 11
    'EleutherAI-pythia-70m'
) 

for (model_name in models_list){
    cat("Context Plot| ", model_name, " --------- \n")
    run_context_plots(
        context_plot_type = "zoom_in_only_for_rr",
        model_name = model_name,
        result_dir = preprocess_type,
        file_name = file_name,
        plot_fillers = FALSE,
        article_level_val = article_level_val
    )
    run_context_plots(
        context_plot_type = "zoom_in_only_for_rr",
        model_name = model_name,
        result_dir = preprocess_type,
        file_name = file_name,
        plot_fillers = TRUE,
        article_level_val = article_level_val
    )
}