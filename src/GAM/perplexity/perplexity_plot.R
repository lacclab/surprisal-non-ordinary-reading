library(ggrepel)
library(here)
source(here("src", "GAM", "GAM_analysis_funcs.R"))

plot_dll_by_perplexity <- function(    
    dll_dfs, 
    path_plot_by_family,
    path_plot, 
    path_2,
    linear_val,
    x_condition_name,
    y_condition_name,
    x_condition_labels,
    y_condition_labels)
    {
    dll_dfs = dll_dfs %>% filter(linear == linear_val)
    cat("dll_dfs size: ", dim(dll_dfs)[0], "\n")
    dll_dfs[[x_condition_name]] <- factor(dll_dfs[[x_condition_name]], labels = x_condition_labels)
    dll_dfs[[y_condition_name]] <- factor(dll_dfs[[y_condition_name]], labels = y_condition_labels)
    
    # y_cut = 0
    # dll_dfs <- dll_dfs %>% mutate(lower = ifelse(lower < y_cut, y_cut, lower))

    # Calculate the jump size for y-axis ticks
    min_y <- min(dll_dfs$m)
    max_y <- max(dll_dfs$m)
    jump <- (max_y - min_y) / 3

    # Create breaks and labels
    breaks <- seq(0, max_y+2*jump, jump)
    labels <- sprintf("%0.4f", breaks)
    labels[1] <- "0"  # Set the label for 0 to "0.0000"

    plot <- dll_dfs %>%
    ggplot(aes(x = log(ppl), y = m, color = model_size_milions)) +
    scale_color_gradient2(low = "#E74C3C", mid = "#D4AC0D", high = "#1E8449", midpoint = 3000) +
    geom_point(position = position_dodge(width = ), size=3) +
    geom_errorbar(aes(ymin = lower, ymax = upper, width = 0.1), position = position_dodge(width = 0.5)) +
    geom_text(aes(label = model_size),angle = 90, vjust = 0.4, hjust = -1, size=4, color="black") + # Add labels
    # geom_text_repel(aes(label = model_size), size = 4, color = "black") + # Add repelled labels for model_size) +
    # geom_line(data=dll_dfs, aes(x=ppl, y=m, color="black"), size=0.6) +
    facet_grid(
        formula(paste(as.name(y_condition_name), "~", as.name(x_condition_name), "~model_family"))) +
    # scale_color_manual(name = "Model Type", labels = c("Linear", "Non-linear"), values = c("#005CAB", "#AF0038")) +
    # scale_shape_manual(name = "Model Type", labels = c("Linear", "Non-linear"), values = c(19, 17)) +
    labs(
        x = "Log Perplexity",
        y = "Delta Log Likelihood (per word)",
        color = "Model Size (M)", # Change legend title for color
        shape = "Model Family") +
    scale_y_continuous(breaks = breaks, labels=labels, limits = c(-0.0005, 0.0025)) +
    coord_cartesian(clip = "off") +
    expand_limits(y = 0) +
    theme(
        # legend.position = "bottom",
        text = element_text(family = "serif", size = 22),
        strip.text = element_text(size = 22),
    )

    # Save the plot
    ggsave(path_plot_by_family, plot, device = "pdf", width = 20, height = 12)

    shapes <- c("Gemma" = 16, "GPT-2" = 17, "GPT-J" = 15, "GPT-Neo" = 18, "Llama-2" = 8, "Mistral" = 4, "OPT" = 5, "Pythia" = 6)
    
    plot <- dll_dfs %>%
    ggplot(aes(x = log(ppl), y = m, color = model_size_milions, shape = model_family)) +
    scale_color_gradient2(low = "#E74C3C", mid = "#D4AC0D", high = "#1E8449", midpoint = 3000) +
    scale_shape_manual(values = shapes) +
    geom_point(position = position_dodge(width = ), size=4, stroke=1.3) +
    # geom_errorbar(aes(ymin = lower, ymax = upper, width = 0.4), position = position_dodge(width = 0.5)) +
    geom_text_repel(aes(label = model_size), size = 5, color = "black") + # Add repelled labels for model_size) + # Add labels
    # geom_text(aes(label = model_size),angle = 90, vjust = 0.4, hjust = -1, size=5, color="black") + # Add labels
    # geom_text_repel(aes(label = model_family), size = 3, color = "black", nudge_y = -0.0005) + # Add repelled labels for model_size) + # Add labels
    # geom_line(data=dll_dfs, aes(x=ppl, y=m, color="black"), size=0.6) +
    facet_grid(
        formula(paste(as.name(y_condition_name), "~", as.name(x_condition_name)))) +
    # scale_color_manual(name = "Model Type", labels = c("Linear", "Non-linear"), values = c("#005CAB", "#AF0038")) +
    # scale_shape_manual(name = "Model Type", labels = c("Linear", "Non-linear"), values = c(19, 17)) +
    labs(
        x = "Log Perplexity",
        y = "Delta Log Likelihood (per word)",
        color = "Model Size (M)", # Change legend title for color
        shape = "Model Family") +
    scale_y_continuous(breaks = breaks, labels=labels, limits = c(-0.0002, 0.0024)) +
    coord_cartesian(clip = "off") +
    expand_limits(y = 0) +
    theme(
        # legend.position = "bottom",
        text = element_text(family = "serif", size = 22),
        strip.text = element_text(size = 22),
    )

    # Save the plot
    ggsave(path_plot, plot, device = "pdf", width = 20, height = 12)
    
    shapes <- c("Gemma" = 1, "GPT-2" = 1, "GPT-J" = 1, "GPT-Neo" = 1, "Llama-2" = 1, "Mistral" = 1, "OPT" = 1, "Pythia" = 1)
    colors <- c("Gemma" = "#E74C3C",   # Bright red
                "GPT-2" = "#F4D03F",   # Bright yellow
                "GPT-J" = "#2749A8",   # Dark Blue
                "GPT-Neo" = "#9422B8", # Purple
                "Llama-2" = "#3498DB", # Medium blue
                "Mistral" = "#95600F", # Brown
                "OPT" = "#E247A0",     # Pink
                "Pythia" = "#239526")  # Green
    
    plot <- dll_dfs %>%
    ggplot(aes(x = log(ppl), y = m, color = model_family, shape = model_family)) + # Added shape to differentiate the model families
    scale_color_manual(values = colors) + # Custom colors for each model family
    scale_shape_manual(values = shapes) +
    geom_point(size = 4, stroke = 1.3) + # Removed shape aesthetic
    geom_text_repel(aes(label = model_size), size = 5, color = "black") +
    facet_grid(
        formula(paste(as.name(y_condition_name), "~", as.name(x_condition_name)))) +
    labs(
        x = "Log Perplexity",
        y = "Delta Log Likelihood (per word)",
        color = "Model Family") + # Changed legend title for color
    scale_y_continuous(breaks = breaks, labels = labels, limits = c(-0.0002, 0.0024)) +
    coord_cartesian(clip = "off") +
    expand_limits(y = 0) +
    guides(shape = FALSE) +
    theme(
        text = element_text(family = "serif", size = 22),
        strip.text = element_text(size = 22)
    )

    # Save the plot
    ggsave(path_2, plot, device = "pdf", width = 20, height = 12)
    print("end")
}


get_dll_dfs <- function(
    results_path,
    RE,
    use_CV,
    additive_model,
    psychometric,
    ppl_df
) {
    ppl_df = ppl_df %>% filter(sentence_level_ppl>0)
    n = length(ppl_df$surp_col)
    cat(n, "models", "\n")

    all_dlls = data.frame()
    all_dll_p_vals = data.frame()
    for (i in c(1:n)){
        surp_column = ppl_df$surp_col[i]
        model_id = ppl_df$model_id[i]
        ppl = ppl_df$sentence_level_ppl[i]
        model_family = ppl_df$model_family[i]
        model_size = ppl_df$model_size[i]
        model_size_milions = ppl_df$model_size_milions[i]
        model_name_with_size = ppl_df$model_name_with_size[i]
        cat(model_id, "\n")
        prefix_comp_results = paste0(psychometric, " - ", surp_column, "_re=", RE, "_useCV=", use_CV, "_")
        # load csv
        linear_comp_df_path <- here(results_path, paste0(prefix_comp_results, "linear_comp_df.csv"))
        dll_p_vals_path <- here(results_path, paste0(prefix_comp_results, "dll_p_vals_df.csv"))

        # Check if the file exists and read it if it does, otherwise print a message
        if (file.exists(linear_comp_df_path)) {
            linear_comp_df <- read.csv(linear_comp_df_path) %>% 
                mutate(
                surp_column = surp_column,
                model_id = model_id,
                model_family = model_family,
                model_size = model_size,
                model_size_milions = model_size_milions,
                model_name_with_size = model_name_with_size,
                ppl = ppl
                )
            all_dlls = rbind(all_dlls, linear_comp_df)
        } else {
        print(paste("File does not exist:", linear_comp_df_path))
        }

        if (file.exists(dll_p_vals_path)) {
            dll_p_vals <- read.csv(dll_p_vals_path) %>% 
                mutate(
                surp_column = surp_column,
                model_id = model_id,
                model_family = model_family,
                model_size = model_size,
                model_size_milions = model_size_milions,
                model_name_with_size = model_name_with_size,
                ppl = ppl
                )
            all_dll_p_vals = rbind(all_dll_p_vals, dll_p_vals)
        } else {
        print(paste("File does not exist:", dll_p_vals_path))
        }
    }

    prefix_context_results = paste0("Perplexity-", psychometric, "-")
    write.csv(all_dll_p_vals, here(results_path, paste0(prefix_context_results, "dll_p_vals_df.csv")))

    return(list(
        all_dlls=all_dlls,
        all_dll_p_vals=all_dll_p_vals)
    )
}

# params
result_dir = "results 0<RT<3000 firstpassNA"
results_path = here("src", "GAM", result_dir, "results_different_surp_estimates", "et_20240505_with_all_surp20240624")
ppl_df = read.csv(here("src", "GAM", "perplexity", "models_names_with_ppl.csv"))

RE = F
use_CV = T
additive_model = T
psychometric = "FirstPassGD"

x_condition_name="has_preview_condition"
x_condition_labels <- c("Ordinary Reading", "Information Seeking")
x_condition_vals=c("Gathering", "Hunting")
y_condition_name="reread_condition"
y_condition_labels <- c("First Reading", "Repeated Reading")
y_condition_vals=c(0, 1)

all_data = get_dll_dfs(
    results_path=results_path,
    RE=RE,
    use_CV=use_CV,
    additive_model=additive_model,
    psychometric=psychometric,
    ppl_df=ppl_df
)
all_dlls = all_data$all_dlls

models_names <- all_dlls %>% distinct(linear) %>% pull(linear)

# Plot context dll
for (linear in models_names){
    plot_dll_by_perplexity(
        dll_dfs=all_dlls, 
        path_plot_by_family=here(results_path, paste0("Perplexity-", psychometric, "-", linear, "-dll_plot-by_family.pdf")),
        path_plot=here(results_path, paste0("Perplexity-", psychometric, "-", linear, "-dll_plot.pdf")),
        path_2=here(results_path, paste0("Perplexity-", psychometric, "-", linear, "-dll_plot-2.pdf")),
        linear_val=linear,
        x_condition_name=x_condition_name,
        y_condition_name=y_condition_name,
        x_condition_labels=x_condition_labels,
        y_condition_labels=y_condition_labels)
}
