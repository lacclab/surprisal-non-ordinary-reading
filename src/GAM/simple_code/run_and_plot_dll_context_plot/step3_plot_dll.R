library(here)
library(ggplot2)
library(ggh4x) #> Loading required package: ggplot2
library(scales)

order_context_type <- function(df) {
    # TODO optional: adapt to your context columns
    context_order = c("standard_context", "dots_control_context", "text_control_context", "regime_context", "prompt_regime_context")
    df <- df[order(match(df$context_type, context_order)), ]
    df$context_type = factor(df$context_type, levels=c(
        "standard_context", 
        "regime_context", 
        "prompt_regime_context"
        ), ordered = TRUE)
    return(df)
}

get_context_labeller <- function(plot_fillers) {
    # TODO: adapt to your context columns
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
    save_to_path, 
    x_condition_name,
    y_condition_name,
    x_condition_labels,
    y_condition_labels,
    )
    {
    # all_dlls = order_context_type(all_dlls, plot_fillers)

    all_dlls <- add_symbol_by_p_val(all_dlls)
    all_dlls[[x_condition_name]] <- factor(all_dlls[[x_condition_name]], labels = x_condition_labels)
    all_dlls[[y_condition_name]] <- factor(all_dlls[[y_condition_name]], labels = y_condition_labels)
    # context labels
    context_labeller <- get_context_labeller(plot_fillers)
    
    # defenitions of plot size - so the plot will not cut the *** of the dlls
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

    # Define colors and shpaes for each model
    # Define the possible model names and their corresponding labels, colors, and shapes
    model_names_all <- c("linear", "nonlinear")
    labels_all <- c("Linear", "Non-linear")
    colors_all <- c("#005CAB", "#AF0038", "#9b59b6", "#8a560f")
    shapes_all <- c(19, 17, 15, 13)
    # Get the distinct model names that actually appear in your data
    model_names_present <- all_dlls %>% distinct(linear) %>% pull(linear)
    # Filter the labels, colors, and shapes based on the present model names
    filtered_indices <- match(model_names_present, model_names_all)
    filtered_labels <- labels_all[filtered_indices]
    filtered_colors <- colors_all[filtered_indices]
    filtered_shapes <- shapes_all[filtered_indices]

    # The actuacl Plot
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
    ggsave(save_to_path, plot, device = "pdf", width = 9, height = 10)
}