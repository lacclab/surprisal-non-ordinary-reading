# install.packages("gratia")
# install.packages("scales")
library(scales)

# Define the function to map p-values to symbols
p2stars <- function(p) {
    if (p < 0) {
        return("()")
    } else if (p <= 0.001) {
        return("***")
    } else if (p <= 0.01) {
        return("**")
    } else if (p <= 0.05) {
        return("*")
    } else {
        return("(.)")
    }
}

# Apply the function to create the 'symbol' column
add_symbol_by_p_val <- function(linear_comp_df) {
    linear_comp_df$p_val_symbol <- sapply(linear_comp_df$new_p_val, p2stars)
    return(linear_comp_df)
}

## Plot Surprisal / RT Link
plot_smooths_RT_by_surprisal <- function(
    linear_smooths, 
    nonlinear_smooths, 
    density_data, 
    path,
    x_condition_name,
    y_condition_name,
    x_condition_labels,
    y_condition_labels,
    RT_col,
    additive_model
    ) {    
    # make sure to get the labels
    # y_condition
    density_data[[x_condition_name]] <- factor(density_data[[x_condition_name]], labels = x_condition_labels)
    linear_smooths[[x_condition_name]] <- factor(linear_smooths[[x_condition_name]], labels = x_condition_labels)
    nonlinear_smooths[[x_condition_name]] <- factor(nonlinear_smooths[[x_condition_name]], labels = x_condition_labels)
    # y_condition
    density_data[[y_condition_name]] <- factor(density_data[[y_condition_name]], labels = y_condition_labels)
    linear_smooths[[y_condition_name]] <- factor(linear_smooths[[y_condition_name]], labels = y_condition_labels)
    nonlinear_smooths[[y_condition_name]] <- factor(nonlinear_smooths[[y_condition_name]], labels = y_condition_labels)
    max_y <- max(max(linear_smooths$y_upper), max(nonlinear_smooths$y_upper))
    min_y_lower <- min(min(linear_smooths$y_lower), min(nonlinear_smooths$y_lower))
    max_y_d = max(density_data$y)

    get_jump <- function(min_y_lower, max_y) {
        start <- ceiling(min_y_lower / 20) * 20
        end <- max_y + 20
        jump <- ifelse(length(seq(start, end, by = 20)) > 8, 40, 20)
        return(jump)
    }
    create_labels <- function(min_y_lower, max_y, jump) {
        start <- ceiling(min_y_lower / jump) * jump
        end <- max_y + jump
        labels <- seq(start, end, by = jump)
        labels <- c(labels, 0)
        return(labels)
    }
    jump = get_jump(min_y_lower, max_y)
    y_labels = create_labels(min_y_lower, max_y, jump)
    floor_y_min = floor(min_y_lower / jump) * jump
    y_min_dens_plot = min(-jump,floor_y_min)
    y_max_dens_plot = y_min_dens_plot+jump*0.7

    scale_range <- function(y, a, b) {
        y_min <- min(y)
        y_max <- max(y)
        scaled_y <- a + (y - y_min) * (b - a) / (y_max - y_min)
        return(scaled_y)
    }

    ggplot() +
        annotate("rect", 
            xmin=0, xmax=20, 
            ymin=y_min_dens_plot, ymax=y_max_dens_plot, 
            fill="#f4f4f4", color="grey", alpha=1, size=0.2) +
        geom_line(data=density_data, aes(x=x, y=scale_range(y, a=y_min_dens_plot, b=y_max_dens_plot)), color="#8E8991", size=0.5) +
        geom_line(data=linear_smooths, aes(x=surp, y=y, color=linear), size=0.5) +
        geom_line(data=nonlinear_smooths, aes(x=surp, y=y, color=linear), size=0.5) +
        geom_ribbon(data=linear_smooths, aes(x=surp, ymin=y_lower, ymax=y_upper, fill=linear), alpha=0.2, size=0.5) +
        geom_ribbon(data=nonlinear_smooths, aes(x=surp, ymin=y_lower, ymax=y_upper, fill=linear), alpha=0.2, size=0.5) +
        scale_x_continuous(labels=c(0, 10, 20), breaks=c(0, 10, 20), minor_breaks=NULL) +
        scale_y_continuous(labels=y_labels, breaks=y_labels, minor_breaks=NULL) +
        facet_grid(formula(paste(as.name(y_condition_name), "~", as.name(x_condition_name)))) +
        ylab("Slowdown (ms)") +
        xlab("Surprisal (bits)") +
        scale_color_manual(name="Model Type", labels=c("Linear", "Non-linear"), values=c("#005CAB", "#AF0038")) +
        scale_fill_manual(name="Model Type", labels=c("Linear", "Non-linear"), values=c("#005CAB", "#AF0038")) +
        scale_linetype_manual(values=c("a", "b")) +
        theme(
            text=element_text(family="serif", size=16),
            legend.position="bottom",
            panel.grid.minor=element_blank(),
            strip.text=element_text(size=16),
            axis.title.y=element_text(margin=margin(t=0, r=10, b=0, l=0))
        )

    ggsave(path, device="pdf", height=6, width=6)
}

plot_dll_by_model <- function(
    linear_comp_df, 
    path, 
    x_condition_name,
    y_condition_name,
    x_condition_labels,
    y_condition_labels,
    RT_col,
    use_CV
    ) {
    linear_comp_df <- add_symbol_by_p_val(linear_comp_df)
    linear_comp_df[[x_condition_name]] <- factor(linear_comp_df[[x_condition_name]], labels = x_condition_labels)
    linear_comp_df[[y_condition_name]] <- factor(linear_comp_df[[y_condition_name]], labels = y_condition_labels)
    # linear_comp_df <- linear_comp_df %>% filter(linear == "linear" | linear == "nonlinear")
    
    # Calculate the jump size for y-axis ticks
    min_y <- min(linear_comp_df$lower)
    max_y <- max(linear_comp_df$upper)
    jump <- round((max_y - min_y) / 3,4)

    # Create breaks and labels
    breaks <- seq(0, max_y+jump, jump)
    labels <- sprintf("%0.4f", breaks)
    labels[1] <- "0"  # Set the label for 0 to "0.0000"

    y_max_plot = max(linear_comp_df$upper) + 1.2 * jump
    linear_comp_df <- linear_comp_df %>%
    mutate(
        vjust_value = -((3.8*5)/y_max_plot*(upper-m)+0.8) # Calculate the vjust value based on m
    )

    # Define the possible model names and their corresponding labels, colors, and shapes
    model_names_all <- c("linear", "nonlinear")
    labels_all <- c("Linear", "Non-linear")
    colors_all <- c("#005CAB", "#AF0038")
    shapes_all <- c(19, 17)

    # Get the distinct model names that actually appear in your data
    model_names_present <- linear_comp_df %>% distinct(linear) %>% pull(linear)

    # Filter the labels, colors, and shapes based on the present model names
    filtered_indices <- match(model_names_present, model_names_all)
    filtered_labels <- labels_all[filtered_indices]
    filtered_colors <- colors_all[filtered_indices]
    filtered_shapes <- shapes_all[filtered_indices]

    linear_comp_df %>%
        ggplot(aes(x = linear, y = m, color = linear, shape = linear)) +
        geom_point(position = position_dodge(width = 0.5)) +
        geom_errorbar(aes(ymin = lower, ymax= upper, width = 0.1), position = position_dodge(width = 0.5)) +
        geom_text(aes(label = p_val_symbol, vjust=vjust_value), size = 4.5, show.legend = FALSE) + # Add repelled labels for p val symbol
        facet_grid(formula(paste(as.name(y_condition_name), "~", as.name(x_condition_name)))) +
        scale_color_manual(name = "Model Type", labels = filtered_labels, values = filtered_colors) +
        scale_shape_manual(name = "Model Type", labels = filtered_labels, values = filtered_shapes) +
        ylab("Delta Log Likelihood (per word)") +
        scale_y_continuous(labels=labels, breaks=breaks, minor_breaks=NULL) +
        expand_limits(y = 0) +
        expand_limits(y = y_max_plot) +
        theme(
            legend.position="bottom",
            axis.title.x=element_blank(),
            axis.text.x=element_blank(),  # Remove x-axis text
            axis.ticks.x=element_blank(),  # Remove x-axis ticks
            text=element_text(family="serif", size=16),
            strip.text=element_text(size=16),
        )

    ggsave(path, device="pdf", width=6, height=6)

}
