## Density Data

get_d_points <- function(df, zoom_in=FALSE, zoom_in_threshold=NULL) {
    if (zoom_in) {
        min_surp = min(df$surp)
        max_surp = max(df$surp)
        
        density_range_1 = density(df$surp, n=2048, from=min_surp, to=zoom_in_threshold)
        density_range_2 = density(df$surp, n=2048, from=zoom_in_threshold, to=max_surp)
        
        x = c(density_range_1$x, density_range_2$x)
        y = c(density_range_1$y, density_range_2$y)
    } else {
        x = density(df$surp, n=2048)$x
        y = density(df$surp, n=2048)$y
    }
    
    return(data.frame(x, y))
}

# Function to get density data
get_density_data <- function(eye_df, zoom_in, zoom_in_threshold) {
    cat("------ Getting density_data...\n")
    density_data = data.frame()

    for (x_condition_val in x_condition_vals){
        for(y_condition_val in y_condition_vals){
            sub_df = eye_df %>% filter(!!sym(x_condition_name) == x_condition_val, !!sym(y_condition_name) == y_condition_val)
            cat("Density | ", x_condition_name, " =", x_condition_val, " | ", y_condition_name, " =", y_condition_val, "| ", dim(sub_df)[1], "/", dim(eye_df)[1], "rows ---------- \n")

            dummy_df = sub_df %>%
                do({get_d_points(., zoom_in, zoom_in_threshold)}) %>%
                filter(x>=0, x<=20)
            density_data = rbind(density_data, dummy_df %>% mutate(!!sym(x_condition_name) := x_condition_val, !!sym(y_condition_name) := y_condition_val))
        }
    }
    return(density_data)
}