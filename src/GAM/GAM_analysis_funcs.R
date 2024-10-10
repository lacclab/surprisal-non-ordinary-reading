
shhh <- suppressPackageStartupMessages # It's a library, so shhh!
shhh(library( mgcv ))
shhh(library(dplyr))
shhh(library(ggplot2))
shhh(library(lme4))
shhh(library(tidymv))
shhh(library(gamlss))
shhh(library(gsubfn))
shhh(library(lmerTest))
shhh(library(tidyverse))
shhh(library(boot))
shhh(library(rsample))
shhh(library(plotrix))
shhh(library(ggrepel))
shhh(library(mgcv))
library(tidyr)
library(jmuOutlier) # For paired permutation tests
library(purrr)
library(CIPerm)

theme_set(theme_bw())
options(digits=4)
options(dplyr.summarise.inform = FALSE)

## Compare Linear and Non-Linear GAMs using Cross-Validation

get_surp_coef <- function(model, model_name) {
  if (model_name == "linear") {
    # Extract and return the coefficient for 'surp'
    return(coef(model)["surp"])
  } else {
    # If the model is not linear, return a message
    return(NA)
  }
}

model_cross_val <- function(model_name, model_formula, df, RT_col, num_folds=10){
    cat("CV | model: ", model_name, "\n")
    
    folds <- cut(seq(1,nrow(df)),breaks=num_folds,labels=FALSE) 

    result_df = data.frame()
    for(i in 1:num_folds){
        testIndexes = which(folds==i,arr.ind=TRUE)
        test_df = df[testIndexes,]
        train_df = df[-testIndexes,]
        # Fit
        model = gam(as.formula(model_formula), data = train_df)
        # Predict
        stdev = sigma(model)
        cat("CV | i:", i, "\t train size:", dim(train_df)[1], "\t sd:", stdev, "\n")
        preds_vec = predict(model, newdata=test_df)
        logliks_vec <- log(dnorm(test_df[[RT_col]],
                            mean=preds_vec,
                            sd=stdev))
        if (-Inf %in% logliks_vec) {
            print("logliks contains -Inf")
        } 
        # save preds_vec and logliks_vec
        res_df <- test_df %>%
            mutate(RT_true = !!sym(RT_col)) %>%
            select("unique_paragraph_id", "RT_true")
        res_df$RT_pred <- as.numeric(preds_vec)
        res_df$logliks <- as.numeric(logliks_vec)
        res_df$squared_error <- as.numeric((res_df$RT_pred - res_df$RT_true)^2)
        res_df$surp_coef <- get_surp_coef(model, model_name)
        result_df <- rbind(result_df, res_df)
    }
    cat(sprintf("CV | logliks: %d rows", dim(logliks_vec)[1]))
    return(result_df)
}

get_smooths_new <- function(smooths_db, m, linear, additive_model, x_condition_val, y_condition_val) {
    terms_to_predict = get_terms_to_predict(linear, additive_model)
    per_term_predictions = predict(m, newdata=testData, terms=terms_to_predict, type="terms")
    predictions = rowSums(per_term_predictions)
    y_preds = testData %>% mutate(y=predictions)

    # Extract mean and 5% and 95% percentile y-values for each surprisal value
    smooths = models %>% 
        unnest(y_preds) %>% 
        dplyr::select(surp, y) %>% 
        group_by(surp) %>% 
        summarise(y_lower=quantile(y, alpha / 2), 
                    y_upper=quantile(y, 1 - alpha / 2),
                    y=mean(y)) %>% 
        ungroup()

    smooths = smooths %>% mutate(delta = 0 - y[1], y=y + delta, y_lower= y_lower + delta, y_upper=y_upper + delta) # Fix 0 surprisal = 0 ms
    smooths_db = rbind(smooths_db, smooths %>% mutate(!!sym(x_condition_name) := x_condition_val, !!sym(y_condition_name) := y_condition_val, linear = linear, additive_model=additive_model))

    return(smooths_db)
}


fit_without_cross_val <- function(model_name, model_formula, df, RT_col, num_folds){
    cat("CV | model: ", model_name, "\n")
    test_df = df
    train_df = df

    # Fit
    model = gam(as.formula(model_formula), data = train_df)
    # Predict
    stdev = sigma(model)
    cat("CV | i:", i, "\t train size:", dim(train_df)[1], "\t sd:", stdev, "\n")
    preds_vec = predict(model, newdata=test_df)
    logliks_vec <- log(dnorm(test_df[[RT_col]],
                        mean=preds_vec,
                        sd=stdev))
    if (-Inf %in% curr_logliks_vec) {
        print("logliks contains -Inf")
    } 
    # add preds_vec and logliks_vec to test_df
    test_df$preds <- preds_vec
    test_df$logliks <- logliks_vec
    test_df$squared_error <- (test_df$pred_vec - test_df$true_vec)^2
    # r = cor(pred_vec, true_vec)

    cat(sprintf("CV | logliks: %d rows", dim(logliks_vec)[1]))
    return(test_df)
}

get_models_forms <- function(models_names, RT_col, as_formulas = FALSE) {
    models_forms <- list()

    for (model_name in models_names) {
        if (model_name == "baseline") {
            model_formula <- paste0(
                RT_col, " ~ te(freq, len, bs = 'cr') + te(prev_freq, prev_len, bs = 'cr')"
            )
        } else if (model_name == "linear") {
            model_formula <- paste0(
                RT_col, " ~ surp + prev_surp + te(freq, len, bs = 'cr') + te(prev_freq, prev_len, bs = 'cr')"
            )
        } else if (model_name == "nonlinear") {
            model_formula <- paste0(
                RT_col, " ~ s(surp, bs = 'cr', k = 6) + s(prev_surp, bs = 'cr', k = 6) + te(freq, len, bs = 'cr') + te(prev_freq, prev_len, bs = 'cr')"
            )
        }
        
        if (as_formulas == TRUE){
            models_forms[[model_name]] <- as.formula(model_formula)
        } else {
            models_forms[[model_name]] <- model_formula
        }
    }

    return(models_forms)
}


fit_models_and_get_dll_raw_df <- function(eye_d, models_names, RT_col, use_CV=T, num_folds=10){
    cat("----- Getting dll_raw_df...\n")
    models_forms = get_models_forms(models_names, RT_col) # GAM formulas
    print(models_forms)

    dll_raw_df = data.frame()
    for (x_condition_val in x_condition_vals) {
        for(y_condition_val in y_condition_vals){
            sub_df = eye_df %>% filter(!!sym(x_condition_name) == x_condition_val, !!sym(y_condition_name) == y_condition_val)
            cat("Fit | ", x_condition_name, " =", x_condition_val, " | ",y_condition_name, " =", y_condition_val, "| ", dim(sub_df)[1], "/", dim(eye_df)[1], "rows ---------- \n")
            
            if (use_CV) {
                result_df = data.frame(model_name=models_names, model_formula=models_forms) %>%
                mutate(result_df = map2(models_names, models_forms, model_cross_val, df=sub_df, RT_col=RT_col, num_folds=num_folds))
            } else {
                result_df = data.frame(model_name=models_names, model_formula=models_forms) %>%
                mutate(result_df = map2(models_names, models_forms, fit_without_cross_val, df=sub_df, RT_col=RT_col, num_folds=1))
            }
            result_df = (result_df %>% unnest(cols = c(result_df)) 
                %>% select(-contains("model_formula"))
                %>% mutate(!!sym(x_condition_name) := x_condition_val, !!sym(y_condition_name) := y_condition_val, RT_col = RT_col, use_CV = use_CV))
            dll_raw_df = rbind(dll_raw_df, result_df)
        }
    }
    return(as.data.frame(dll_raw_df))
}

get_target_model_dll_df <- function(dll_raw_df, x_condition_val, y_condition_val, model_name_val){
    # Filter by data params
    target_df = dll_raw_df %>% filter(!!sym(x_condition_name) == x_condition_val, !!sym(y_condition_name) == y_condition_val, model_name == model_name_val)
    # Print n rows
    cat("Raw df | ", x_condition_name, " =", x_condition_val, " | ",y_condition_name, " =", y_condition_val, " | model = ", model_name_val, " | n rows: ",dim(target_df)[1], "\n")
    return(target_df)
}
get_baseline_model_dll_df <- function(dll_raw_df, x_condition_val, y_condition_val){
    # Filter by data params
    baseline_df = dll_raw_df %>% filter(!!sym(x_condition_name) == x_condition_val, !!sym(y_condition_name) == y_condition_val, model_name == "baseline")
    # Print n rows
    cat("Raw df | ", x_condition_name, " =", x_condition_val, " | ",y_condition_name, " =", y_condition_val, " | n rows: ",dim(baseline_df)[1], "\n")
    return(baseline_df)
}

get_dll_vec_by_params <- function(dll_raw_df, x_condition_val, y_condition_val, model_name_val){
    target_df = get_target_model_dll_df(dll_raw_df, x_condition_val, y_condition_val, model_name_val)
    baseline_df = get_baseline_model_dll_df(dll_raw_df, x_condition_val, y_condition_val)
    # Calculate Delta Log Likelihood
    dll_vec = target_df$logliks - baseline_df$logliks
    return(dll_vec)
}

get_delta_SE_vec_by_params <- function(dll_raw_df, x_condition_val, y_condition_val, model_name_val){
    target_df = get_target_model_dll_df(dll_raw_df, x_condition_val, y_condition_val, model_name_val)
    baseline_df = get_baseline_model_dll_df(dll_raw_df, x_condition_val, y_condition_val)
    # Calculate Delta of the Squared Errors
    delta_SE = target_df$squared_error - baseline_df$squared_error
    return(delta_SE)
}

get_delta_MSE_vec_by_params <- function(dll_raw_df, x_condition_val, y_condition_val, model_name_val){
    target_df = get_target_model_dll_df(dll_raw_df, x_condition_val, y_condition_val, model_name_val)
    baseline_df = get_baseline_model_dll_df(dll_raw_df, x_condition_val, y_condition_val)
    # Calculate Delta of MSE
    delta_MSE = mean(target_df$squared_error) - mean(baseline_df$squared_error)
    return(delta_MSE)
}

get_delta_r_by_params <- function(dll_raw_df, x_condition_val, y_condition_val, model_name_val){
    target_df = get_target_model_dll_df(dll_raw_df, x_condition_val, y_condition_val, model_name_val)
    baseline_df = get_baseline_model_dll_df(dll_raw_df, x_condition_val, y_condition_val)
    # Calculate Delta of Correlation (r)
    target_r = cor(target_df$RT_true, target_df$RT_pred)
    baseline_r = cor(baseline_df$RT_true, baseline_df$RT_pred)
    delta_r = target_r - baseline_r
    return(delta_r)
}

update_dll_stats_df <- function(dll_stats_df, dll_vals, model_name_val, x_condition_val, y_condition_val) {
    dll_stats_df <- rbind(dll_stats_df, setNames(
        data.frame(
            dll = dll_vals,
            linear = model_name_val,
            x_condition_name = x_condition_val,
            y_condition_name = y_condition_val
        ),
        c("dll", "linear", x_condition_name, y_condition_name))
    )
    return(dll_stats_df)
}

update_dll_comp_stats_df <- function(dll_comp_stats_df, models_dll_list, x_condition_val, y_condition_val) {
    # Convert the named list of vectors into a data frame
    dll_models_df <- as.data.frame(models_dll_list)
    
    # Add the x_condition_name and y_condition_name columns
    dll_models_df$x_condition_temp <- x_condition_val
    dll_models_df$y_condition_temp <- y_condition_val
    
    # Rename the columns using setNames
    colnames(dll_models_df)[colnames(dll_models_df) == "x_condition_temp"] <- x_condition_name
    colnames(dll_models_df)[colnames(dll_models_df) == "y_condition_temp"] <- y_condition_name

    # Append the new data to the existing data frame
    dll_comp_stats_df <- rbind(dll_comp_stats_df, dll_models_df)
    return(dll_comp_stats_df)
}

get_dll_comp_stats_df <- function(dll_raw_df, models_names) {
    cat("----- Getting dll_comp_stats_df... \n")
    dll_stats_df = data.frame()
    dll_comp_stats_df = data.frame()

    for (x_condition_val in x_condition_vals) {
        for(y_condition_val in y_condition_vals){
            models_dll_list <- list()

            for (model_name_val in models_names) {
                # Get the dll vector for the current model and data
                cat("DLL | ", x_condition_name, " =", x_condition_val, " | ",y_condition_name, " =", y_condition_val, "\n")
                dll_vec <- get_dll_vec_by_params(dll_raw_df, x_condition_val, y_condition_val, model_name = model_name_val)
                
                # Update the stats dataframe
                dll_stats_df <- update_dll_stats_df(dll_stats_df, dll_vec, model_name = model_name_val, x_condition_val, y_condition_val)
                
                # Add the current model's dll vector to the models_dll_list
                models_dll_list[[paste0("dll_", model_name_val)]] <- dll_vec
            }

            # Update the compiled stats dataframe
            dll_comp_stats_df <- update_dll_comp_stats_df(dll_comp_stats_df, models_dll_list, x_condition_val, y_condition_val)
        }
    }
    return(dll_comp_stats_df)
}

get_dll_df_for_permu_test <- function(dll_raw_df, x_condition_val, y_condition_val, model_name_val) {
    dll_vec = get_dll_vec_by_params(dll_raw_df, x_condition_val, y_condition_val, model_name=model_name_val)
    filtered_outliers = data.frame(dll = dll_vec) %>%
                drop_na() %>%
                mutate( mean = mean(dll), sd=sd(dll)) %>%
                filter(dll < mean + sd * 3, dll > mean - sd * 3)
    cleaned_dll_vec = filtered_outliers$dll
    return(cleaned_dll_vec)
}

create_p_df <- function(model_name_val, p_val, x_condition_val, y_condition_val) {
    p_df = setNames(
        data.frame(
            x_condition_name = x_condition_val, 
            y_condition_name = y_condition_val,
            linear = model_name_val,
            p_val = p_val
        ),
        c(x_condition_name, y_condition_name, "linear", "p_val")
    )
    return(p_df)
}

create_dll_results_df <- function(
    dll_raw_df, model_name_val, x_condition_val, y_condition_val
    ) {
    dll_vec = get_dll_df_for_permu_test(dll_raw_df, x_condition_val, y_condition_val, model_name=model_name_val)
    # ttest_p_val = perm.test(dll_vec, num.sim = 1)$p.value
    # perm_test = perm.test(dll_vec)
    # new_p_val = perm_test$p.value
    # n_sim <- as.numeric(str_extract(perm_test[2], "\\d+"))
    n_sim = 1000
    zero_vec = rep(0, length(dll_vec))
    dset_result = dset(dll_vec, zero_vec, nmc=n_sim)
    new_p_val = pval(dset_result, tail = c("Two"))
    CI_result = cint(dset_result, conf.level = 0.95, tail = c("Two"))
    norm_upper = mean(dll_vec) + (1.96 * std.error(dll_vec))
    norm_lower = mean(dll_vec) - (1.96 * std.error(dll_vec))
    w_norm_CI = norm_upper - norm_lower
    upper = CI_result$conf.int[2]
    lower = CI_result$conf.int[1]
    w_new_CI = upper - lower

    dll_results_df = setNames(
        data.frame(
            m = mean(dll_vec), 
            upper = upper,
            lower = lower,
            norm_upper = norm_upper,
            norm_lower = norm_lower,
            w_norm_CI = w_norm_CI,
            w_new_CI = w_new_CI,
            # ttest_pval = ttest_p_val,
            x_condition_name = x_condition_val,
            y_condition_name = y_condition_val,
            linear = model_name_val,
            new_p_val = new_p_val,
            n_sim = n_sim
        ),
        # c("m", "upper", "lower", "ttest_pval", x_condition_name, y_condition_name, "linear", "new_p_val", "n_sim")
        c("m", "upper", "lower", "norm_upper", "norm_lower", "w_norm_CI", "w_new_CI", x_condition_name, y_condition_name, "linear", "new_p_val", "n_sim")
    )
    return(dll_results_df)
}

get_linear_comp_df_using_permutation_test <- function(dll_raw_df, models_names, model_names_to_compare = NULL) {
    cat("----- Getting linear_comp_df using permutation test... \n")
    
    # If model_names_to_compare is NULL, use models_names without "baseline" and "linear"
    if (is.null(model_names_to_compare)) {
        model_names_to_compare <- setdiff(models_names, c("baseline", "linear"))
        print(model_names_to_compare)
    }
    
    p_vals_df <- data.frame()
    linear_comp_df <- data.frame()
    
    for (x_condition_val in x_condition_vals) {
        for (y_condition_val in y_condition_vals) {
            # Get linear model dll
            dll_linear <- get_dll_df_for_permu_test(dll_raw_df, x_condition_val, y_condition_val, model_name = "linear")
            dll_comp_linear <- create_dll_results_df(dll_raw_df, model_name = "linear", x_condition_val, y_condition_val)
            linear_comp_df <- rbind(linear_comp_df, dll_comp_linear)
            
            for (model_name_val in model_names_to_compare) {
                # Get comparable model dll
                dll_comparable <- get_dll_df_for_permu_test(dll_raw_df, x_condition_val, y_condition_val, model_name = model_name_val)
                
                # Test mean(dll_linear) != mean(dll_other)
                comparable_vs_linear <- perm.test(dll_comparable, dll_linear, num.sim = 1000)
                p_comparable_vs_linear <- create_p_df(model_name = model_name_val, comparable_vs_linear$p.value, x_condition_val, y_condition_val)
                p_vals_df <- rbind(p_vals_df, p_comparable_vs_linear)

                # Test mean(dll) != 0
                dll_comp_comparable <- create_dll_results_df(dll_raw_df, model_name = model_name_val, x_condition_val, y_condition_val)
                linear_comp_df <- rbind(linear_comp_df, dll_comp_comparable)

                cat("Comp | ", x_condition_name, " =", x_condition_val, " | ", y_condition_name, " =", y_condition_val, "| model =", model_name_val, "| dll len:", length(dll_comparable), "CI of models ---------- \n")
            }
        }
    }
    cat("Comp | linear_comp_df:", dim(linear_comp_df)[1], "CI rows ---------- \n")
    return(list(p_vals_df = p_vals_df, linear_comp_df = linear_comp_df))
}

## Fit GAM using bootstrap

train_gam_bootstrap <- function(linear, curr_df, weights) {
    if (linear == "linear") {
        formula <- RT_col ~ surp + prev_surp + te(freq, len, bs = 'cr') + te(prev_freq, prev_len, bs = 'cr')
    } else if (linear == "nonlinear") {
        formula <- RT_col ~ s(surp, bs = 'cr', k = 6) + s(prev_surp, bs = 'cr', k = 6) + te(freq, len, bs = 'cr') + te(prev_freq, prev_len, bs = 'cr')
    }
    m = gam(formula, data = curr_df, weights = weights)
    return(m)
}

get_terms_to_predict <- function(linear, additive_model) {
    if (linear == "linear" && additive_model) {
        terms_to_predict <- c("surp", "prev_surp")
    } else if (linear == "linear" && !additive_model) {
        terms_to_predict <- c("surp")
    } else if (linear == "nonlinear" && additive_model) {
        terms_to_predict <- c("s(surp)", "s(prev_surp)")
    } else if (linear == "nonlinear") {
        terms_to_predict <- c("s(surp)")
    } else {
        terms_to_predict <- c("undefined")
    }
    
    return(terms_to_predict)
}

fit_gam_inner = function(bootstrap_sample, mean_predictors, linear, additive_model) {
    curr_df = bootstrap_sample$data
    weights = tabulate(as.integer(bootstrap_sample), nrow(curr_df))

    # GAM formulas
    model = train_gam_bootstrap(linear, curr_df, weights)
    terms_to_predict = get_terms_to_predict(linear, additive_model)

    # p value of terms
    if (linear == "linear"){
        surp_p_val = summary(model)$p.table['surp',4]
        prev_surp_p_val = summary(model)$p.table['prev_surp',4]
    }
    else {
        surp_p_val = summary(model)$s.table['s(surp)',4]
        prev_surp_p_val = summary(model)$s.table['s(prev_surp)',4]
    }
    converged = model$converged
    
    if (converged==FALSE){
        cat("model did not converge", "\n")
    }
    if (surp_p_val <0) {
        cat("Negative surp_p_val ", surp_p_val, "\n")
    } 
    if (prev_surp_p_val <0) {
        cat("Negative prev_surp_p_val ", prev_surp_p_val, "\n")
    }
    
    # new data
    if (zoom_in==TRUE){
        sampling_range <- seq(0, zoom_in_threshold, by = 0.00001)
    } else {
        sampling_range = seq(0,20,by=0.1)
    }

    newdata = data.frame(
        surp=sampling_range, prev_surp=mean_predictors$surp,
        freq=mean_predictors$freq, prev_freq=mean_predictors$freq,
        len=mean_predictors$freq, prev_len=mean_predictors$freq)
    newdata$surp_squared <- newdata$surp^2
    
    # Returns a matrix N_samples * N_terms.
    per_term_predictions = predict(model, newdata=newdata, terms=terms_to_predict, type="terms")

    # Additive model -- sum across predictor response contributions (matrix columns).
    predictions = rowSums(per_term_predictions)
    preds = newdata %>% mutate(y=predictions)
    return(list(preds=preds, surp_p_val=surp_p_val, prev_surp_p_val=prev_surp_p_val, converged=converged))
}

fit_gam = function(df, mean_predictors, linear, additive_model, alpha=0.05) {
    df <- df %>% rename(RT_col = !!sym(RT_col))
    # Bootstrap-resample data
    boot_results = df %>% bootstraps(times=10) %>% 
        # Fit a GAM and get predictions for each sample
        mutate(
            res=map(splits, fit_gam_inner, mean_predictors=mean_predictors, linear=linear, additive_model=additive_model)
        ) %>% 
        unnest_wider(
            res, 
            names_repair = ~ str_remove(str_c("res_", .x), "[.]+")
        )
    
    smooths_p_vals = boot_results %>% 
        mutate(
            surp_p_val = res_surp_p_val,
            prev_surp_p_val = res_prev_surp_p_val,
            converged = res_converged
        ) %>% 
        dplyr::select(surp_p_val, prev_surp_p_val, converged)

    # Extract mean and 5% and 95% percentile y-values for each surprisal value
    smooths = boot_results %>% 
        unnest(res_preds) %>% 
        dplyr::select(surp, y) %>% 
        group_by(surp) %>% 
        summarise(y_lower=quantile(y, alpha / 2), 
                    y_upper=quantile(y, 1 - alpha / 2),
                    y=mean(y)) %>% 
        ungroup()
    
    return (list(
        smooths=smooths,
        smooths_p_vals=smooths_p_vals
        ))
}

get_gam_smooths_including_bootstrap <- function(eye_df, additive_model){
    cat("------ Getting gam_smooths including bootstrap...\n")
    linear_smooths = data.frame()
    nonlinear_smooths = data.frame()
    all_p_vals = data.frame()

    for (x_condition_val in x_condition_vals){
        for(y_condition_val in y_condition_vals){
            sub_df = eye_df %>% filter(!!sym(x_condition_name) == x_condition_val, !!sym(y_condition_name) == y_condition_val)
            cat("Fit | ", x_condition_name, " =", x_condition_val, " | ",y_condition_name, " =", y_condition_val, "| ", dim(sub_df)[1], "/", dim(eye_df)[1], "rows ---------- \n")

            mean_predictors = sub_df %>% summarise(surp = mean(surp), len = mean(len), freq = mean(freq))
            
            cat("Fit | linear \n")
            linear="linear"
            smooths_results = sub_df %>% fit_gam(., mean_predictors, linear=linear, additive_model=additive_model)
            smooths = smooths_results$smooths
            smooths_p_vals = smooths_results$smooths_p_vals
            all_p_vals = rbind(all_p_vals, smooths_p_vals %>% mutate(!!sym(x_condition_name) := x_condition_val, !!sym(y_condition_name) := y_condition_val, linear = linear, additive_model=additive_model))
            smooths = smooths %>% mutate(delta = 0 - y[1], y=y + delta, y_lower= y_lower + delta, y_upper=y_upper + delta) # Fix 0 surprisal = 0 ms
            linear_smooths = rbind(linear_smooths, smooths %>% mutate(!!sym(x_condition_name) := x_condition_val, !!sym(y_condition_name) := y_condition_val, linear = linear, additive_model=additive_model))

            cat("Fit | nonlinear \n")
            linear="nonlinear"
            smooths_results = sub_df %>% fit_gam(., mean_predictors, linear=linear, additive_model=additive_model)
            smooths = smooths_results$smooths
            smooths_p_vals = smooths_results$smooths_p_vals
            all_p_vals = rbind(all_p_vals, smooths_p_vals %>% mutate(!!sym(x_condition_name) := x_condition_val, !!sym(y_condition_name) := y_condition_val, linear = linear, additive_model=additive_model))
            smooths = smooths %>% mutate(delta = 0 - y[1], y=y + delta, y_lower= y_lower + delta, y_upper=y_upper + delta) # Fix 0 surprisal = 0 ms
            nonlinear_smooths = rbind(nonlinear_smooths, smooths %>% mutate(!!sym(x_condition_name) := x_condition_val, !!sym(y_condition_name) := y_condition_val, linear = linear, additive_model=additive_model))
        
        }
    }
    return(list(
        linear_smooths=linear_smooths, 
        nonlinear_smooths=nonlinear_smooths,
        all_p_vals=all_p_vals
        ))
}

get_zoom_in_threshold <- function(eye_df) {
    return(quantile(eye_df$surp, 0.80))
}

print_surp_quantiles <- function(eye_df) {
    quantile_99 = quantile(eye_df$surp, 0.99)
    quantile_95 = quantile(eye_df$surp, 0.95)
    quantile_90 = quantile(eye_df$surp, 0.90)
    quantile_80 = quantile(eye_df$surp, 0.80)
    quantile_50 = quantile(eye_df$surp, 0.50)
    cat("quantile_99 of surp col ", surp_column, " = ", quantile_99, "\n")
    cat("quantile_95 of surp col ", surp_column, " = ", quantile_95, "\n")
    cat("quantile_90 of surp col ", surp_column, " = ", quantile_90, "\n")
    cat("quantile_80 of surp col ", surp_column, " = ", quantile_80, "\n")
    cat("quantile_50 of surp col ", surp_column, " = ", quantile_50, "\n")
}

update_configs_by_surp_col <- function(surp_column) {
    if (analysis_type =="Different_Surprisal_Context"){
        # get reread config
        reread = surp_df %>% filter(surp_col == surp_column) %>% select(reread)
        if (reread == 1){
            y_condition_vals=c(1)
            y_condition_labels=c("Repeated Reading")
        }
        else if (reread == 0){
            y_condition_vals=c(0)
            y_condition_labels=c("First Reading")
        }
        else { # reread = "both"
            y_condition_vals=c(0, 1)
            y_condition_labels=c("First Reading", "Repeated Reading")
        }
    }
    return(list(
        reread=reread,
        y_condition_vals=y_condition_vals,
        y_condition_labels=y_condition_labels
        ))
}

filter_eye_df_by_config <- function(eye_df) {
    if (reread==1){
        dim_before = dim(eye_df)[1]
        eye_df = eye_df %>% filter(reread_condition == 1) # repeated reading
        cat("Repeated Reading | ", dim(eye_df)[1], "/", dim_before, "rows ---------- \n")
    }
    if (reread==0){
        dim_before = dim(eye_df)[1]
        eye_df = eye_df %>% filter(reread_condition == 0) # repeated reading
        cat("First Reading | ", dim(eye_df)[1], "/", dim_before, "rows ---------- \n")
    }
    if (reread_10_11==TRUE){
        dim_before = dim(eye_df)[1]
        eye_df = eye_df %>% filter(reread_consecutive_condition %in% c(10, 11)) # repeated reading
        cat("Articles 10 11 | ", dim(eye_df)[1], "/", dim_before, "rows ---------- \n")
    }
    if (filter_hunting==TRUE){
        dim_before = dim(eye_df)[1]
        eye_df = eye_df %>% filter(has_preview_condition == "Hunting")
        cat("Hunting | ", dim(eye_df)[1], "/", dim_before, "rows ---------- \n")
    }    
    return(eye_df)
}

calc_density_data <- function(eye_df, density_path, zoom_in_threshold){
    # Density Data
    if ((only_plot == FALSE & only_dll_tests == FALSE) | only_density == TRUE) {   
        density_data <- get_density_data(eye_df, zoom_in, zoom_in_threshold)
        write.csv(density_data, density_path)
    }
    return(eye_df)
}

filter_eye_df_by_zoom_in <- function(eye_df, zoom_in_threshold){
    # zoom in filter
    if (zoom_in==TRUE){
        dim_before = dim(eye_df)[1]
        eye_df = eye_df %>% filter(surp < zoom_in_threshold)
        cat("Zoom in surp | ", dim(eye_df)[1], "/", dim_before, "rows ---------- \n")
    }
    return(eye_df)
}

save_filter_params <- function(zoom_in, zoom_in_threshold, reread, reread_10_11, filter_hunting){
    new_path <- here(results_path, paste0(RT_col, " - ", surp_column, "_log_filter_params.csv"))

    # Get the directory of the file path
    dir_path <- dirname(new_path)
    
    # Create the directory if it doesn't exist
    if (!dir.exists(dir_path)) {
        dir.create(dir_path, recursive = TRUE)
    }

    # Get the current time
    current_time <- Sys.time()

    # Convert the parameters and current time to a single-row data frame
    data_to_append <- data.frame(
    time = current_time,
    surp_column = surp_column,
    zoom_in = zoom_in,
    zoom_in_threshold = zoom_in_threshold,
    reread = reread,
    reread_10_11 = reread_10_11,
    filter_hunting = filter_hunting
    )

    # Append the data to the CSV file at the new path
    write.table(data_to_append, file = new_path, append = TRUE, sep = ",", col.names = !file.exists(new_path), row.names = FALSE)
}