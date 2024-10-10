
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
shhh(library(tidyr))
shhh(library(CIPerm))
library(jmuOutlier) 
library(here)
theme_set(theme_bw())
options(digits=4)
options(dplyr.summarise.inform = FALSE)
# install.packages("gratia")
library(gratia)

## Import files
source(here("src", "GAM", "preprocess_df.R"))

basic_file_path = "/data/home/shared/onestop/processed/et_20240505_with_all_surp20240624.csv"
surp_column = "EleutherAI-gpt-j-6B-Surprisal-Context-P"
original_eye_df <- get_eye_data(basic_file_path, surp_column=surp_column)
RT_col = "FirstPassGD"
eye_df <- clean_eye_df(original_eye_df, RT_col, result_dir="results 0<RT<3000 firstpassNA")
eye_df = filter_eye_df_by_surp(eye_df)

# filter ordinary reading, repeated reading reading
sub_df = eye_df %>% filter(has_preview_condition == "Gathering", reread_condition == 1)

baseline_formula = paste0(
                RT_col, " ~ te(freq, len, bs = 'cr') + te(prev_freq, prev_len, bs = 'cr')"
            )
linear_formula <- paste0(
                RT_col, " ~ surp + prev_surp + te(freq, len, bs = 'cr') + te(prev_freq, prev_len, bs = 'cr')"
            )
nonlinear_formula <- paste0(
                RT_col, " ~ s(surp, bs = 'cr', k = 6) + s(prev_surp, bs = 'cr', k = 6) + te(freq, len, bs = 'cr') + te(prev_freq, prev_len, bs = 'cr')"
            )

get_surp_coef <- function(model, model_name) {
    if (model_name == "linear") {
        # Extract and return the coefficient for 'surp'
        return(coef(model)["surp"])
    } else {
        # If the model is not linear, return a message
        return(NA)
    }
}

model_cross_val_2 <- function(model_name, model_formula, df, RT_col, num_folds=10){
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
            select("subject_id", "unique_paragraph_id", "RT_true")
        res_df$RT_pred <- as.numeric(preds_vec)
        res_df$logliks <- as.numeric(logliks_vec)
        res_df$squared_error <- as.numeric((res_df$RT_pred - res_df$RT_true)^2)
        res_df$surp_coef <- get_surp_coef(model, model_name)
        result_df <- rbind(result_df, res_df)
    }
    result_df$model_name = model_name
    cat(sprintf("CV | logliks: %d rows", dim(logliks_vec)[1]))
    return(list(result_df=result_df, model=model))
}

base_resuls = model_cross_val_2("baseline", baseline_formula, sub_df, RT_col)
base_df = base_resuls$result_df
base_model = base_resuls$model

lin_resuls = model_cross_val_2("linear", linear_formula, sub_df, RT_col)
lin_df = lin_resuls$result_df
lin_model = lin_resuls$model

nonlin_resuls = model_cross_val_2("nonlinear", nonlinear_formula, sub_df, RT_col)
nonlin_df = nonlin_resuls$result_df
nonlin_model = nonlin_resuls$model

dll_raw_df = rbind(base_df, lin_df, nonlin_df)
dll_raw_df$has_preview_condition = "Gathering"
dll_raw_df$reread_condition = 1

write.csv(dll_raw_df, "/src/GAM/tests/dll_raw_df_test.csv")

dll_lin = lin_df$logliks - base_df$logliks
dll_nonlin = nonlin_df$logliks - base_df$logliks

print(c(mean(dll_lin), mean(dll_nonlin)))

# statistical test
n_sim = 1000
zero_vec = rep(0, length(dll_vec))
dset_result = dset(dll_vec, zero_vec, nmc=n_sim)
CI_result = cint(dset_result, conf.level = 0.95, tail = c("Two"))
upper = CI_result$conf.int[2]
lower = CI_result$conf.int[1]
m = mean(dll_vec)

plot_dll <- function(dll_df){
    linear_comp_df %>%
        ggplot(aes(x = model, y = m, color = model)) +
        geom_point(position = position_dodge(width = 0.5)) +
        geom_errorbar(aes(ymin = lower, ymax= upper, width = 0.1), position = position_dodge(width = 0.5)) +
        scale_color_manual(name = "Model Type") +
        ylab("Delta Log Likelihood (per word)") +
        expand_limits(y = 0) +
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
