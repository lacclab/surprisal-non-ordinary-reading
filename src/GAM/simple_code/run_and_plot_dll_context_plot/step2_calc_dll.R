
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

## Import files
source(here("src", "GAM", "GAM_analysis_funcs.R"))

dll_path = "/src/GAM/tests/dll_raw_df_test.csv"

dll_raw_df = read.csv(dll_path)

x_condition_name="has_preview_condition"
x_condition_labels=c("Ordinary Reading", "Information Seeking")
# x_condition_vals=c("Gathering", "Hunting")
x_condition_vals=c("Gathering")

y_condition_name="reread_condition"
y_condition_labels=c("First Reading", "Repeated Reading")
# y_condition_vals=c(0, 1)
y_condition_vals=c(1)

models_names = c("linear", "nonlinear", "baseline")
# dll_comp_stats_df = get_dll_comp_stats_df(dll_raw_df, models_names)

dll_comp_df = get_linear_comp_df_using_permutation_test(dll_raw_df, models_names)
write.csv(dll_comp_df$linear_comp_df, "/src/GAM/tests/dll_comp_df_test.csv")
write.csv(dll_comp_df$p_vals_df, "/src/GAM/tests/dll_p_vals_df_test.csv")
colnames(dll_comp_df)