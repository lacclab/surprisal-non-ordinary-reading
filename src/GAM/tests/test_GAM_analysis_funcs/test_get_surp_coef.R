# ------------
# These are not actual tests, just validating how to extrect the coef from a GAM model
# ------------


library(testthat)
library(dplyr)
library(here)
## Import files
source(here("src", "GAM", "GAM_analysis_funcs.R"))

print("First Function")
# Load the necessary packages
library(mgcv)
library(testthat)


# Create some mock data
set.seed(123) # for reproducibility
data <- data.frame(
RT = rnorm(100, mean = 500, sd = 50),
surp = rnorm(100, mean = 0, sd = 1),
prev_surp = rnorm(100, mean = 0, sd = 1),
freq = runif(100, min = 1, max = 100),
len = runif(100, min = 1, max = 10),
prev_freq = runif(100, min = 1, max = 100),
prev_len = runif(100, min = 1, max = 10)
)

# Fit the GAM model
model <- gam(RT ~ surp + prev_surp + te(freq, len, bs = 'cr') + te(prev_freq, prev_len, bs = 'cr'), 
            data = data)

# Extract the surprisal coefficient
surprisal_coefficient <- coef(model)["surp"]
print(surprisal_coefficient)

print("Second Function")
# Load the necessary packages
library(mgcv)
library(ggplot2)

# Set seed for reproducibility
set.seed(123)

# Step 1: Create some example data
n <- 200
surp <- runif(n, 0, 10)  # Random values of 'surp' between 0 and 10
RT <- 3 * sin(surp) + rnorm(n, 0, 0.5)  # Non-linear relationship with some noise
data <- data.frame(RT = RT, surp = surp)

# Step 2: Fit a GAM using cubic regression splines
model <- gam(RT ~ s(surp, bs = "cr"), data = data)

# Print summary of the model
summary(model)

# Step 3: Visualize the smooth term
# Predict the values to plot the smooth
pred <- predict(model, se.fit = TRUE)

# Create a data frame for plotting
plot_data <- data.frame(surp = data$surp, RT = data$RT, fit = pred$fit, se = pred$se.fit)

# Plot using ggplot2
p <- ggplot(plot_data, aes(x = surp, y = RT)) +
  geom_point(alpha = 0.5) +
  geom_line(aes(y = fit), color = "blue") +
  geom_ribbon(aes(ymin = fit - 2 * se, ymax = fit + 2 * se), alpha = 0.2) +
  labs(title = "GAM Fit using Cubic Regression Splines", y = "RT", x = "Surp") +
  theme_minimal()

# Step 4: Save the plot
ggsave("gam_cubic_regression_splines_plot.pdf", plot = p, width = 8, height = 6)

print("Third Function")
# Load the necessary packages
library(mgcv)
library(ggplot2)

# Set seed for reproducibility
set.seed(123)

# Step 1: Create some example data
n <- 200
surp <- runif(n, 0, 10)  # Random values of 'surp' between 0 and 10
RT <- 3 * sin(surp) + rnorm(n, 0, 0.5)  # Non-linear relationship with some noise
data <- data.frame(RT = RT, surp = surp)

# Step 2: Fit a GAM using B-splines
model <- gam(RT ~ s(surp, bs = "bs"), data = data)

# Print summary of the model
summary(model)

# Step 3: Visualize the smooth term
# Predict the values to plot the smooth
pred <- predict(model, se.fit = TRUE)

# Create a data frame for plotting
plot_data <- data.frame(surp = data$surp, RT = data$RT, fit = pred$fit, se = pred$se.fit)

# Plot using ggplot2
p <- ggplot(plot_data, aes(x = surp, y = RT)) +
  geom_point(alpha = 0.5) +
  geom_line(aes(y = fit), color = "blue") +
  geom_ribbon(aes(ymin = fit - 2 * se, ymax = fit + 2 * se), alpha = 0.2) +
  labs(title = "GAM Fit using B-Splines", y = "RT", x = "Surp") +
  theme_minimal()

# Step 4: Save the plot
ggsave("gam_b_splines_plot.pdf", plot = p, width = 8, height = 6)
