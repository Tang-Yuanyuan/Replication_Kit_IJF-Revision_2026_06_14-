args <- commandArgs(trailingOnly = TRUE)

if (length(args) >= 1) {
  PROJECT_ROOT <- normalizePath(args[[1]], winslash = "/", mustWork = TRUE)
} else if (!exists("PROJECT_ROOT", inherits = TRUE)) {
  PROJECT_ROOT <- normalizePath(getwd(), winslash = "/", mustWork = TRUE)
}

source(file.path(PROJECT_ROOT, "R", "00_setup.R"), encoding = "UTF-8")

read_subdir  <- if (length(args) >= 2) args[[2]] else "empirical4.2"
write_subdir <- if (length(args) >= 3) args[[3]] else paste0(read_subdir, "_weighted")

data_dir <- file.path(paths$temp_data, read_subdir)
out_dir  <- file.path(paths$temp_data, write_subdir)
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

train_df <- read.csv(file.path(data_dir, "train_data.csv"), check.names = FALSE, fileEncoding = "UTF-8")
test_df  <- read.csv(file.path(data_dir, "test_data.csv"),  check.names = FALSE, fileEncoding = "UTF-8")

prepare_4_2_data <- function(data) {
  factor_vars <- c(
    "location", "province", "weekday", "heard_about_global_warming",
    "know_about_low_carbon", "know_about_carbon_neutrality",
    "know_about_carbon_policy"
  )
  for (var in factor_vars) {
    data[[var]] <- as.factor(data[[var]])
  }
  data$location <- relevel(data$location, ref = "city")
  data$heard_about_global_warming <- relevel(data$heard_about_global_warming, ref = "no")
  levels_know <- c("never", "heard but do not know", "heard and know", "familiar")
  data$know_about_low_carbon <- factor(data$know_about_low_carbon, levels = levels_know)
  data$know_about_carbon_neutrality <- factor(data$know_about_carbon_neutrality, levels = levels_know)
  data$know_about_carbon_policy <- factor(data$know_about_carbon_policy, levels = levels_know)
  data
}

train_df <- prepare_4_2_data(train_df)
test_df  <- prepare_4_2_data(test_df)

base_demos <- paste(
  "ifpollution", "living_area_ln", "age_ln", "is_bachelor", "location",
  "female", "married", "income_level", "youth", "older_adults",
  "partymember", "province", "weekday", "ifsunny",
  sep = " + "
)
base_all <- paste(
  base_demos,
  "heard_about_global_warming",
  "know_about_low_carbon",
  "know_about_carbon_neutrality",
  "know_about_carbon_policy",
  sep = " + "
)

# Subset and normalize weights per energy-type group
train_car   <- subset(train_df, publictrans < 5)
train_car$weights   <- train_car$weights   / mean(train_car$weights)

train_elec  <- subset(train_df, conditionernumber == 1)
train_elec$weights  <- train_elec$weights  / mean(train_elec$weights)

train_green <- subset(train_df, energy_consume2020 > 1000)
train_green$weights <- train_green$weights / mean(train_green$weights)

fit_logit_weighted <- function(outcome, controls, exposure, data) {
  glm(
    stats::reformulate(c(strsplit(controls, " \\+ ")[[1]], exposure), response = outcome),
    data    = data,
    family  = binomial(link = "logit"),
    weights = data$weights
  )
}

model_demo_car   <- fit_logit_weighted("y_car",   base_demos, "caruse",            train_car)
model_all_car    <- fit_logit_weighted("y_car",   base_all,  "caruse",            train_car)
model_demo_elec  <- fit_logit_weighted("y_elec",  base_demos, "conditioner1month", train_elec)
model_all_elec   <- fit_logit_weighted("y_elec",  base_all,  "conditioner1month", train_elec)
model_demo_green <- fit_logit_weighted("y_green", base_demos, "mainuseelec",       train_green)
model_all_green  <- fit_logit_weighted("y_green", base_all,  "mainuseelec",       train_green)

logit_probs_demos <- data.frame(
  prob_car   = predict(model_demo_car,   newdata = test_df, type = "response"),
  prob_elec  = predict(model_demo_elec,  newdata = test_df, type = "response"),
  prob_green = predict(model_demo_green, newdata = test_df, type = "response")
)
logit_probs_all <- data.frame(
  prob_car   = predict(model_all_car,   newdata = test_df, type = "response"),
  prob_elec  = predict(model_all_elec,  newdata = test_df, type = "response"),
  prob_green = predict(model_all_green, newdata = test_df, type = "response")
)

write.csv(logit_probs_demos, file.path(out_dir, "logit_probs_demos.csv"), row.names = FALSE)
write.csv(logit_probs_all,   file.path(out_dir, "logit_probs_all.csv"),   row.names = FALSE)

message("Weighted logit probabilities written to: ", out_dir)
