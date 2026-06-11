args <- commandArgs(trailingOnly = TRUE)

if (length(args) >= 1) {
  PROJECT_ROOT <- normalizePath(args[[1]], winslash = "/", mustWork = TRUE)
} else if (!exists("PROJECT_ROOT", inherits = TRUE)) {
  PROJECT_ROOT <- normalizePath(getwd(), winslash = "/", mustWork = TRUE)
}

source(file.path(PROJECT_ROOT, "R", "00_setup.R"), encoding = "UTF-8")

data_dir <- file.path(PROJECT_ROOT, "results", "empirical4.2")
if (length(args) >= 2) {
  data_dir <- file.path(PROJECT_ROOT, "results", args[[2]])
}
train_file <- file.path(data_dir, "train_data.csv")
test_file <- file.path(data_dir, "test_data.csv")

train_df <- read.csv(train_file, check.names = FALSE, fileEncoding = "UTF-8")
test_df <- read.csv(test_file, check.names = FALSE, fileEncoding = "UTF-8")

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
test_df <- prepare_4_2_data(test_df)

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

train_car <- subset(train_df, publictrans < 5)
train_elec <- subset(train_df, conditionernumber == 1)
train_green <- subset(train_df, energy_consume2020 > 1000)

fit_logit <- function(outcome, controls, exposure, data) {
  glm(
    stats::reformulate(c(strsplit(controls, " \\+ ")[[1]], exposure), response = outcome),
    data = data,
    family = binomial(link = "logit")
  )
}

model_demo_car <- fit_logit("y_car", base_demos, "caruse", train_car)
model_all_car <- fit_logit("y_car", base_all, "caruse", train_car)

model_demo_elec <- fit_logit("y_elec", base_demos, "conditioner1month", train_elec)
model_all_elec <- fit_logit("y_elec", base_all, "conditioner1month", train_elec)

model_demo_green <- fit_logit("y_green", base_demos, "mainuseelec", train_green)
model_all_green <- fit_logit("y_green", base_all, "mainuseelec", train_green)

logit_probs_demos <- data.frame(
  prob_car = predict(model_demo_car, newdata = test_df, type = "response"),
  prob_elec = predict(model_demo_elec, newdata = test_df, type = "response"),
  prob_green = predict(model_demo_green, newdata = test_df, type = "response")
)

logit_probs_all <- data.frame(
  prob_car = predict(model_all_car, newdata = test_df, type = "response"),
  prob_elec = predict(model_all_elec, newdata = test_df, type = "response"),
  prob_green = predict(model_all_green, newdata = test_df, type = "response")
)

write.csv(logit_probs_demos, file.path(data_dir, "logit_probs_demos.csv"), row.names = FALSE)
write.csv(logit_probs_all, file.path(data_dir, "logit_probs_all.csv"), row.names = FALSE)

message("Logit probabilities written to: ", data_dir)
