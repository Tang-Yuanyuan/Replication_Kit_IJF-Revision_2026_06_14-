args <- commandArgs(trailingOnly = TRUE)

if (length(args) >= 1) {
  PROJECT_ROOT <- normalizePath(args[[1]], winslash = "/", mustWork = TRUE)
} else if (!exists("PROJECT_ROOT", inherits = TRUE)) {
  PROJECT_ROOT <- normalizePath(getwd(), winslash = "/", mustWork = TRUE)
}

source(file.path(PROJECT_ROOT, "R", "00_setup.R"), encoding = "UTF-8")

train_subdir <- if (length(args) >= 2) args[[2]] else "empirical4.2"
sim_subdir <- if (length(args) >= 3) args[[3]] else "empirical4.3"

train_file <- file.path(paths$temp_data, train_subdir, "train_data.csv")
sim_dir <- file.path(paths$temp_data, sim_subdir)
test_file <- file.path(sim_dir, "test_data_simulated.csv")
dir.create(sim_dir, recursive = TRUE, showWarnings = FALSE)

train_df <- read.csv(train_file, check.names = FALSE, fileEncoding = "UTF-8")
test_df <- read.csv(test_file, check.names = FALSE, fileEncoding = "UTF-8")

prepare_common_data <- function(data, ordered_wta = FALSE) {
  if (ordered_wta) {
    wta_levels <- as.character(1:7)
    data$wta_car <- factor(data$wta_car, levels = wta_levels, ordered = TRUE)
    data$wta_elec <- factor(data$wta_elec, levels = wta_levels, ordered = TRUE)
    data$wta_green <- factor(data$wta_green, levels = wta_levels, ordered = TRUE)
  }

  factor_vars <- c(
    "location", "province", "weekday", "heard_about_global_warming",
    "know_about_low_carbon", "know_about_carbon_neutrality",
    "know_about_carbon_policy"
  )
  for (var in factor_vars) {
    data[[var]] <- as.factor(data[[var]])
  }

  data$location <- relevel(data$location, ref = "city")
  data$heard_about_global_warming <- factor(
    data$heard_about_global_warming,
    levels = c("no", "yes and agree", "yes but disagree")
  )
  levels_know <- c("never", "heard but do not know", "heard and know", "familiar")
  data$know_about_low_carbon <- factor(data$know_about_low_carbon, levels = levels_know)
  data$know_about_carbon_neutrality <- factor(data$know_about_carbon_neutrality, levels = levels_know)
  data$know_about_carbon_policy <- factor(data$know_about_carbon_policy, levels = levels_know)
  data
}

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

fit_glm_logit <- function(outcome, controls, exposure, data) {
  glm(
    stats::reformulate(c(strsplit(controls, " \\+ ")[[1]], exposure), response = outcome),
    data = data,
    family = binomial(link = "logit")
  )
}

fit_polr_wta <- function(outcome, controls, exposure, data) {
  MASS::polr(
    stats::reformulate(c(strsplit(controls, " \\+ ")[[1]], exposure), response = outcome),
    data = data,
    Hess = TRUE
  )
}

calc_expected_wta_safe <- function(model, train_data, test_data) {
  resp_var <- as.character(formula(model)[[2]])
  predictor_vars <- setdiff(all.vars(formula(model)), resp_var)

  for (var in predictor_vars) {
    if (is.factor(train_data[[var]])) {
      test_data[[var]] <- factor(test_data[[var]], levels = levels(train_data[[var]]))
    }
  }

  tryCatch({
    probs <- predict(model, newdata = test_data, type = "probs", na.action = na.pass)
    levels_num <- as.numeric(colnames(probs))
    as.vector(probs %*% levels_num)
  }, error = function(e) {
    warning("Prediction failed: ", e$message)
    rep(NA_real_, nrow(test_data))
  })
}

train_logit <- prepare_common_data(train_df, ordered_wta = FALSE)
test_logit <- prepare_common_data(test_df, ordered_wta = FALSE)

train_car <- subset(train_logit, publictrans < 5)
train_elec <- subset(train_logit, conditionernumber == 1)
train_green <- subset(train_logit, energy_consume2020 > 1000)

model_demo_car <- fit_glm_logit("y_car", base_demos, "caruse", train_car)
model_demo_elec <- fit_glm_logit("y_elec", base_demos, "conditioner1month", train_elec)
model_demo_green <- fit_glm_logit("y_green", base_demos, "mainuseelec", train_green)
model_all_car <- fit_glm_logit("y_car", base_all, "caruse", train_car)
model_all_elec <- fit_glm_logit("y_elec", base_all, "conditioner1month", train_elec)
model_all_green <- fit_glm_logit("y_green", base_all, "mainuseelec", train_green)

logit_probs_demos <- data.frame(
  prob_car = predict(model_demo_car, newdata = test_logit, type = "response"),
  prob_elec = predict(model_demo_elec, newdata = test_logit, type = "response"),
  prob_green = predict(model_demo_green, newdata = test_logit, type = "response")
)
logit_probs_all <- data.frame(
  prob_car = predict(model_all_car, newdata = test_logit, type = "response"),
  prob_elec = predict(model_all_elec, newdata = test_logit, type = "response"),
  prob_green = predict(model_all_green, newdata = test_logit, type = "response")
)

write.csv(logit_probs_demos, file.path(sim_dir, "logit_probs_demos_simulated.csv"), row.names = FALSE)
write.csv(logit_probs_all, file.path(sim_dir, "logit_probs_all_simulated.csv"), row.names = FALSE)

train_wta <- prepare_common_data(train_df, ordered_wta = TRUE)
test_wta <- prepare_common_data(test_df, ordered_wta = TRUE)
train_wta_car <- subset(train_wta, publictrans < 5)
train_wta_elec <- subset(train_wta, conditionernumber == 1)
train_wta_green <- subset(train_wta, energy_consume2020 > 1000)

model_demo_car_wta <- fit_polr_wta("wta_car", base_demos, "caruse", train_wta_car)
model_demo_elec_wta <- fit_polr_wta("wta_elec", base_demos, "conditioner1month", train_wta_elec)
model_demo_green_wta <- fit_polr_wta("wta_green", base_demos, "mainuseelec", train_wta_green)
model_all_car_wta <- fit_polr_wta("wta_car", base_all, "caruse", train_wta_car)
model_all_elec_wta <- fit_polr_wta("wta_elec", base_all, "conditioner1month", train_wta_elec)
model_all_green_wta <- fit_polr_wta("wta_green", base_all, "mainuseelec", train_wta_green)

wta_preds_demos <- data.frame(
  pred_wta_car = calc_expected_wta_safe(model_demo_car_wta, train_wta_car, test_wta),
  pred_wta_elec = calc_expected_wta_safe(model_demo_elec_wta, train_wta_elec, test_wta),
  pred_wta_green = calc_expected_wta_safe(model_demo_green_wta, train_wta_green, test_wta)
)
wta_preds_all <- data.frame(
  pred_wta_car = calc_expected_wta_safe(model_all_car_wta, train_wta_car, test_wta),
  pred_wta_elec = calc_expected_wta_safe(model_all_elec_wta, train_wta_elec, test_wta),
  pred_wta_green = calc_expected_wta_safe(model_all_green_wta, train_wta_green, test_wta)
)

write.csv(wta_preds_demos, file.path(sim_dir, "wta_preds_demos_simulated.csv"), row.names = FALSE)
write.csv(wta_preds_all, file.path(sim_dir, "wta_preds_all_simulated.csv"), row.names = FALSE)

message("Simulated logit and ordered-logit predictions written to: ", sim_dir)
