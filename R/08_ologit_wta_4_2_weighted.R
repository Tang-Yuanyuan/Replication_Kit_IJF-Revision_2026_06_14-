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

prepare_wta_data <- function(data) {
  wta_levels <- as.character(1:7)
  data$wta_car   <- factor(data$wta_car,   levels = wta_levels, ordered = TRUE)
  data$wta_elec  <- factor(data$wta_elec,  levels = wta_levels, ordered = TRUE)
  data$wta_green <- factor(data$wta_green, levels = wta_levels, ordered = TRUE)

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

train_df <- prepare_wta_data(train_df)
test_df  <- prepare_wta_data(test_df)

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

fit_polr_wta_weighted <- function(outcome, controls, exposure, data) {
  MASS::polr(
    stats::reformulate(c(strsplit(controls, " \\+ ")[[1]], exposure), response = outcome),
    data    = data,
    Hess    = TRUE,
    weights = data$weights
  )
}

model_demo_car   <- fit_polr_wta_weighted("wta_car",   base_demos, "caruse",            train_car)
model_demo_elec  <- fit_polr_wta_weighted("wta_elec",  base_demos, "conditioner1month", train_elec)
model_demo_green <- fit_polr_wta_weighted("wta_green", base_demos, "mainuseelec",       train_green)
model_all_car    <- fit_polr_wta_weighted("wta_car",   base_all,  "caruse",            train_car)
model_all_elec   <- fit_polr_wta_weighted("wta_elec",  base_all,  "conditioner1month", train_elec)
model_all_green  <- fit_polr_wta_weighted("wta_green", base_all,  "mainuseelec",       train_green)

wta_preds_demos <- data.frame(
  pred_wta_car   = calc_expected_wta_safe(model_demo_car,   train_car,   test_df),
  pred_wta_elec  = calc_expected_wta_safe(model_demo_elec,  train_elec,  test_df),
  pred_wta_green = calc_expected_wta_safe(model_demo_green, train_green, test_df)
)
wta_preds_all <- data.frame(
  pred_wta_car   = calc_expected_wta_safe(model_all_car,   train_car,   test_df),
  pred_wta_elec  = calc_expected_wta_safe(model_all_elec,  train_elec,  test_df),
  pred_wta_green = calc_expected_wta_safe(model_all_green, train_green, test_df)
)

write.csv(wta_preds_demos, file.path(out_dir, "wta_preds_demos.csv"), row.names = FALSE)
write.csv(wta_preds_all,   file.path(out_dir, "wta_preds_all.csv"),   row.names = FALSE)

message("Weighted ordered-logit WTA predictions written to: ", out_dir)
