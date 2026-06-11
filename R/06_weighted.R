create_post_stratification_weights <- function(df) {
  df_clean <- df %>%
    mutate(
      edu_strat = case_when(
        education == "uneducated" ~ "Uneducated",
        education == "primary" ~ "Primary",
        education == "junior high" ~ "Junior high",
        education == "senior high" ~ "Senior high",
        education %in% c("associate", "bachelor", "postgraduate") ~ "College or higher",
        TRUE ~ NA_character_
      ),
      loc_strat = case_when(
        location %in% c("city", "county seat") ~ "Urban",
        location == "rural" ~ "Rural",
        TRUE ~ NA_character_
      ),
      gender_strat = factor(as.character(female), levels = c("0", "1")),
      loc_strat = factor(loc_strat, levels = c("Rural", "Urban")),
      edu_strat = factor(
        edu_strat,
        levels = c("Uneducated", "Primary", "Junior high", "Senior high", "College or higher")
      )
    ) %>%
    filter(!is.na(gender_strat), !is.na(loc_strat), !is.na(edu_strat), !is.na(age))

  n_clean <- nrow(df_clean)
  targets <- list(
    "(Intercept)" = n_clean,
    "gender_strat1" = n_clean * 0.488,
    "loc_stratUrban" = n_clean * 0.639,
    "edu_stratPrimary" = n_clean * 0.244,
    "edu_stratJunior high" = n_clean * 0.345,
    "edu_stratSenior high" = n_clean * 0.151,
    "edu_stratCollege or higher" = n_clean * 0.155,
    "age" = n_clean * 38.8
  )

  mm <- model.matrix(~ gender_strat + loc_strat + edu_strat + age, data = df_clean)
  pop_totals <- numeric(ncol(mm))
  names(pop_totals) <- colnames(mm)

  for (name in names(pop_totals)) {
    if (name %in% names(targets)) {
      pop_totals[[name]] <- targets[[name]]
    } else {
      stop("Undefined post-stratification target: ", name, call. = FALSE)
    }
  }

  df_clean$base_weight <- 1
  ids <- survey::svydesign(ids = ~1, weights = ~base_weight, data = df_clean)
  cal_design <- survey::calibrate(
    design = ids,
    formula = ~ gender_strat + loc_strat + edu_strat + age,
    population = pop_totals,
    calfun = "raking",
    epsilon = 1e-7,
    maxit = 2000
  )

  raw_w <- stats::weights(cal_design)
  mean_w <- mean(raw_w)
  cal_trimmed <- survey::trimWeights(
    cal_design,
    lower = mean_w * 0.29,
    upper = mean_w * 9.5,
    strict = TRUE
  )

  df_clean$weights <- stats::weights(cal_trimmed)
  df_clean$weights <- df_clean$weights / mean(df_clean$weights)
  df$weights <- NA_real_
  df[rownames(df_clean), "weights"] <- df_clean$weights

  list(data = df, clean = df_clean)
}

target_balance_values <- function() {
  list(
    age = 38.8,
    female = 0.488,
    urban = 0.639,
    education = c(
    "Uneducated" = 0.027,
    "Primary" = 0.244,
    "Junior high" = 0.345,
    "Senior high" = 0.151,
    "College or higher" = 0.155
  )
  )
}

format_one_decimal <- function(x) {
  sprintf("%.1f", round(as.numeric(x), 1))
}

format_percent <- function(x) {
  paste0(format_one_decimal(100 * x), "%")
}

format_signed_percent <- function(x) {
  paste0(ifelse(x >= 0, "+", ""), format_one_decimal(100 * x), "%")
}

balance_values <- function(df_clean, weight_var = NULL) {
  targets <- target_balance_values()

  if (is.null(weight_var)) {
    age <- mean(df_clean$age, na.rm = TRUE)
    female <- mean(as.numeric(as.character(df_clean$gender_strat)) == 1, na.rm = TRUE)
    urban <- mean(df_clean$loc_strat == "Urban", na.rm = TRUE)
    edu <- prop.table(table(df_clean$edu_strat))
  } else {
    design <- survey::svydesign(
      ids = ~1,
      weights = stats::as.formula(paste0("~", weight_var)),
      data = df_clean
    )
    age <- as.numeric(coef(survey::svymean(~age, design)))
    female <- as.numeric(prop.table(survey::svytable(~gender_strat, design))["1"])
    urban <- as.numeric(prop.table(survey::svytable(~loc_strat, design))["Urban"])
    edu <- prop.table(survey::svytable(~edu_strat, design))
  }

  edu <- edu[names(targets$education)]

  data.frame(
    Variable = c("Age", "Gender", "Location", rep("Education", length(targets$education))),
    Category = c("Mean", "Female", "Urban", names(targets$education)),
    Survey = c(
      format_one_decimal(age),
      format_percent(female),
      format_percent(urban),
      format_percent(as.numeric(edu))
    ),
    Population = c(
      format_one_decimal(targets$age),
      format_percent(targets$female),
      format_percent(targets$urban),
      format_percent(targets$education)
    ),
    Difference = c(
      format_signed_percent((age - targets$age) / targets$age),
      format_signed_percent(female - targets$female),
      format_signed_percent(urban - targets$urban),
      format_signed_percent(as.numeric(edu) - targets$education)
    ),
    stringsAsFactors = FALSE
  )
}

write_balance_table <- function(balance, file, title) {
  write.csv(
    balance,
    file = file,
    row.names = FALSE,
    fileEncoding = "UTF-8"
  )
  write_latex_table(balance, sub("\\.csv$", ".tex", file), title = title)
  balance
}

write_weight_balance <- function(df_clean) {
  before <- balance_values(df_clean, weight_var = NULL)
  after <- balance_values(df_clean, weight_var = "weights")

  write_balance_table(
    before,
    file.path(paths$tables, "Table_G.10_unweighted_balance.csv"),
    "Table G.10. Differences Between Unweighted Survey Means and Population Targets"
  )
  write_balance_table(
    after,
    file.path(paths$tables, "Table_G.11_weighted_balance_test.csv"),
    "Table G.11. Differences Between Weighted Survey Means and Population Targets"
  )

  list(before = before, after = after)
}

fit_svyolr_set <- function(design, outcome, controls, exposure) {
  models <- vector("list", length(knowledge_vars))
  names(models) <- c("global_warming", "low_carbon", "neutrality", "policy")

  for (i in seq_along(knowledge_vars)) {
    models[[i]] <- survey::svyolr(
      make_model_formula(outcome, controls, exposure, knowledge_vars[[i]]),
      design = design
    )
  }

  models
}

run_weighted_analysis <- function(prepared) {
  weighted <- create_post_stratification_weights(prepared$data)
  df <- weighted$data
  write_weight_balance(weighted$clean)

  write.csv(
    df %>% select(any_of(c(
      "id", "wta_car", "wta_elec", "wta_green", prepared$base_controls,
      "caruse", "conditioner1month", "mainuseelec", knowledge_vars,
      "publictrans", "conditionernumber", "energy_consume2020", "weights"
    ))),
    file = file.path(paths$results, "energy_wta_with_post_weights.csv"),
    row.names = FALSE,
    fileEncoding = "UTF-8"
  )

  subsets <- analysis_subsets(df)
  designs <- list(
    car = survey::svydesign(ids = ~1, weights = ~weights, data = subsets$car),
    elec = survey::svydesign(ids = ~1, weights = ~weights, data = subsets$elec),
    green = survey::svydesign(ids = ~1, weights = ~weights, data = subsets$green)
  )

  models <- list(
    car = fit_svyolr_set(designs$car, "wta_car", prepared$base_controls, "caruse"),
    elec = fit_svyolr_set(designs$elec, "wta_elec", prepared$base_controls, "conditioner1month"),
    green = fit_svyolr_set(designs$green, "wta_green", prepared$base_controls, "mainuseelec")
  )

  export_compact_results(
    models$car,
    "Table G.12. Weighted Transportation WTA",
    "^location|^heard_about_global_warming|^know_about_low_carbon|^know_about_carbon_neutrality|^know_about_carbon_policy",
    file.path(paths$tables, "Table_G.12_weighted_transport.csv")
  )
  export_compact_results(
    models$elec,
    "Table G.13. Weighted Home Energy WTA",
    "female|is_bachelor|^know_about_low_carbon|^know_about_carbon_neutrality|^know_about_carbon_policy",
    file.path(paths$tables, "Table_G.13_weighted_home_energy.csv")
  )
  export_compact_results(
    models$green,
    "Table G.14. Weighted Green Electricity WTA",
    "^location|income_level|^know_about_carbon_neutrality|^know_about_carbon_policy",
    file.path(paths$tables, "Table_G.14_weighted_green_electricity.csv")
  )

  models
}
