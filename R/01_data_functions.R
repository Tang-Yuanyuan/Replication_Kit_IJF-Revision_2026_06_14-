standardize_column_names <- function(df) {
  names(df) <- trimws(names(df))
  df
}

factor_if_present <- function(df, vars) {
  for (var in intersect(vars, names(df))) {
    df[[var]] <- as.factor(df[[var]])
  }
  df
}

relevel_if_present <- function(df, var, ref) {
  if (var %in% names(df)) {
    df[[var]] <- relevel(as.factor(df[[var]]), ref = ref)
  }
  df
}

# nolint start: object_usage_linter.
load_raw_data <- function(file = paths$raw_data) {
  # nolint end
  df <- read.csv(file, check.names = FALSE, fileEncoding = "UTF-8")
  standardize_column_names(df)
}

prepare_analysis_data <- function(df) {
  df <- standardize_column_names(df)

  df$wta_car <- factor(df$wta_car, levels = 1:7, ordered = TRUE)
  df$wta_elec <- factor(df$wta_elec, levels = 1:7, ordered = TRUE)
  df$wta_green <- factor(df$wta_green, levels = 1:7, ordered = TRUE)

  factor_vars <- c(
    "education", "location", "marriage", "youth", "older_adults",
    "province", "weekday", "partymember", "weather", "ifsunny",
    "mainuseelec", "cityanswer", "heard_about_global_warming"
  )
  df <- factor_if_present(df, factor_vars)

  df <- relevel_if_present(df, "mainuseelec", "0")
  df <- relevel_if_present(df, "education", "uneducated")
  df <- relevel_if_present(df, "location", "city")
  df <- relevel_if_present(df, "marriage", "unmarried")
  df <- relevel_if_present(df, "youth", "0")
  df <- relevel_if_present(df, "older_adults", "0")
  df <- relevel_if_present(df, "weather", "0")
  df <- relevel_if_present(df, "ifsunny", "0")
  df <- relevel_if_present(df, "partymember", "0")
  df <- relevel_if_present(df, "heard_about_global_warming", "no")

  knowledge_levels <- c(
    "never", "heard but do not know", "heard and know", "familiar"
  )
  for (var in c(
    "know_about_low_carbon", "know_about_carbon_neutrality",
    "know_about_carbon_policy"
  )) {
    if (var %in% names(df)) {
      df[[var]] <- factor(df[[var]], levels = knowledge_levels, ordered = FALSE)
    }
  }

  df$ifpollution <- ifelse(df$aqi >= 101, 1, 0)
  df$married <- as.numeric(df$marriage == "married")
  df$age_ln <- log(df$age)
  df$living_area_ln <- log(df$living_area)
  df$is_bachelor <- ifelse(
    df$education %in% c("bachelor", "postgraduate"), 1, 0
  )
  df$caruse <- ifelse(df$carusetime == 0, 1, 0)

  weather_control <- if ("weather" %in% names(df)) "weather" else "ifsunny"

  list(
    data = df,
    weather_control = weather_control,
    base_controls = c(
      "ifpollution", "living_area_ln", "age_ln", "is_bachelor", "location",
      "female", "married", "income_level", "youth", "older_adults",
      "partymember", "province", "weekday", weather_control
    ),
    robust_controls = c(
      "ifpollution", "living_area_ln", "age_ln", "is_bachelor", "location",
      "female", "married", "income_level", "youth", "older_adults",
      "partymember", "cityanswer", "weekday", weather_control
    )
  )
}

analysis_subsets <- function(df) {
  list(
    car = df[df$publictrans < 5, ],
    elec = df[df$conditionernumber == 1, ],
    green = df[df$energy_consume2020 > 1000, ]
  )
}
