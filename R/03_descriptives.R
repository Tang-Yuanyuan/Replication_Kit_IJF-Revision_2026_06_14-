run_descriptive_tables <- function(df) {
  continuous_vars <- c(
    "income_level", "living_area", "age", "conditioner1month",
    "female", "youth", "older_adults", "ifpollution", "is_bachelor",
    "married", "partymember", "caruse", "mainuseelec"
  )

  continuous_data <- df %>%
    select(any_of(continuous_vars)) %>%
    mutate(across(everything(), ~ as.numeric(as.character(.))))

  continuous_summary <- data.frame(
    Variable = names(continuous_data),
    N = vapply(continuous_data, function(x) sum(!is.na(x)), integer(1)),
    Mean = vapply(continuous_data, function(x) mean(x, na.rm = TRUE), numeric(1)),
    SD = vapply(continuous_data, function(x) stats::sd(x, na.rm = TRUE), numeric(1)),
    Min = vapply(continuous_data, function(x) min(x, na.rm = TRUE), numeric(1)),
    Max = vapply(continuous_data, function(x) max(x, na.rm = TRUE), numeric(1)),
    row.names = NULL
  )
  continuous_summary[, c("Mean", "SD", "Min", "Max")] <-
    round(continuous_summary[, c("Mean", "SD", "Min", "Max")], 3)

  write.csv(
    continuous_summary,
    file = file.path(paths$tables, "Table1_continuous.csv"),
    row.names = FALSE,
    fileEncoding = "UTF-8"
  )

  categorical_vars <- c(
    "location", "province", "education", "weather", "ifsunny",
    "heard_about_global_warming", "know_about_carbon_neutrality",
    "know_about_carbon_policy", "know_about_low_carbon", "weekday"
  )

  categorical_data <- df %>% select(any_of(categorical_vars))
  categorical_summary <- do.call(
    rbind,
    lapply(names(categorical_data), function(var) {
      tab <- table(categorical_data[[var]], useNA = "ifany")
      data.frame(
        Variable = var,
        Category = names(tab),
        N = as.integer(tab),
        Percent = round(100 * as.numeric(tab) / sum(tab), 3),
        row.names = NULL
      )
    })
  )

  write.csv(
    categorical_summary,
    file = file.path(paths$tables, "Table1_categorical.csv"),
    row.names = FALSE,
    fileEncoding = "UTF-8"
  )

  table2_data <- rbind(
    transportation = prop.table(table(df$wta_car)),
    home_energy = prop.table(table(df$wta_elec)),
    green_electricity = prop.table(table(df$wta_green))
  )

  write.csv(
    table2_data,
    file = file.path(paths$tables, "Table2_wta_distribution.csv"),
    row.names = TRUE,
    fileEncoding = "UTF-8"
  )
}
