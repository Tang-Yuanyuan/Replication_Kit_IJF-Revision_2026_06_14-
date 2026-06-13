utils::globalVariables(c("paths", "analysis_subsets"))

knowledge_vars <- c(
  "heard_about_global_warming",
  "know_about_low_carbon",
  "know_about_carbon_neutrality",
  "know_about_carbon_policy"
)

knowledge_labels <- c("Global Warming", "Low Carbon", "Neutrality", "Policy")

make_model_formula <- function(outcome, controls, exposure, knowledge) {
  stats::reformulate(c(controls, exposure, knowledge), response = outcome)
}

fit_polr_set <- function(data, outcome, controls, exposure) {
  models <- vector("list", length(knowledge_vars))
  names(models) <- c("global_warming", "low_carbon", "neutrality", "policy")

  for (i in seq_along(knowledge_vars)) {
    f <- make_model_formula(outcome, controls, exposure, knowledge_vars[[i]])
    m <- MASS::polr(f, data = data, Hess = TRUE)
    # brant() rebuilds a stats::model.frame() call from model$call and eval()s it
    # in the caller's environment. Store evaluated formula and data directly so
    # brant doesn't try to look up local variable names that no longer exist.
    m$call$formula <- f
    m$call$data    <- data
    models[[i]] <- m
  }

  models
}

fit_main_models <- function(prepared) {
  df <- prepared$data
  subsets <- analysis_subsets(df)

  list(
    car = fit_polr_set(subsets$car, "wta_car", prepared$base_controls, "caruse"),
    elec = fit_polr_set(subsets$elec, "wta_elec", prepared$base_controls, "conditioner1month"),
    green = fit_polr_set(subsets$green, "wta_green", prepared$base_controls, "mainuseelec")
  )
}

ic_lines <- function(models) {
  list(
    c("AIC", round(vapply(models, AIC, numeric(1)), 2)),
    c("BIC", round(vapply(models, BIC, numeric(1)), 2))
  )
}

export_stargazer_models <- function(models, file, keep = NULL, omit = NULL,
                                    type = "latex", title = NULL) {
  table <- build_model_table(models, keep = keep, omit = omit)
  csv_file <- sub("\\.tex$", ".csv", file)
  write.csv(table, csv_file, row.names = FALSE, fileEncoding = "UTF-8")
  write_latex_table(table, file, title = title)
  invisible(file)
}

build_model_table <- function(models, keep = NULL, omit = NULL) {
  labels <- c("Global Warming", "Low Carbon", "Neutrality", "Policy")
  tidy_models <- lapply(models, tidy_ordinal_model)
  all_terms <- unique(unlist(lapply(tidy_models, `[[`, "term")))

  if (!is.null(keep)) {
    keep_pattern <- paste(keep, collapse = "|")
    display_terms <- all_terms[grepl(keep_pattern, all_terms)]
  } else {
    display_terms <- all_terms
  }

  if (!is.null(omit)) {
    omit_pattern <- paste(omit, collapse = "|")
    display_terms <- display_terms[!grepl(omit_pattern, display_terms)]
  }

  table <- data.frame(Variable = display_terms, stringsAsFactors = FALSE)
  for (i in seq_along(tidy_models)) {
    res <- tidy_models[[i]]
    values <- character(length(display_terms))
    for (j in seq_along(display_terms)) {
      idx <- which(res$term == display_terms[[j]])
      values[[j]] <- if (length(idx) > 0) {
        format_coef(res$coef[[idx]], res$se[[idx]], res$statistic[[idx]])
      } else {
        ""
      }
    }
    table[[labels[[i]]]] <- values
  }

  names(table) <- c("Variable", labels)

  aic_row <- as.data.frame(as.list(c("AIC", round(vapply(models, AIC, numeric(1)), 2))),
                           stringsAsFactors = FALSE)
  bic_row <- as.data.frame(as.list(c("BIC", round(vapply(models, BIC, numeric(1)), 2))),
                           stringsAsFactors = FALSE)
  names(aic_row) <- names(table)
  names(bic_row) <- names(table)

  table <- rbind(table, aic_row, bic_row)
  table
}

escape_latex <- function(x) {
  x <- gsub("\\\\", "\\\\textbackslash{}", x)
  x <- gsub("([#$%&_{}])", "\\\\\\1", x, perl = TRUE)
  x
}

write_latex_table <- function(table, file, title = NULL) {
  align <- paste0("l", paste(rep("c", ncol(table) - 1), collapse = ""))
  header <- paste(escape_latex(names(table)), collapse = " & ")
  rows <- apply(table, 1, function(row) {
    paste(escape_latex(as.character(row)), collapse = " & ")
  })

  lines <- c(
    "\\begin{table}[!htbp]",
    "\\centering",
    if (!is.null(title)) paste0("\\caption{", escape_latex(title), "}") else NULL,
    paste0("\\begin{tabular}{", align, "}"),
    "\\hline",
    paste0(header, " \\\\"),
    "\\hline",
    paste0(rows, " \\\\"),
    "\\hline",
    "\\end{tabular}",
    "\\end{table}"
  )

  writeLines(lines, con = file, useBytes = TRUE)
}

export_main_model_tables <- function(models, prepared) {
  omit_controls <- c("province", "weekday", prepared$weather_control, "cityanswer")

  export_stargazer_models(
    models$car,
    file.path(paths$tables, "Table_3_main_transport.tex"),
    keep = c("^location", "^heard_about_global_warming", "^know_about_low_carbon",
             "^know_about_carbon_neutrality", "^know_about_carbon_policy"),
    omit = omit_controls,
    title = "Table 3. Transportation WTA"
  )

  export_stargazer_models(
    models$elec,
    file.path(paths$tables, "Table_4_main_home_energy.tex"),
    keep = c("female", "is_bachelor", "^know_about_low_carbon",
             "^know_about_carbon_neutrality", "^know_about_carbon_policy"),
    omit = omit_controls,
    title = "Table 4. Home Energy WTA"
  )

  export_stargazer_models(
    models$green,
    file.path(paths$tables, "Table_5_main_green_electricity.tex"),
    keep = c("^location", "income_level", "^know_about_carbon_neutrality",
             "^know_about_carbon_policy"),
    omit = omit_controls,
    title = "Table 5. Green Electricity WTA"
  )
}

capture_to_file <- function(expr, file) {
  con <- file(file, open = "wt", encoding = "UTF-8")
  on.exit(close(con), add = TRUE)
  utils::capture.output(expr, file = con)
}

run_brant_e6 <- function(models, prepared) {
  df      <- prepared$data
  subsets <- analysis_subsets(df)

  # car2, green3, green4: drop province/weekday/weather before running brant
  # because brant fits many internal binary logits and convergence is unreliable
  # when factor fixed effects add too many columns with sparse cells.
  no_fe_controls <- setdiff(
    prepared$base_controls,
    c("province", "weekday", prepared$weather_control)
  )

  fit_no_fe <- function(outcome, controls, exposure, knowledge, data) {
    f <- make_model_formula(outcome, controls, exposure, knowledge)
    m <- MASS::polr(f, data = data, Hess = TRUE)
    m$call$formula <- f
    m$call$data    <- data
    m
  }

  car2_nfe   <- fit_no_fe("wta_car",   no_fe_controls, "caruse",
                          "know_about_low_carbon",        subsets$car)
  green3_nfe <- fit_no_fe("wta_green", no_fe_controls, "mainuseelec",
                          "know_about_carbon_neutrality", subsets$green)
  green4_nfe <- fit_no_fe("wta_green", no_fe_controls, "mainuseelec",
                          "know_about_carbon_policy",     subsets$green)

  omnibus <- function(model) {
    res <- tryCatch(
      withCallingHandlers(
        brant::brant(model),
        warning = function(w) invokeRestart("muffleWarning")
      ),
      error = function(e) {
        warning("brant error: ", e$message)
        NULL
      }
    )
    if (is.null(res) || !is.matrix(res) || nrow(res) == 0) {
      return(c(Chi2 = NA_real_, DF = NA_integer_, P_value = NA_real_))
    }
    c(Chi2 = res[1, 1], DF = res[1, 2], P_value = res[1, 3])
  }

  model_list <- list(
    models$car[["global_warming"]],
    car2_nfe,
    models$car[["neutrality"]],
    models$car[["policy"]],
    models$elec[["global_warming"]],
    models$elec[["low_carbon"]],
    models$elec[["neutrality"]],
    models$elec[["policy"]],
    models$green[["global_warming"]],
    models$green[["low_carbon"]],
    green3_nfe,
    green4_nfe
  )

  rows <- lapply(model_list, omnibus)
  result <- as.data.frame(do.call(rbind, rows))
  result$Model <- paste0("(", seq_len(nrow(result)), ")")
  result <- result[, c("Model", "Chi2", "DF", "P_value")]

  csv_path <- file.path(paths$tables, "Table_E.6_brant_tests.csv")
  write.csv(result, csv_path, row.names = FALSE, fileEncoding = "UTF-8")

  tex_path <- file.path(paths$tables, "Table_E.6_brant_tests.tex")
  write_latex_table(result, tex_path, title = "Table E.6. Brant Test (Omnibus)")
}

tidy_ordinal_model <- function(model) {
  s <- coef(summary(model))
  is_threshold <- grepl("\\|", rownames(s))
  s_coef <- s[!is_threshold, , drop = FALSE]
  estimate_col <- if ("Value" %in% colnames(s_coef)) "Value" else "Estimate"

  data.frame(
    term = rownames(s_coef),
    coef = as.numeric(s_coef[, estimate_col]),
    se = as.numeric(s_coef[, "Std. Error"]),
    statistic = as.numeric(s_coef[, ncol(s_coef)]),
    stringsAsFactors = FALSE
  )
}

format_coef <- function(coef, se, statistic) {
  stars <- ifelse(
    abs(statistic) > 2.576, "***",
    ifelse(abs(statistic) > 1.960, "**", ifelse(abs(statistic) > 1.645, "*", ""))
  )
  sprintf("%.3f%s (%.3f)", coef, stars, se)
}

export_compact_results <- function(model_list, group_name, keep_pattern, file) {
  labels <- c("GW", "LC", "Neu", "Pol")
  all_terms <- unique(unlist(lapply(model_list, function(x) tidy_ordinal_model(x)$term)))
  display_terms <- all_terms[grepl(keep_pattern, all_terms)]

  if (length(display_terms) == 0) {
    stop("No variables matched keep pattern for ", group_name, call. = FALSE)
  }

  final_tab <- data.frame(Variable = display_terms, stringsAsFactors = FALSE)
  for (i in seq_along(model_list)) {
    res <- tidy_ordinal_model(model_list[[i]])
    values <- character(length(display_terms))
    for (j in seq_along(display_terms)) {
      idx <- which(res$term == display_terms[[j]])
      values[[j]] <- if (length(idx) > 0) {
        format_coef(res$coef[[idx]], res$se[[idx]], res$statistic[[idx]])
      } else {
        ""
      }
    }
    final_tab[[labels[[i]]]] <- values
  }

  add_stat_row <- function(label, fun) {
    vals <- vapply(model_list, function(m) as.character(fun(m)), character(1))
    row <- as.data.frame(as.list(c(label, vals)), stringsAsFactors = FALSE)
    names(row) <- names(final_tab)
    row
  }

  final_tab <- rbind(
    final_tab,
    add_stat_row("N",   function(m) length(m$fitted.values)),
    add_stat_row("AIC", function(m) round(AIC(m), 2)),
    add_stat_row("BIC", function(m) round(BIC(m), 2))
  )

  write.csv(final_tab, file = file, row.names = FALSE, fileEncoding = "UTF-8")
  tex_file <- sub("\\.csv$", ".tex", file)
  if (!identical(tex_file, file)) {
    write_latex_table(final_tab, tex_file, title = group_name)
  }
  final_tab
}

build_combined_model_table <- function(model_sets, set_labels, keep = NULL, omit = NULL) {
  short_labels <- c("GW", "LC", "Neu", "Pol")
  tidy_sets <- lapply(model_sets, function(model_list) lapply(model_list, tidy_ordinal_model))
  all_terms <- unique(unlist(lapply(tidy_sets, function(set) {
    unlist(lapply(set, `[[`, "term"))
  })))

  if (!is.null(keep)) {
    keep_pattern <- paste(keep, collapse = "|")
    display_terms <- all_terms[grepl(keep_pattern, all_terms)]
  } else {
    display_terms <- all_terms
  }

  if (!is.null(omit)) {
    omit_pattern <- paste(omit, collapse = "|")
    display_terms <- display_terms[!grepl(omit_pattern, display_terms)]
  }

  table <- data.frame(Variable = display_terms, stringsAsFactors = FALSE)

  for (set_i in seq_along(tidy_sets)) {
    for (model_i in seq_along(tidy_sets[[set_i]])) {
      res <- tidy_sets[[set_i]][[model_i]]
      values <- character(length(display_terms))
      for (row_i in seq_along(display_terms)) {
        idx <- which(res$term == display_terms[[row_i]])
        values[[row_i]] <- if (length(idx) > 0) {
          format_coef(res$coef[[idx]], res$se[[idx]], res$statistic[[idx]])
        } else {
          ""
        }
      }
      table[[paste(set_labels[[set_i]], short_labels[[model_i]], sep = ": ")]] <- values
    }
  }

  add_ic_row <- function(label, fun) {
    values <- c(label)
    for (model_set in model_sets) {
      values <- c(values, round(vapply(model_set, fun, numeric(1)), 2))
    }
    row <- as.data.frame(as.list(values), stringsAsFactors = FALSE)
    names(row) <- names(table)
    row
  }

  rbind(table, add_ic_row("AIC", AIC), add_ic_row("BIC", BIC))
}

export_combined_model_table <- function(model_sets, set_labels, file,
                                        keep = NULL, omit = NULL, title = NULL) {
  table <- build_combined_model_table(model_sets, set_labels, keep = keep, omit = omit)
  csv_file <- sub("\\.tex$", ".csv", file)
  write.csv(table, csv_file, row.names = FALSE, fileEncoding = "UTF-8")
  write_latex_table(table, file, title = title)
  invisible(table)
}
