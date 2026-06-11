get_plot_data <- function(model, var_prefix, label) {
  coef_table <- as.data.frame(summary(model)$coefficients)
  coef_table$term <- rownames(coef_table)

  coef_data <- coef_table %>%
    filter(grepl(var_prefix, term)) %>%
    mutate(
      term = gsub(var_prefix, "", term),
      estimate = Value,
      std_error = `Std. Error`,
      conf.low = estimate - 1.96 * std_error,
      conf.high = estimate + 1.96 * std_error
    ) %>%
    select(term, estimate, conf.low, conf.high)

  base_row <- data.frame(term = "never", estimate = 0, conf.low = 0, conf.high = 0)
  coef_data <- rbind(base_row, coef_data)
  coef_data$term <- factor(
    coef_data$term,
    levels = c("never", "heard but do not know", "heard and know", "familiar")
  )
  coef_data$group <- label
  coef_data
}

plot_knowledge_pattern <- function(plot_df, file) {
  p <- ggplot(plot_df, aes(x = term, y = estimate, color = group,
                           group = group, shape = group)) +
    geom_hline(yintercept = 0, linetype = "dashed", color = "gray50") +
    geom_line(position = position_dodge(0.2), linewidth = 1) +
    geom_errorbar(
      aes(ymin = conf.low, ymax = conf.high),
      width = 0.1,
      linewidth = 0.8,
      position = position_dodge(0.2)
    ) +
    geom_point(size = 4, position = position_dodge(0.2)) +
    scale_color_manual(values = c("black", "gray40", "gray70")) +
    labs(x = NULL, y = "Marginal Effect") +
    theme(
      legend.position = "bottom",
      axis.text.x = element_text(angle = 15, hjust = 1)
    )

  ggsave(filename = file, plot = p, width = 8, height = 6, dpi = 300)
  invisible(p)
}

run_figures <- function(main_models) {
  neutrality_data <- rbind(
    get_plot_data(main_models$car$neutrality, "know_about_carbon_neutrality", "WTA Transportation"),
    get_plot_data(main_models$elec$neutrality, "know_about_carbon_neutrality", "WTA Home Energy"),
    get_plot_data(main_models$green$neutrality, "know_about_carbon_neutrality", "WTA Green Electricity")
  )

  policy_data <- rbind(
    get_plot_data(main_models$car$policy, "know_about_carbon_policy", "WTA Transportation"),
    get_plot_data(main_models$elec$policy, "know_about_carbon_policy", "WTA Home Energy"),
    get_plot_data(main_models$green$policy, "know_about_carbon_policy", "WTA Green Electricity")
  )

  plot_knowledge_pattern(
    neutrality_data,
    file.path(paths$figures, "F_B.1.png")
  )
  plot_knowledge_pattern(
    policy_data,
    file.path(paths$figures, "F_B.2.png")
  )
}
