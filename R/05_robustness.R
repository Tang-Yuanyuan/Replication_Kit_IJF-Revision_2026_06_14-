run_robustness_checks <- function(prepared) {
  df <- prepared$data
  subsets <- analysis_subsets(df)
  omit_controls <- c("province", "weekday", prepared$weather_control, "cityanswer")

  robust <- list()

  robust$car_carown <- fit_polr_set(
    subsets$car, "wta_car", prepared$base_controls, "carown"
  )
  robust$car_madt <- fit_polr_set(
    subsets$car, "wta_car", prepared$base_controls, "MADT"
  )
  robust$car_city <- fit_polr_set(
    subsets$car, "wta_car", prepared$robust_controls, "caruse"
  )
  robust$elec_time <- fit_polr_set(
    subsets$elec, "wta_elec", prepared$base_controls, "conditioner1time"
  )
  robust$green_importance <- fit_polr_set(
    subsets$green, "wta_green", prepared$base_controls, "impgreenele"
  )

  export_combined_model_table(
    model_sets = list(robust$car_madt, robust$car_carown, robust$car_city),
    set_labels = c("MADT", "Car ownership", "City controls"),
    file = file.path(paths$tables, "Table_F.7_transport_robustness.tex"),
    omit = omit_controls,
    title = "Table F.7. Robustness Checks for Transportation WTA"
  )
  export_stargazer_models(
    robust$elec_time,
    file.path(paths$tables, "Table_F.8_home_energy_conditioner_time.tex"),
    omit = omit_controls,
    title = "Table F.8. Robustness Check for Home Energy WTA"
  )
  export_stargazer_models(
    robust$green_importance,
    file.path(paths$tables, "Table_F.9_green_electricity_importance.tex"),
    omit = omit_controls,
    title = "Table F.9. Robustness Check for Green Electricity WTA"
  )

  robust
}
