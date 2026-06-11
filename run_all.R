args <- commandArgs(trailingOnly = FALSE)
file_arg <- grep("^--file=", args, value = TRUE)

if (length(file_arg) > 0) {
  PROJECT_ROOT <- dirname(normalizePath(sub("^--file=", "", file_arg[[1]]), winslash = "/"))
} else {
  PROJECT_ROOT <- normalizePath(getwd(), winslash = "/", mustWork = TRUE)
}

setwd(PROJECT_ROOT)

source(file.path(PROJECT_ROOT, "R", "00_setup.R"), encoding = "UTF-8")
source(file.path(PROJECT_ROOT, "R", "01_data_functions.R"), encoding = "UTF-8")
source(file.path(PROJECT_ROOT, "R", "02_model_functions.R"), encoding = "UTF-8")
source(file.path(PROJECT_ROOT, "R", "03_descriptives.R"), encoding = "UTF-8")
source(file.path(PROJECT_ROOT, "R", "04_figures.R"), encoding = "UTF-8")
source(file.path(PROJECT_ROOT, "R", "05_robustness.R"), encoding = "UTF-8")
source(file.path(PROJECT_ROOT, "R", "06_weighted.R"), encoding = "UTF-8")

message("Loading and preparing data...")
raw_data <- load_raw_data()
prepared <- prepare_analysis_data(raw_data)

message("Writing descriptive tables...")
run_descriptive_tables(prepared$data)

message("Estimating main ordered logit models...")
main_models <- fit_main_models(prepared)
export_main_model_tables(main_models, prepared)
run_brant_tests(main_models)

message("Drawing figures...")
run_figures(main_models)

message("Running robustness checks...")
robust_models <- run_robustness_checks(prepared)

message("Running post-stratified weighted models...")
weighted_models <- run_weighted_analysis(prepared)

writeLines(
  capture.output(sessionInfo()),
  con = file.path(paths$logs, "session_info.txt"),
  useBytes = TRUE
)

message("Done. Results are in: ", paths$results)
