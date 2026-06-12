args <- commandArgs(trailingOnly = FALSE)
file_arg <- grep("^--file=", args, value = TRUE)

if (length(file_arg) > 0) {
  project_root <- dirname(normalizePath(
    sub("^--file=", "", file_arg[[1]]),
    winslash = "/"
  ))
} else {
  project_root <- normalizePath(getwd(), winslash = "/", mustWork = TRUE)
}

setwd(project_root)

source(file.path(project_root, "R", "00_setup.R"), encoding = "UTF-8")
source(file.path(project_root, "R", "01_data_functions.R"), encoding = "UTF-8")
source(file.path(project_root, "R", "02_model_functions.R"), encoding = "UTF-8")
source(file.path(project_root, "R", "03_descriptives.R"), encoding = "UTF-8")
source(file.path(project_root, "R", "04_figures.R"), encoding = "UTF-8")
source(file.path(project_root, "R", "05_robustness.R"), encoding = "UTF-8")
source(file.path(project_root, "R", "06_weighted.R"), encoding = "UTF-8")

message("Running empirical Section 4.1 workflow...")

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

writeLines(capture.output(sessionInfo()),
           con = file.path(paths$logs, "session_info.txt"),
           useBytes = TRUE)

message("Done. Empirical Section 4.1 results are in: ", paths$empirical4_1)
