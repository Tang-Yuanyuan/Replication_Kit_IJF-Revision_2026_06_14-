options(stringsAsFactors = FALSE)

if (!exists("PROJECT_ROOT", inherits = TRUE)) {
  PROJECT_ROOT <- normalizePath(getwd(), winslash = "/", mustWork = TRUE)
}

EMPIRICAL_SECTION <- "empirical4.1"

paths <- list(
  root = PROJECT_ROOT,
  data = file.path(PROJECT_ROOT, "data"),
  raw_data = file.path(PROJECT_ROOT, "data", "raw", "energy_wta.csv"),
  temp_data = file.path(PROJECT_ROOT, "data", "temp"),
  results = file.path(PROJECT_ROOT, "results"),
  empirical4_1 = file.path(PROJECT_ROOT, "results", EMPIRICAL_SECTION),
  tables = file.path(PROJECT_ROOT, "results", EMPIRICAL_SECTION, "tables"),
  figures = file.path(PROJECT_ROOT, "results", EMPIRICAL_SECTION, "figures"),
  logs = file.path(PROJECT_ROOT, "results", EMPIRICAL_SECTION, "logs")
)

invisible(lapply(
  paths[c("data", "temp_data", "results", "empirical4_1", "tables", "figures", "logs")],
  dir.create,
  recursive = TRUE,
  showWarnings = FALSE
))

custom_lib <- Sys.getenv("R_PACKAGE_DIR", unset = NA_character_)
if (!is.na(custom_lib) && dir.exists(custom_lib)) {
  .libPaths(unique(c(custom_lib, .libPaths())))
}

required_packages <- c(
  "MASS", "dplyr", "brant", "survey", "ggplot2"
)

missing_packages <- required_packages[
  !vapply(required_packages, requireNamespace, logical(1), quietly = TRUE)
]

if (length(missing_packages) > 0) {
  message("Installing missing R packages: ", paste(missing_packages, collapse = ", "))
  install.packages(missing_packages, repos = "https://cloud.r-project.org", quiet = TRUE)
  still_missing <- missing_packages[
    !vapply(missing_packages, requireNamespace, logical(1), quietly = TRUE)
  ]
  if (length(still_missing) > 0) {
    stop(
      "Failed to install R packages: ",
      paste(still_missing, collapse = ", "),
      call. = FALSE
    )
  }
}

invisible(lapply(required_packages, library, character.only = TRUE))

theme_set(ggplot2::theme_classic())
