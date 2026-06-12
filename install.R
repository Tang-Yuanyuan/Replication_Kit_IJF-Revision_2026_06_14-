# Install all R packages required for the reproduction workflow.
# Tested versions are listed for each package.
# Run this script once before running run_all.R:
#   Rscript install.R

if (!requireNamespace("remotes", quietly = TRUE)) {
  install.packages("remotes", repos = "https://cloud.r-project.org")
}

.install <- function(pkg, version) {
  installed_ver <- tryCatch(
    as.character(packageVersion(pkg)),
    error = function(e) NA_character_
  )
  if (!is.na(installed_ver) && installed_ver == version) {
    message(sprintf("  [ok] %s %s", pkg, version))
    return(invisible(NULL))
  }
  if (!is.na(installed_ver)) {
    message(sprintf("  [update] %s: installed %s, target %s", pkg, installed_ver, version))
  } else {
    message(sprintf("  [install] %s %s", pkg, version))
  }
  remotes::install_version(
    pkg,
    version = version,
    repos   = "https://cloud.r-project.org",
    quiet   = TRUE,
    upgrade = "never"
  )
}

message("Installing R packages for reproduction workflow...\n")

# ── Direct dependencies ────────────────────────────────────────────────────
.install("MASS",    "7.3-65")
.install("dplyr",   "1.2.0")
.install("brant",   "0.3-0")
.install("survey",  "4.5")
.install("ggplot2", "4.0.3")

# ── Transitive: dplyr ─────────────────────────────────────────────────────
.install("cli",        "3.6.5")
.install("generics",   "0.1.4")
.install("glue",       "1.8.0")
.install("lifecycle",  "1.0.5")
.install("magrittr",   "2.0.4")
.install("pillar",     "1.11.1")
.install("pkgconfig",  "2.0.3")
.install("R6",         "2.6.1")
.install("rlang",      "1.1.7")
.install("tibble",     "3.3.1")
.install("tidyselect", "1.2.1")
.install("utf8",       "1.2.6")
.install("vctrs",      "0.7.1")
.install("withr",      "3.0.2")

# ── Transitive: ggplot2 ───────────────────────────────────────────────────
.install("farver",      "2.1.2")
.install("gtable",      "0.3.6")
.install("isoband",     "0.3.0")
.install("labeling",    "0.4.3")
.install("RColorBrewer","1.1-3")
.install("scales",      "1.4.0")
.install("viridisLite", "0.4.3")

# ── Transitive: survey ────────────────────────────────────────────────────
.install("DBI",           "1.3.0")
.install("Matrix",        "1.7-4")
.install("minqa",         "1.2.8")
.install("mitools",       "2.4")
.install("numDeriv",      "2016.8-1.1")
.install("Rcpp",          "1.1.1")
.install("RcppArmadillo", "15.2.6-1")
.install("S7",            "0.2.1")
.install("survival",      "3.8-3")

# ── Transitive: misc ──────────────────────────────────────────────────────
.install("cpp11",   "0.5.4")
.install("lattice", "0.22-7")

message("\nDone. Run 'Rscript run_all.R' to start the workflow.")
