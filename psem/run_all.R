# Run the complete PSEM analysis from any working directory.

args_all <- commandArgs(trailingOnly = FALSE)
file_arg <- grep("^--file=", args_all, value = TRUE)
project_root <- if (length(file_arg)) {
  dirname(normalizePath(sub("^--file=", "", file_arg[1]), winslash = "/"))
} else {
  normalizePath(getwd(), winslash = "/")
}
setwd(project_root)

args <- commandArgs(trailingOnly = TRUE)
mem_arg <- grep("^--mem-filter=", args, value = TRUE)
mem_mode <- if (length(mem_arg)) sub("^--mem-filter=", "", mem_arg[1]) else "published"
if (!mem_mode %in% c("published", "reselected")) {
  stop("--mem-filter must be 'published' or 'reselected'.")
}
options(psem.mem_filter = mem_mode)

required_packages <- c(
  "readxl", "dplyr", "lme4", "lmerTest", "nlme", "piecewiseSEM", "spdep"
)
missing_packages <- required_packages[
  !vapply(required_packages, requireNamespace, logical(1), quietly = TRUE)
]
if (length(missing_packages)) {
  stop(
    "Missing R packages: ", paste(missing_packages, collapse = ", "),
    ". Run: Rscript install_packages.R"
  )
}

dir.create("results", recursive = TRUE, showWarnings = FALSE)
log_file <- file("results/run_all.log", open = "wt", encoding = "UTF-8")
sink(log_file, type = "output", split = TRUE)
sink(log_file, type = "message")
on.exit({
  sink(type = "message")
  sink(type = "output")
  close(log_file)
}, add = TRUE)

scripts <- c("R/01_base_psem.R", "R/02_ar1_moran.R")
if (mem_mode == "reselected") scripts <- c(scripts, "R/03a_select_mem.R")
scripts <- c(
  scripts,
  "R/03_mem_rsei.R",
  "R/04_compare_robustness.R",
  "R/05_export_path_lists.R"
)

for (script in scripts) {
  message("\n===== Running ", script, " =====")
  source(script, local = globalenv(), encoding = "UTF-8")
}

capture.output(sessionInfo(), file = "results/session_info.txt")
message("\nAll analyses completed. Results: ", normalizePath("results", winslash = "/"))

