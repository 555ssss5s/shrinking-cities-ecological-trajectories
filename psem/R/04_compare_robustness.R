if (!file.exists("R/00_utils.R")) stop("Run this script from the repository root.")
source("R/00_utils.R", encoding = "UTF-8")

base_file <- file.path(project_paths$baseline, "PSEM_coefficients.csv")
ar1_file <- file.path(project_paths$ar1, "AR1_strict_coefficients.csv")
mem_file <- file.path(project_paths$mem, "RSEI_AR1_vs_MEM_paths.csv")
for (f in c(base_file, ar1_file, mem_file)) if (!file.exists(f)) stop("Missing result: ", f)

base <- read.csv(base_file, check.names = FALSE)
ar1 <- read.csv(ar1_file, check.names = FALSE)
mem <- read.csv(mem_file, check.names = FALSE)
core_terms <- c(
  "POP_log_z", "PD_z", "GPBE_log_z", "ELE_z", "SLO_z", "SR_z", "PRE_z",
  "TEMP_z", "year_z", "ISA_z", "CRO_z", "FOR_z", "GRA_z", "GCR_z", "PM25_z"
)

base_rsei <- base[base$Response == "RSEI_z" & base$Predictor %in% core_terms,
                  c("Predictor", "Estimate", "P.Value")]
names(base_rsei) <- c("predictor", "Baseline_beta", "Baseline_p")
ar1_rsei <- ar1[ar1$Response == "RSEI_z" & ar1$Predictor %in% core_terms,
                c("Predictor", "Estimate", "P.Value")]
names(ar1_rsei) <- c("predictor", "AR1_beta_check", "AR1_p_check")

out <- merge(base_rsei, ar1_rsei, by = "predictor", all = TRUE)
out <- merge(out, mem, by = "predictor", all = TRUE)
out <- out[match(core_terms, out$predictor), ]
out$term <- clean_term(out$predictor)
out$Baseline_AR1_direction_same <- sign(out$Baseline_beta) == sign(out$AR1_beta)
out$AR1_MEM_direction_same <- sign(out$AR1_beta) == sign(out$MEM_beta)
out$Baseline_significant <- out$Baseline_p < 0.05
out$AR1_significant <- out$AR1_p < 0.05
out$MEM_significant <- out$MEM_p < 0.05
out <- out[, c(
  "term", "predictor", "Baseline_beta", "Baseline_p", "AR1_beta", "AR1_p",
  "MEM_beta", "MEM_p", "Baseline_AR1_direction_same", "AR1_MEM_direction_same",
  "Baseline_significant", "AR1_significant", "MEM_significant"
)]
write_csv(out, file.path(project_paths$comparison, "RSEI_path_robustness_comparison.csv"))

summary_counts <- data.frame(
  Metric = c(
    "Baseline significant RSEI paths",
    "Baseline significant paths remaining significant under AR1",
    "AR1 significant paths remaining significant under MEM",
    "All baseline-to-AR1 directions retained",
    "All AR1-to-MEM directions retained"
  ),
  Value = c(
    sum(out$Baseline_significant, na.rm = TRUE),
    sum(out$Baseline_significant & out$AR1_significant, na.rm = TRUE),
    sum(out$AR1_significant & out$MEM_significant, na.rm = TRUE),
    all(out$Baseline_AR1_direction_same, na.rm = TRUE),
    all(out$AR1_MEM_direction_same, na.rm = TRUE)
  )
)
write_csv(summary_counts, file.path(project_paths$comparison, "robustness_summary_counts.csv"))

