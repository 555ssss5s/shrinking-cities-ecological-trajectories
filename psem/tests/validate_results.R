args_all <- commandArgs(trailingOnly = FALSE)
file_arg <- grep("^--file=", args_all, value = TRUE)
test_dir <- if (length(file_arg)) {
  dirname(normalizePath(sub("^--file=", "", file_arg[1]), winslash = "/"))
} else {
  normalizePath(getwd(), winslash = "/")
}
root <- normalizePath(file.path(test_dir, ".."), winslash = "/")
setwd(root)

near <- function(actual, expected, tolerance = 1e-4, label = "value") {
  if (length(actual) != 1L || is.na(actual) || abs(actual - expected) > tolerance) {
    stop(label, " differs: actual=", actual, ", expected=", expected)
  }
}

fit <- read.csv("results/baseline/PSEM_global_fit.csv")
near(fit$Fisher.C, 5.605, label = "Baseline Fisher's C")
near(fit$P.Value, 0.691, label = "Baseline Fisher's C p-value")

r2 <- read.csv("results/baseline/PSEM_R2.csv")
rsei_r2 <- r2[r2$Response == "RSEI_z", ]
near(rsei_r2$Marginal, 0.6387830, tolerance = 1e-6, label = "RSEI marginal R2")
near(rsei_r2$Conditional, 0.8536439, tolerance = 1e-6, label = "RSEI conditional R2")

coef <- read.csv("results/baseline/PSEM_coefficients.csv", check.names = FALSE)
pd <- coef[coef$Response == "RSEI_z" & coef$Predictor == "PD_z", ]
near(pd$Estimate, 0.3288, label = "PD -> RSEI baseline coefficient")

ar1 <- read.csv("results/ar1_moran/AR1_global_fit.csv")
near(ar1$Fisher.C[ar1$Model == "Strict_AR1"], 21.558,
     label = "Strict AR1 Fisher's C")
near(ar1$Fisher.C[ar1$Model == "Refined_AR1_CRO_to_PM25"], 9.635,
     label = "Refined AR1 Fisher's C")

mem <- read.csv("results/mem/RSEI_MEM_Moran_summary.csv")
if (mem$FDR_sig_years[mem$k == 4] != 0L) stop("Expected zero FDR-significant k=4 years.")
if (mem$FDR_sig_years[mem$k == 6] != 1L) stop("Expected one FDR-significant k=6 year.")

counts <- read.csv("results/comparison/robustness_summary_counts.csv")
if (counts$Value[counts$Metric == "Baseline significant RSEI paths"] != "12") {
  stop("Unexpected count of significant baseline RSEI paths.")
}
if (counts$Value[counts$Metric ==
                 "Baseline significant paths remaining significant under AR1"] != "11") {
  stop("Unexpected count of baseline paths retained under AR1.")
}

cat("All validation checks passed.\n")

