if (!file.exists("R/00_utils.R")) stop("Run this script from the repository root.")
source("R/00_utils.R", encoding = "UTF-8")
suppressPackageStartupMessages({
  library(nlme)
  library(piecewiseSEM)
  library(spdep)
})

dat <- read_panel_data()
coords <- read_city_coords(dat)
ctrl <- lmeControl(opt = "optim", maxIter = 500, msMaxIter = 500)

m_ISA_ar1 <- lme(
  ISA_z ~ POP_log_z + PD_z + GPBE_log_z + ELE_z + SLO_z + SR_z + PRE_z +
    TEMP_z + year_z,
  random = ~1 | city, correlation = corAR1(form = ~year | city),
  data = dat, method = "ML", control = ctrl
)
m_CRO_ar1 <- lme(
  CRO_z ~ POP_log_z + PD_z + GPBE_log_z + ELE_z + SLO_z + SR_z + PRE_z +
    TEMP_z + year_z + ISA_z,
  random = ~1 | city, correlation = corAR1(form = ~year | city),
  data = dat, method = "ML", control = ctrl
)
m_FOR_ar1 <- lme(
  FOR_z ~ POP_log_z + PD_z + GPBE_log_z + ELE_z + SLO_z + SR_z + PRE_z +
    TEMP_z + year_z + ISA_z,
  random = ~1 | city, correlation = corAR1(form = ~year | city),
  data = dat, method = "ML", control = ctrl
)
m_GRA_ar1 <- lme(
  GRA_z ~ POP_log_z + PD_z + GPBE_log_z + ELE_z + SLO_z + SR_z + PRE_z +
    TEMP_z + year_z + ISA_z,
  random = ~1 | city, correlation = corAR1(form = ~year | city),
  data = dat, method = "ML", control = ctrl
)
m_GCR_ar1 <- lme(
  GCR_z ~ POP_log_z + PD_z + GPBE_log_z + ELE_z + SLO_z + SR_z + PRE_z +
    TEMP_z + year_z + ISA_z,
  random = ~1 | city, correlation = corAR1(form = ~year | city),
  data = dat, method = "ML", control = ctrl
)

# Strict robustness model: retain the baseline causal structure.
m_PM25_ar1 <- lme(
  PM25_z ~ POP_log_z + PD_z + GPBE_log_z + ELE_z + SLO_z + SR_z + PRE_z +
    TEMP_z + year_z + ISA_z + FOR_z + GRA_z,
  random = ~1 | city, correlation = corAR1(form = ~year | city),
  data = dat, method = "ML", control = ctrl
)
m_RSEI_ar1 <- lme(
  RSEI_z ~ POP_log_z + PD_z + GPBE_log_z + ELE_z + SLO_z + SR_z + PRE_z +
    TEMP_z + year_z + ISA_z + CRO_z + FOR_z + GRA_z + GCR_z + PM25_z,
  random = ~1 | city, correlation = corAR1(form = ~year | city),
  data = dat, method = "ML", control = ctrl
)

psem_ar1_strict <- psem(
  m_ISA_ar1, m_CRO_ar1, m_FOR_ar1, m_GRA_ar1, m_GCR_ar1,
  m_PM25_ar1, m_RSEI_ar1,
  CRO_z %~~% FOR_z,
  CRO_z %~~% GRA_z,
  FOR_z %~~% GRA_z,
  GCR_z %~~% PM25_z,
  data = dat
)

strict_summary <- summary(psem_ar1_strict)
strict_dsep <- dSep(psem_ar1_strict)
strict_fisher <- fisherC(strict_dsep)
strict_coef <- coefs(psem_ar1_strict, standardize = "none")
strict_r2 <- rsquared(psem_ar1_strict)

write_csv(strict_dsep, file.path(project_paths$ar1, "AR1_strict_dSep.csv"))
write_csv(strict_coef, file.path(project_paths$ar1, "AR1_strict_coefficients.csv"))
write_csv(strict_r2, file.path(project_paths$ar1, "AR1_strict_R2.csv"))
capture.output(strict_summary, file = file.path(project_paths$ar1, "AR1_strict_summary.txt"))

# Directed separation identified CRO -> PM25 as the only added path in the
# refined AR(1) specification. Strict and refined models are both retained.
m_PM25_ar1_refined <- lme(
  PM25_z ~ POP_log_z + PD_z + GPBE_log_z + ELE_z + SLO_z + SR_z + PRE_z +
    TEMP_z + year_z + ISA_z + FOR_z + GRA_z + CRO_z,
  random = ~1 | city, correlation = corAR1(form = ~year | city),
  data = dat, method = "ML", control = ctrl
)

psem_ar1_refined <- psem(
  m_ISA_ar1, m_CRO_ar1, m_FOR_ar1, m_GRA_ar1, m_GCR_ar1,
  m_PM25_ar1_refined, m_RSEI_ar1,
  CRO_z %~~% FOR_z,
  CRO_z %~~% GRA_z,
  FOR_z %~~% GRA_z,
  GCR_z %~~% PM25_z,
  data = dat
)

refined_summary <- summary(psem_ar1_refined)
refined_dsep <- dSep(psem_ar1_refined)
refined_fisher <- fisherC(refined_dsep)
refined_coef <- coefs(psem_ar1_refined, standardize = "none")
refined_r2 <- rsquared(psem_ar1_refined)

write_csv(refined_dsep, file.path(project_paths$ar1, "AR1_refined_dSep.csv"))
write_csv(refined_coef, file.path(project_paths$ar1, "AR1_refined_coefficients.csv"))
write_csv(refined_r2, file.path(project_paths$ar1, "AR1_refined_R2.csv"))
capture.output(refined_summary, file = file.path(project_paths$ar1, "AR1_refined_summary.txt"))

global_fit <- bind_rows(
  transform(as.data.frame(strict_fisher), Model = "Strict_AR1"),
  transform(as.data.frame(refined_fisher), Model = "Refined_AR1_CRO_to_PM25")
) %>% select(Model, everything())
write_csv(global_fit, file.path(project_paths$ar1, "AR1_global_fit.csv"))

rho_table <- data.frame(
  Response = c("ISA", "CRO", "FOR", "GRA", "GCR", "PM25", "RSEI"),
  rho = c(
    extract_rho(m_ISA_ar1), extract_rho(m_CRO_ar1), extract_rho(m_FOR_ar1),
    extract_rho(m_GRA_ar1), extract_rho(m_GCR_ar1),
    extract_rho(m_PM25_ar1_refined), extract_rho(m_RSEI_ar1)
  )
)
write_csv(rho_table, file.path(project_paths$ar1, "AR1_rho.csv"))

# Use the refined PM25 equation for its residual diagnostic; other equations
# are identical between strict and refined specifications.
robust_models <- list(
  ISA = m_ISA_ar1, CRO = m_CRO_ar1, FOR = m_FOR_ar1, GRA = m_GRA_ar1,
  GCR = m_GCR_ar1, PM25 = m_PM25_ar1_refined, RSEI = m_RSEI_ar1
)
run_all_moran <- function(k) {
  bind_rows(lapply(names(robust_models), function(nm) {
    annual_moran(robust_models[[nm]], dat, coords, nm, k = k, nsim = 999)
  }))
}

moran_k4 <- run_all_moran(4)
moran_k6 <- run_all_moran(6)
write_csv(moran_k4, file.path(project_paths$ar1, "Moran_AR1_normalized_k4.csv"))
write_csv(moran_k6, file.path(project_paths$ar1, "Moran_AR1_normalized_k6.csv"))

moran_summary <- bind_rows(
  moran_k4 %>% group_by(Response) %>% summarise(
    k = 4, Mean_I = mean(Moran_I), Min_I = min(Moran_I), Max_I = max(Moran_I),
    Raw_sig_years = sum(Significant_raw), FDR_sig_years = sum(Significant_FDR),
    .groups = "drop"
  ),
  moran_k6 %>% group_by(Response) %>% summarise(
    k = 6, Mean_I = mean(Moran_I), Min_I = min(Moran_I), Max_I = max(Moran_I),
    Raw_sig_years = sum(Significant_raw), FDR_sig_years = sum(Significant_FDR),
    .groups = "drop"
  )
)
write_csv(moran_summary, file.path(project_paths$ar1, "Moran_AR1_summary.csv"))

