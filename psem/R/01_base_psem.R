if (!file.exists("R/00_utils.R")) stop("Run this script from the repository root.")
source("R/00_utils.R", encoding = "UTF-8")
suppressPackageStartupMessages({
  library(lme4)
  library(lmerTest)
  library(piecewiseSEM)
})

dat <- read_panel_data()

# Maximum-likelihood component models with a city random intercept.
m_ISA <- lmer(
  ISA_z ~ POP_log_z + PD_z + GPBE_log_z + ELE_z + SLO_z + SR_z + PRE_z +
    TEMP_z + year_z + (1 | city),
  data = dat, REML = FALSE
)

m_CRO <- lmer(
  CRO_z ~ POP_log_z + PD_z + GPBE_log_z + ELE_z + SLO_z + SR_z + PRE_z +
    TEMP_z + year_z + ISA_z + (1 | city),
  data = dat, REML = FALSE
)

m_FOR <- lmer(
  FOR_z ~ POP_log_z + PD_z + GPBE_log_z + ELE_z + SLO_z + SR_z + PRE_z +
    TEMP_z + year_z + ISA_z + (1 | city),
  data = dat, REML = FALSE
)

m_GRA <- lmer(
  GRA_z ~ POP_log_z + PD_z + GPBE_log_z + ELE_z + SLO_z + SR_z + PRE_z +
    TEMP_z + year_z + ISA_z + (1 | city),
  data = dat, REML = FALSE
)

m_GCR <- lmer(
  GCR_z ~ POP_log_z + PD_z + GPBE_log_z + ELE_z + SLO_z + SR_z + PRE_z +
    TEMP_z + year_z + ISA_z + (1 | city),
  data = dat, REML = FALSE
)

m_PM25 <- lmer(
  PM25_z ~ POP_log_z + PD_z + GPBE_log_z + ELE_z + SLO_z + SR_z + PRE_z +
    TEMP_z + year_z + ISA_z + FOR_z + GRA_z + (1 | city),
  data = dat, REML = FALSE
)

m_RSEI <- lmer(
  RSEI_z ~ POP_log_z + PD_z + GPBE_log_z + ELE_z + SLO_z + SR_z + PRE_z +
    TEMP_z + year_z + ISA_z + CRO_z + FOR_z + GRA_z + GCR_z + PM25_z +
    (1 | city),
  data = dat, REML = FALSE
)

# Land-cover variables share an area-allocation constraint. Correlated residuals
# are used instead of imposing arbitrary causal directions among them.
psem_model <- psem(
  m_ISA, m_CRO, m_FOR, m_GRA, m_GCR, m_PM25, m_RSEI,
  CRO_z %~~% FOR_z,
  CRO_z %~~% GRA_z,
  FOR_z %~~% GRA_z,
  GCR_z %~~% PM25_z,
  data = dat
)

summary_psem <- summary(psem_model)
dsep_tab <- dSep(psem_model)
fit_fisher <- fisherC(dsep_tab)
fit_aic_loglik <- AIC_psem(psem_model, AIC.type = "loglik")
fit_aic_dsep <- AIC_psem(psem_model, AIC.type = "dsep")
coef_tab <- coefs(psem_model, standardize = "none")
r2_tab <- rsquared(psem_model)

print(summary_psem)
print(fit_fisher)
print(r2_tab)

write_csv(coef_tab, file.path(project_paths$baseline, "PSEM_coefficients.csv"))
write_csv(r2_tab, file.path(project_paths$baseline, "PSEM_R2.csv"))
write_csv(dsep_tab, file.path(project_paths$baseline, "PSEM_dSep.csv"))
write_csv(as.data.frame(fit_fisher), file.path(project_paths$baseline, "PSEM_global_fit.csv"))

capture.output(summary_psem, file = file.path(project_paths$baseline, "PSEM_summary.txt"))
capture.output(fit_aic_loglik, file = file.path(project_paths$baseline, "PSEM_AIC_loglik.txt"))
capture.output(fit_aic_dsep, file = file.path(project_paths$baseline, "PSEM_AIC_dsep.txt"))

models <- list(ISA = m_ISA, CRO = m_CRO, FOR = m_FOR, GRA = m_GRA,
               GCR = m_GCR, PM25 = m_PM25, RSEI = m_RSEI)
singularity <- data.frame(
  Response = names(models),
  Singular = vapply(models, lme4::isSingular, logical(1), tol = 1e-4)
)
write_csv(singularity, file.path(project_paths$baseline, "singularity_checks.csv"))

