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

m_RSEI_ar1_mem_base <- lme(
  RSEI_z ~ POP_log_z + PD_z + GPBE_log_z + ELE_z + SLO_z + SR_z + PRE_z +
    TEMP_z + year_z + ISA_z + CRO_z + FOR_z + GRA_z + GCR_z + PM25_z,
  random = ~1 | city, correlation = corAR1(form = ~year | city),
  data = dat, method = "ML", control = ctrl
)

mem_obj <- make_positive_mem_basis(coords, k = 4)
E <- mem_obj$basis
mem_mode <- getOption("psem.mem_filter", "published")
filter_file <- if (identical(mem_mode, "reselected")) {
  file.path(project_paths$mem_selection, "mem_filter_reselected.csv")
} else {
  file.path("config", "mem_filter_published.csv")
}
if (!file.exists(filter_file)) stop("MEM filter file not found: ", filter_file)
filter_list <- read.csv(filter_file)
assert_columns(filter_list, c("year", "mem"), "MEM filter")
filter_list$year <- as.integer(filter_list$year)
filter_list$mem <- as.integer(filter_list$mem)
if (anyDuplicated(filter_list[c("year", "mem")])) stop("Duplicate year-MEM rows in filter.")
if (any(!filter_list$year %in% dat$year)) stop("MEM filter contains years outside the panel.")
if (any(filter_list$mem < 1L | filter_list$mem > ncol(E))) {
  stop("MEM filter contains an unavailable eigenvector index.")
}

mem_names <- character(0)
for (i in seq_len(nrow(filter_list))) {
  yy <- filter_list$year[i]
  mm <- filter_list$mem[i]
  nm <- paste0("MEM", mm, "_Y", yy)
  mem_names <- c(mem_names, nm)
  city_mem <- setNames(E[, mm], coords$city_id)
  dat[[nm]] <- 0
  hit <- dat$year == yy
  dat[[nm]][hit] <- city_mem[as.character(dat$city_code[hit])]
}

core_terms <- c(
  "POP_log_z", "PD_z", "GPBE_log_z", "ELE_z", "SLO_z", "SR_z", "PRE_z",
  "TEMP_z", "year_z", "ISA_z", "CRO_z", "FOR_z", "GRA_z", "GCR_z", "PM25_z"
)
form_mem <- as.formula(
  paste("RSEI_z ~", paste(c(core_terms, mem_names), collapse = " + "))
)
m_RSEI_mem <- lme(
  form_mem, random = ~1 | city, correlation = corAR1(form = ~year | city),
  data = dat, method = "ML", control = ctrl
)

fit_cmp <- AIC(m_RSEI_ar1_mem_base, m_RSEI_mem)
r2_ar1 <- piecewiseSEM::rsquared(m_RSEI_ar1_mem_base)
r2_mem <- piecewiseSEM::rsquared(m_RSEI_mem)
mem_fit <- data.frame(
  Model = c("RSEI_AR1", paste0("RSEI_AR1_MEM_", mem_mode)),
  N_parameters = fit_cmp$df,
  AIC = fit_cmp$AIC,
  logLik = c(as.numeric(logLik(m_RSEI_ar1_mem_base)), as.numeric(logLik(m_RSEI_mem))),
  rho = c(extract_rho(m_RSEI_ar1_mem_base), extract_rho(m_RSEI_mem)),
  Marginal_R2 = c(r2_ar1$Marginal[1], r2_mem$Marginal[1]),
  Conditional_R2 = c(r2_ar1$Conditional[1], r2_mem$Conditional[1])
)
write_csv(mem_fit, file.path(project_paths$mem, "RSEI_MEM_model_fit.csv"))
write_csv(filter_list, file.path(project_paths$mem, "RSEI_MEM_filter_used.csv"))

coef_base <- summary(m_RSEI_ar1_mem_base)$tTable
coef_mem <- summary(m_RSEI_mem)$tTable
core_out <- data.frame(
  predictor = core_terms,
  AR1_beta = coef_base[core_terms, "Value"],
  AR1_p = coef_base[core_terms, "p-value"],
  MEM_beta = coef_mem[core_terms, "Value"],
  MEM_p = coef_mem[core_terms, "p-value"]
)
core_out$direction_same <- sign(core_out$AR1_beta) == sign(core_out$MEM_beta)
write_csv(core_out, file.path(project_paths$mem, "RSEI_AR1_vs_MEM_paths.csv"))

moran_mem_k4 <- annual_moran(m_RSEI_mem, dat, coords, "RSEI_MEM", k = 4, nsim = 999)
moran_mem_k6 <- annual_moran(m_RSEI_mem, dat, coords, "RSEI_MEM", k = 6, nsim = 999)
write_csv(moran_mem_k4, file.path(project_paths$mem, "RSEI_MEM_Moran_k4.csv"))
write_csv(moran_mem_k6, file.path(project_paths$mem, "RSEI_MEM_Moran_k6.csv"))

mem_moran_summary <- bind_rows(
  data.frame(
    k = 4, Mean_I = mean(moran_mem_k4$Moran_I),
    Raw_sig_years = sum(moran_mem_k4$Significant_raw),
    FDR_sig_years = sum(moran_mem_k4$Significant_FDR)
  ),
  data.frame(
    k = 6, Mean_I = mean(moran_mem_k6$Moran_I),
    Raw_sig_years = sum(moran_mem_k6$Significant_raw),
    FDR_sig_years = sum(moran_mem_k6$Significant_FDR)
  )
)
write_csv(mem_moran_summary, file.path(project_paths$mem, "RSEI_MEM_Moran_summary.csv"))
capture.output(summary(m_RSEI_ar1_mem_base),
               file = file.path(project_paths$mem, "RSEI_AR1_summary.txt"))
capture.output(summary(m_RSEI_mem),
               file = file.path(project_paths$mem, "RSEI_AR1_MEM_summary.txt"))

