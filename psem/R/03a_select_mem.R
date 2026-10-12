if (!file.exists("R/00_utils.R")) stop("Run this script from the repository root.")
source("R/00_utils.R", encoding = "UTF-8")
suppressPackageStartupMessages({
  library(nlme)
  library(spdep)
})

dat <- read_panel_data()
coords <- read_city_coords(dat)

m_RSEI_ar1_select <- lme(
  RSEI_z ~ POP_log_z + PD_z + GPBE_log_z + ELE_z + SLO_z + SR_z + PRE_z +
    TEMP_z + year_z + ISA_z + CRO_z + FOR_z + GRA_z + GCR_z + PM25_z,
  random = ~1 | city,
  correlation = corAR1(form = ~year | city),
  data = dat, method = "ML",
  control = lmeControl(opt = "optim", maxIter = 500, msMaxIter = 500)
)

mem_obj <- make_positive_mem_basis(coords, k = 4)
E <- mem_obj$basis
lw <- spdep::nb2listw(mem_obj$nb, style = "W", zero.policy = TRUE)
rr <- data.frame(
  city_id = dat$city_code,
  year = dat$year,
  residual = as.numeric(resid(m_RSEI_ar1_select, type = "normalized"))
)

perm_moran <- function(x, yy, step, nsim = 999) {
  set.seed(20260830 + yy * 100 + step)
  mc <- spdep::moran.mc(x, lw, nsim = nsim, alternative = "two.sided", zero.policy = TRUE)
  c(I = as.numeric(mc$statistic), p = mc$p.value)
}

select_one_year <- function(y, yy, alpha_moran = 0.05, alpha_enter = 0.05,
                            max_terms = 10L) {
  initial <- perm_moran(y, yy, 0L)
  selected <- integer(0)
  current <- y
  current_moran <- initial
  step <- 0L

  while (current_moran[["p"]] < alpha_moran && length(selected) < max_terms) {
    remaining <- setdiff(seq_len(ncol(E)), selected)
    candidate_p <- vapply(remaining, function(j) {
      design <- data.frame(y = y, E[, c(selected, j), drop = FALSE])
      fit <- lm(y ~ ., data = design)
      coef(summary(fit))[nrow(coef(summary(fit))), "Pr(>|t|)"]
    }, numeric(1))
    best_pos <- which.min(candidate_p)
    best <- remaining[best_pos]
    if (!is.finite(candidate_p[best_pos]) || candidate_p[best_pos] >= alpha_enter) break

    selected <- c(selected, best)
    fit <- lm(y ~ ., data = data.frame(y = y, E[, selected, drop = FALSE]))
    current <- residuals(fit)
    step <- step + 1L
    current_moran <- perm_moran(current, yy, step)
  }

  list(
    selected = selected,
    diagnostic = data.frame(
      year = yy, initial_I = initial[["I"]], initial_p = initial[["p"]],
      n_selected = length(selected), final_I = current_moran[["I"]],
      final_p = current_moran[["p"]]
    )
  )
}

selected_rows <- list()
diagnostics <- list()
for (yy in sort(unique(rr$year))) {
  tmp <- rr[rr$year == yy, ]
  y <- tmp$residual[match(coords$city_id, tmp$city_id)]
  ans <- select_one_year(y, yy)
  diagnostics[[as.character(yy)]] <- ans$diagnostic
  if (length(ans$selected)) {
    selected_rows[[as.character(yy)]] <- data.frame(year = yy, mem = ans$selected)
  }
}

filter_reselected <- bind_rows(selected_rows)
selection_diagnostics <- bind_rows(diagnostics)
write_csv(filter_reselected, file.path(project_paths$mem_selection, "mem_filter_reselected.csv"))
write_csv(selection_diagnostics, file.path(project_paths$mem_selection, "mem_selection_diagnostics.csv"))

