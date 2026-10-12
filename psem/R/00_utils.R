suppressPackageStartupMessages({
  library(readxl)
  library(dplyr)
})

project_paths <- list(
  panel = file.path("data", "psem_panel.xlsx"),
  coords = file.path("data", "city_coordinates.xlsx"),
  baseline = file.path("results", "baseline"),
  ar1 = file.path("results", "ar1_moran"),
  mem_selection = file.path("results", "mem_selection"),
  mem = file.path("results", "mem"),
  comparison = file.path("results", "comparison"),
  paths = file.path("results", "path_lists")
)

for (p in project_paths[c("baseline", "ar1", "mem_selection", "mem", "comparison", "paths")]) {
  dir.create(p, recursive = TRUE, showWarnings = FALSE)
}

assert_columns <- function(x, required, label) {
  missing <- setdiff(required, names(x))
  if (length(missing)) {
    stop(label, " is missing required columns: ", paste(missing, collapse = ", "))
  }
}

read_panel_data <- function() {
  if (!file.exists(project_paths$panel)) stop("Missing input: ", project_paths$panel)
  dat <- readxl::read_excel(project_paths$panel, sheet = 1)

  if (!"CRO" %in% names(dat) && "FAR" %in% names(dat)) dat$CRO <- dat$FAR

  required <- c(
    "year", "city", "city_code", "RSEI", "POP", "PD", "GPBE", "ELE", "SLO",
    "SR", "PRE", "TEMP", "ISA", "CRO", "FOR", "GRA", "GCR", "PM25"
  )
  assert_columns(dat, required, "Panel workbook")

  dat <- dat %>%
    mutate(
      year = as.integer(year),
      city_code = as.integer(city_code),
      city = factor(city_code),
      POP_log = log(POP),
      GPBE_log = log(GPBE)
    ) %>%
    arrange(city_code, year)

  if (any(!is.finite(dat$POP_log)) || any(!is.finite(dat$GPBE_log))) {
    stop("POP and GPBE must be positive and non-missing before log transformation.")
  }
  if (anyDuplicated(dat[c("city_code", "year")])) {
    stop("Panel workbook contains duplicate city_code-year rows.")
  }

  vars_to_scale <- c(
    "RSEI", "POP_log", "PD", "GPBE_log", "ELE", "SLO", "SR", "PRE", "TEMP",
    "ISA", "CRO", "FOR", "GRA", "GCR", "PM25", "year"
  )
  if (any(!complete.cases(dat[vars_to_scale]))) {
    stop("Variables used by the PSEM contain missing values.")
  }
  for (v in vars_to_scale) {
    if (stats::sd(dat[[v]]) == 0) stop("Cannot standardize constant variable: ", v)
    dat[[paste0(v, "_z")]] <- as.numeric(scale(dat[[v]]))
  }

  if (n_distinct(dat$city_code) != 113L || !identical(sort(unique(dat$year)), 2000:2021)) {
    warning("Input differs from the published 113-city, 2000-2021 panel.")
  }
  dat
}

read_city_coords <- function(panel) {
  if (!file.exists(project_paths$coords)) stop("Missing input: ", project_paths$coords)
  raw <- readxl::read_excel(project_paths$coords, sheet = 1)
  assert_columns(raw, c("city_id", "x", "y"), "Coordinate workbook")

  pairs <- raw %>%
    transmute(city_id = as.integer(city_id), x = as.numeric(x), y = as.numeric(y)) %>%
    distinct()
  if (anyDuplicated(pairs$city_id)) {
    stop("At least one city_id has more than one coordinate pair.")
  }
  if (any(!is.finite(pairs$x)) || any(!is.finite(pairs$y))) {
    stop("Coordinate workbook contains missing or non-finite coordinates.")
  }

  ids <- sort(unique(panel$city_code))
  if (!setequal(ids, pairs$city_id)) {
    stop("city_code values in the panel do not match city_id values in coordinates.")
  }
  pairs[match(ids, pairs$city_id), ]
}

write_csv <- function(x, path) {
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
  utils::write.csv(x, path, row.names = FALSE, fileEncoding = "UTF-8")
}

extract_rho <- function(model) {
  as.numeric(stats::coef(model$modelStruct$corStruct, unconstrained = FALSE))
}

make_knn_listw <- function(coords, k) {
  xy <- as.matrix(coords[, c("x", "y")])
  knn <- spdep::knearneigh(xy, k = k)
  nb <- spdep::knn2nb(knn, row.names = as.character(coords$city_id), sym = TRUE)
  spdep::nb2listw(nb, style = "W", zero.policy = TRUE)
}

annual_moran <- function(model, dat, coords, response_name, k = 4, nsim = 999,
                         seed_base = 20260830) {
  lw <- make_knn_listw(coords, k)
  rr <- data.frame(
    city_id = dat$city_code,
    year = dat$year,
    residual = as.numeric(stats::resid(model, type = "normalized"))
  )

  out <- lapply(sort(unique(rr$year)), function(yy) {
    tmp <- rr[rr$year == yy, ]
    tmp <- tmp[match(coords$city_id, tmp$city_id), ]
    if (anyNA(tmp$residual)) stop("Residual-city matching failed for year ", yy)
    set.seed(seed_base + yy + k)
    mc <- spdep::moran.mc(
      tmp$residual, lw, nsim = nsim, alternative = "two.sided", zero.policy = TRUE
    )
    data.frame(
      Response = response_name,
      Year = yy,
      Moran_I = as.numeric(mc$statistic),
      P_perm = mc$p.value
    )
  })
  out <- dplyr::bind_rows(out)
  out$P_BH_FDR <- stats::p.adjust(out$P_perm, method = "BH")
  out$Significant_raw <- out$P_perm < 0.05
  out$Significant_FDR <- out$P_BH_FDR < 0.05
  out
}

make_positive_mem_basis <- function(coords, k = 4) {
  xy <- as.matrix(coords[, c("x", "y")])
  knn <- spdep::knearneigh(xy, k = k)
  nb <- spdep::knn2nb(knn, row.names = as.character(coords$city_id), sym = TRUE)
  adjacency <- spdep::nb2mat(nb, style = "B", zero.policy = TRUE)
  n <- nrow(adjacency)
  centering <- diag(n) - matrix(1 / n, n, n)
  eig <- eigen(centering %*% adjacency %*% centering, symmetric = TRUE)
  keep <- which(eig$values > 1e-8)
  basis <- scale(eig$vectors[, keep, drop = FALSE])
  colnames(basis) <- paste0("MEM", seq_len(ncol(basis)))
  list(basis = basis, eigenvalues = eig$values[keep], nb = nb)
}

clean_term <- function(x) {
  x <- sub("^~~", "", x)
  x <- sub("_z$", "", x)
  x <- sub("_log$", "", x)
  x[x == "PM25"] <- "PM2.5"
  x
}

