if (!file.exists("R/00_utils.R")) stop("Run this script from the repository root.")
source("R/00_utils.R", encoding = "UTF-8")

input_file <- file.path(project_paths$baseline, "PSEM_coefficients.csv")
if (!file.exists(input_file)) stop("Missing baseline coefficients: ", input_file)
d <- read.csv(input_file, check.names = FALSE, stringsAsFactors = FALSE)
names(d)[8] <- "Significance"

directed <- d[
  !grepl("~~", d$Response, fixed = TRUE) & d$Predictor != "year_z" & d$P.Value < 0.05,
]
controls <- c("POP_log_z", "PD_z", "GPBE_log_z", "ELE_z", "SLO_z",
              "SR_z", "PRE_z", "TEMP_z")
directed_out <- data.frame(
  Type = ifelse(directed$Predictor %in% controls,
                "Exogenous/control path", "Core structural path"),
  Predictor = clean_term(directed$Predictor),
  Response = clean_term(directed$Response),
  Path = paste(clean_term(directed$Predictor), "->", clean_term(directed$Response)),
  Estimate = directed$Estimate,
  P_value = directed$P.Value,
  Significance = directed$Significance,
  stringsAsFactors = FALSE
)

resid <- d[grepl("~~", d$Response, fixed = TRUE), ]
resid_out <- data.frame(
  Type = "Correlated residual",
  Predictor = clean_term(resid$Response),
  Response = clean_term(resid$Predictor),
  Path = paste(clean_term(resid$Response), "<->", clean_term(resid$Predictor)),
  Estimate = resid$Estimate,
  P_value = resid$P.Value,
  Significance = resid$Significance,
  stringsAsFactors = FALSE
)
path_list <- rbind(directed_out, resid_out)
write_csv(path_list, file.path(project_paths$paths, "significant_paths_no_year.csv"))

direct_rsei <- directed_out[directed_out$Response == "RSEI", ]
write_csv(direct_rsei, file.path(project_paths$paths, "direct_RSEI_paths.csv"))

mediators <- c("ISA", "CRO", "FOR", "GRA", "GCR", "PM2.5")
candidate_mediation <- directed_out[
  directed_out$Response %in% mediators |
    (directed_out$Response == "RSEI" & directed_out$Predictor %in% mediators),
]
write_csv(candidate_mediation,
          file.path(project_paths$paths, "candidate_mediation_component_paths.csv"))

