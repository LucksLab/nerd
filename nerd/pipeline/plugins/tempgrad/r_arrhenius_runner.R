#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(jsonlite)
})

`%||%` <- function(x, y) if (is.null(x) || length(x) == 0) y else x

R_GAS_CONSTANT <- 1.98720425864083

build_numeric <- function(values, default = numeric()) {
  vec <- as.numeric(values %||% default)
  vec[is.na(vec)] <- NA_real_
  vec
}

compute_fit <- function(entry) {
  series_id <- entry$series_id %||% ""
  inv_t <- build_numeric(entry$inv_t)
  log_rates <- build_numeric(entry$log_rates)

  n <- min(length(inv_t), length(log_rates))
  if (!is.finite(n) || n < 2) {
    return(list(
      series_id = series_id,
      status = "failed",
      reason = "At least two observations are required for Arrhenius fitting.",
      params = list(),
      diagnostics = list(status = "failed"),
      artifacts = list()
    ))
  }

  inv_t <- inv_t[seq_len(n)]
  log_rates <- log_rates[seq_len(n)]

  if (any(!is.finite(inv_t)) || any(!is.finite(log_rates))) {
    return(list(
      series_id = series_id,
      status = "failed",
      reason = "Non-finite data detected in Arrhenius inputs.",
      params = list(),
      diagnostics = list(status = "failed"),
      artifacts = list()
    ))
  }

  weights <- entry$weights
  if (!is.null(weights) && length(weights) > 0) {
    weights <- as.numeric(weights)
    weights <- weights[seq_len(n)]
    if (any(!is.finite(weights)) || any(weights <= 0)) {
      weights <- NULL
    }
  } else {
    weights <- NULL
  }

  df <- data.frame(inv_t = inv_t, log_rates = log_rates)

  fit <- tryCatch(
    if (is.null(weights)) {
      lm(log_rates ~ inv_t, data = df)
    } else {
      lm(log_rates ~ inv_t, data = df, weights = weights)
    },
    error = function(err) err
  )

  if (inherits(fit, "error")) {
    reason <- conditionMessage(fit)
    return(list(
      series_id = series_id,
      status = "failed",
      reason = reason,
      params = list(),
      diagnostics = list(status = "failed", reason = reason),
      artifacts = list()
    ))
  }

  summary_fit <- summary(fit)
  coeffs <- coef(fit)

  slope <- coeffs[["inv_t"]]
  intercept <- coeffs[["(Intercept)"]]

  slope_err <- NA_real_
  intercept_err <- NA_real_
  coef_table <- summary_fit$coefficients
  if (!is.null(coef_table) && all(c("inv_t", "(Intercept)") %in% rownames(coef_table))) {
    slope_err <- coef_table["inv_t", "Std. Error"]
    intercept_err <- coef_table["(Intercept)", "Std. Error"]
  }

  residuals <- summary_fit$residuals
  chisq <- sum(residuals^2)
  r2 <- summary_fit$r.squared
  ndata <- length(log_rates)
  n_params <- length(coeffs)
  nfree <- max(ndata - n_params, 0)

  activation_energy <- -slope * R_GAS_CONSTANT
  activation_energy_err <- if (is.finite(slope_err)) abs(slope_err) * R_GAS_CONSTANT else NA_real_

  params <- list(
    slope = slope,
    intercept = intercept,
    activation_energy_cal_per_mol = activation_energy
  )
  if (is.finite(slope_err)) {
    params$slope_err <- slope_err
  }
  if (is.finite(intercept_err)) {
    params$intercept_err <- intercept_err
  }
  if (is.finite(activation_energy_err)) {
    params$activation_energy_err_cal_per_mol <- activation_energy_err
  }

  diagnostics <- list(
    r2 = r2,
    chisq = chisq,
    ndata = ndata,
    nfree = nfree,
    status = "completed"
  )

  list(
    series_id = series_id,
    status = "completed",
    reason = NULL,
    params = params,
    diagnostics = diagnostics,
    artifacts = list()
  )
}

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) {
  stop("Usage: r_arrhenius_runner.R <input.json> <output.json>")
}

input_path <- args[[1]]
output_path <- args[[2]]

payload <- fromJSON(readLines(input_path, warn = FALSE), simplifyVector = FALSE)
series_entries <- payload$series %||% list()

results <- lapply(series_entries, compute_fit)

output <- list(
  metadata = payload$metadata %||% list(),
  global_params = list(),
  artifacts = list(),
  series = results
)

json_text <- toJSON(output, auto_unbox = TRUE, pretty = TRUE, digits = NA, null = "null")
writeLines(json_text, con = output_path)
