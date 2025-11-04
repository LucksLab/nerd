#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(jsonlite)
})

if (!requireNamespace("brms", quietly = TRUE)) {
  stop("Package 'brms' must be installed to use the Bayesian Arrhenius engine.")
}
suppressPackageStartupMessages({
  library(brms)
})

if (!requireNamespace("posterior", quietly = TRUE)) {
  stop("Package 'posterior' must be installed to use the Bayesian Arrhenius engine.")
}
suppressPackageStartupMessages({
  library(posterior)
})

`%||%` <- function(x, y) if (is.null(x) || length(x) == 0) y else x

R_GAS_CONSTANT <- 1.98720425864083

prepare_series <- function(entry) {
  sid <- entry$series_id %||% ""
  inv_t <- as.numeric(entry$inv_t %||% numeric())
  log_rates <- as.numeric(entry$log_rates %||% numeric())
  n <- min(length(inv_t), length(log_rates))
  if (!is.finite(n) || n < 2) {
    return(list(
      ok = FALSE,
      series_id = sid,
      reason = "At least two observations are required for Arrhenius fitting.",
      n = n
    ))
  }

  inv_t <- inv_t[seq_len(n)]
  log_rates <- log_rates[seq_len(n)]

  if (any(!is.finite(inv_t)) || any(!is.finite(log_rates))) {
    return(list(
      ok = FALSE,
      series_id = sid,
      reason = "Non-finite inverse temperatures or log rates detected.",
      n = n
    ))
  }

  std_errs <- entry$log_rate_std_errors
  has_std_err <- FALSE
  if (!is.null(std_errs)) {
    std_errs <- as.numeric(std_errs)
    if (length(std_errs) >= n) {
      std_errs <- std_errs[seq_len(n)]
      if (all(is.finite(std_errs)) && all(std_errs > 0)) {
        std_errs <- pmax(std_errs, 1e-12)
        has_std_err <- TRUE
      } else {
        std_errs <- NULL
      }
    } else {
      std_errs <- NULL
    }
  } else {
    std_errs <- NULL
  }

  df <- data.frame(
    series_id = rep(sid, n),
    inv_t = inv_t,
    log_rates = log_rates,
    stringsAsFactors = FALSE
  )

  if (!is.null(std_errs)) {
    df$log_rate_std_err <- std_errs
  }

  group_id <- entry$group_id %||% sid
  if (!is.null(group_id)) {
    df$group_id <- rep(group_id, n)
  }

  list(
    ok = TRUE,
    series_id = sid,
    df = df,
    n = n,
    has_std_err = has_std_err
  )
}

resolve_bool <- function(value, default = FALSE) {
  if (is.null(value) || length(value) == 0) {
    return(default)
  }
  if (is.logical(value)) {
    return(isTRUE(value))
  }
  if (is.numeric(value)) {
    return(!is.na(value) && value != 0)
  }
  if (is.character(value)) {
    lower <- tolower(trimws(value))
    if (lower %in% c("true", "t", "yes", "y", "1")) {
      return(TRUE)
    }
    if (lower %in% c("false", "f", "no", "n", "0")) {
      return(FALSE)
    }
  }
  default
}

resolve_formula <- function(use_random_effects, has_std_err, custom_formula = NULL) {
  if (!is.null(custom_formula) && nzchar(custom_formula)) {
    parsed <- tryCatch(eval(parse(text = custom_formula)), error = function(err) err)
    if (inherits(parsed, "error")) {
      stop(paste0("Failed to parse custom Bayesian formula: ", conditionMessage(parsed)))
    }
    return(list(string = custom_formula, object = parsed))
  }

  if (use_random_effects) {
    if (has_std_err) {
      formula_str <- "bf(log_rates | se(log_rate_std_err, sigma = FALSE) ~ inv_t + (1 + inv_t | series_id))"
    } else {
      formula_str <- "bf(log_rates ~ inv_t + (1 + inv_t | series_id))"
    }
  } else {
    if (has_std_err) {
      formula_str <- "bf(log_rates | se(log_rate_std_err, sigma = FALSE) ~ inv_t)"
    } else {
      formula_str <- "bf(log_rates ~ inv_t)"
    }
  }

  list(
    string = formula_str,
    object = eval(parse(text = formula_str))
  )
}

resolve_numeric <- function(value, default) {
  if (is.null(value) || length(value) == 0) {
    return(default)
  }
  numeric <- suppressWarnings(as.numeric(value[1]))
  if (!is.finite(numeric)) {
    return(default)
  }
  numeric
}

resolve_integer <- function(value, default) {
  num <- resolve_numeric(value, default)
  as.integer(round(num))
}

resolve_priors <- function(options) {
  priors <- options$priors
  if (is.null(priors) || length(priors) == 0) {
    return(set_prior("normal(0, 10)", class = "b"))
  }

  if (is.character(priors)) {
    priors <- as.list(priors)
  }

  parsed <- lapply(priors, function(expr) {
    tryCatch(eval(parse(text = expr)), error = function(err) err)
  })
  if (any(vapply(parsed, inherits, logical(1), "error"))) {
    messages <- vapply(parsed, function(x) {
      if (inherits(x, "error")) conditionMessage(x) else ""
    }, character(1))
    stop(paste0("Failed to parse provided priors: ", paste(messages[messages != ""], collapse = "; ")))
  }

  Reduce(`c`, parsed)
}

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) {
  stop("Usage: r_arrhenius_bayesian_runner.R <input.json> <output.json>")
}

input_path <- args[[1]]
output_path <- args[[2]]

payload <- fromJSON(readLines(input_path, warn = FALSE), simplifyVector = FALSE)
series_entries <- payload$series %||% list()
options_payload <- payload$options %||% list()
engine_options <- payload$engine_options %||% list()
bayes_options <- payload$bayes_options %||% list()

prepared <- lapply(series_entries, prepare_series)

valid_entries <- Filter(function(item) isTRUE(item$ok), prepared)
invalid_entries <- Filter(function(item) !isTRUE(item$ok), prepared)

series_results <- list()
results_map <- list()

if (length(valid_entries) == 0) {
  for (item in invalid_entries) {
    result <- list(
      series_id = item$series_id,
      status = "failed",
      reason = item$reason %||% "No valid data supplied for Bayesian Arrhenius fitting.",
      params = list(),
      diagnostics = list(status = "failed", reason = item$reason),
      artifacts = list()
    )
    series_results[[length(series_results) + 1]] <- result
    results_map[[item$series_id]] <- result
  }

  output <- list(
    metadata = payload$metadata %||% list(),
    global_params = list(),
    artifacts = list(),
    series = series_results
  )

  json_text <- toJSON(output, auto_unbox = TRUE, pretty = TRUE, digits = NA, null = "null")
  writeLines(json_text, con = output_path)
  quit(save = "no")
}

combined_df <- do.call(rbind, lapply(valid_entries, function(item) item$df))

has_std_err <- "log_rate_std_err" %in% colnames(combined_df)
random_effects_opt <- bayes_options$random_effects %||% engine_options$random_effects
use_random_effects <- resolve_bool(random_effects_opt, length(valid_entries) > 1)

formula_info <- resolve_formula(
  use_random_effects = use_random_effects,
  has_std_err = has_std_err,
  custom_formula = bayes_options$formula %||% engine_options$formula
)

chains <- max(1L, resolve_integer(bayes_options$chains %||% engine_options$chains, 4L))
iter <- max(1000L, resolve_integer(bayes_options$iter %||% engine_options$iter, 4000L))
warmup <- resolve_integer(bayes_options$warmup %||% engine_options$warmup, floor(iter / 2))
warmup <- max(500L, min(warmup, iter - 100L))
cores <- max(1L, resolve_integer(bayes_options$cores %||% engine_options$cores, min(chains, 4L)))
seed <- resolve_integer(bayes_options$seed %||% engine_options$seed, NA_integer_)
backend <- bayes_options$backend %||% engine_options$backend %||% "rstan"
refresh <- resolve_integer(bayes_options$refresh %||% engine_options$refresh, 0L)
adapt_delta <- resolve_numeric(bayes_options$adapt_delta %||% engine_options$adapt_delta, NA_real_)
max_treedepth <- resolve_integer(bayes_options$max_treedepth %||% engine_options$max_treedepth, NA_integer_)
thin <- resolve_integer(bayes_options$thin %||% engine_options$thin, 1L)
sample_prior <- bayes_options$sample_prior %||% engine_options$sample_prior %||% "no"

priors <- resolve_priors(bayes_options)

control <- list()
if (is.finite(adapt_delta)) {
  control$adapt_delta <- adapt_delta
}
if (!is.na(max_treedepth) && max_treedepth > 0) {
  control$max_treedepth <- max_treedepth
}

threads_opt <- resolve_integer(bayes_options$threads %||% engine_options$threads, NA_integer_)
threading <- NULL
if (!is.na(threads_opt) && threads_opt > 0) {
  threading <- threads(default = threads_opt)
}

options(mc.cores = cores)

rhat_max <- NA_real_
ess_bulk_min <- NA_real_
ess_tail_min <- NA_real_
n_draws_value <- NA_integer_
global_slope <- NULL
global_intercept <- NULL
warnings_captured <- character()
fit <- withCallingHandlers(
  tryCatch(
    brm(
      formula = formula_info$object,
      data = combined_df,
      family = gaussian(),
      prior = priors,
      chains = chains,
      iter = iter,
      warmup = warmup,
      cores = cores,
      seed = seed,
      backend = backend,
      refresh = refresh,
      control = control,
      sample_prior = sample_prior,
      thin = thin,
      threads = threading,
      silent = TRUE,
      open_progress = FALSE
    ),
    error = function(err) err
  ),
  warning = function(w) {
    warnings_captured <<- c(warnings_captured, conditionMessage(w))
    invokeRestart("muffleWarning")
  }
)

if (inherits(fit, "error")) {
  reason <- conditionMessage(fit)
  for (item in valid_entries) {
    result <- list(
      series_id = item$series_id,
      status = "failed",
      reason = reason,
      params = list(),
      diagnostics = list(status = "failed", reason = reason),
      artifacts = list()
    )
    series_results[[length(series_results) + 1]] <- result
    results_map[[item$series_id]] <- result
  }
} else {
  coef_summary <- NULL
  if ("series_id" %in% names(coef(fit, summary = TRUE))) {
    coef_summary <- coef(fit, summary = TRUE)$series_id
  }
  fixed_summary <- fixef(fit, summary = TRUE)

  slope_name <- "inv_t"
  intercept_name <- "Intercept"
  global_slope <- if (slope_name %in% rownames(fixed_summary)) fixed_summary[slope_name, ] else NULL
  global_intercept <- if (intercept_name %in% rownames(fixed_summary)) fixed_summary[intercept_name, ] else NULL

  rhat_values <- tryCatch(rhat(fit), error = function(err) numeric())
  ess_bulk_values <- tryCatch(ess_bulk(fit), error = function(err) numeric())
  ess_tail_values <- tryCatch(ess_tail(fit), error = function(err) numeric())

  rhat_max <- if (length(rhat_values)) max(rhat_values, na.rm = TRUE) else NA_real_
  ess_bulk_min <- if (length(ess_bulk_values)) min(ess_bulk_values, na.rm = TRUE) else NA_real_
  ess_tail_min <- if (length(ess_tail_values)) min(ess_tail_values, na.rm = TRUE) else NA_real_
  n_draws_value <- tryCatch({
    posterior::ndraws(as_draws(fit))
  }, error = function(err) NA_integer_)

  counts_per_series <- table(combined_df$series_id)

  for (item in valid_entries) {
    sid <- item$series_id
    params <- list()
    ndata_val <- counts_per_series[[sid]]
    if (is.null(ndata_val) || any(is.na(ndata_val))) {
      ndata_val <- item$n
    }
    diagnostics <- list(
      status = "completed",
      ndata = unname(ndata_val),
      has_measurement_error = item$has_std_err,
      posterior_rhat_max = rhat_max,
      posterior_ess_bulk_min = ess_bulk_min,
      posterior_ess_tail_min = ess_tail_min,
      ndraws = n_draws_value
    )

    intercept_stats <- NULL
    slope_stats <- NULL
    if (!is.null(coef_summary) && sid %in% dimnames(coef_summary)[[1]]) {
      intercept_stats <- coef_summary[sid, intercept_name, ]
      if (slope_name %in% dimnames(coef_summary)[[2]]) {
        slope_stats <- coef_summary[sid, slope_name, ]
      }
    }

    if (is.null(intercept_stats) && !is.null(global_intercept)) {
      intercept_stats <- global_intercept
    }
    if (is.null(slope_stats) && !is.null(global_slope)) {
      slope_stats <- global_slope
    }

    if (is.null(intercept_stats) || is.null(slope_stats)) {
      reason <- "Unable to extract posterior summaries for intercept or slope."
      series_results[[length(series_results) + 1]] <- list(
        series_id = sid,
        status = "failed",
        reason = reason,
        params = list(),
        diagnostics = list(status = "failed", reason = reason),
        artifacts = list()
      )
      next
    }

    slope_estimate <- slope_stats["Estimate"]
    slope_error <- slope_stats["Est.Error"]
    slope_q2.5 <- slope_stats["Q2.5"]
    slope_q97.5 <- slope_stats["Q97.5"]

    intercept_estimate <- intercept_stats["Estimate"]
    intercept_error <- intercept_stats["Est.Error"]
    intercept_q2.5 <- intercept_stats["Q2.5"]
    intercept_q97.5 <- intercept_stats["Q97.5"]

    activation_energy <- -slope_estimate * R_GAS_CONSTANT
    activation_energy_err <- if (!is.na(slope_error)) abs(slope_error) * R_GAS_CONSTANT else NA_real_
    activation_energy_low <- -slope_q97.5 * R_GAS_CONSTANT
    activation_energy_high <- -slope_q2.5 * R_GAS_CONSTANT

    params <- list(
      slope = slope_estimate,
      slope_sd = slope_error,
      slope_q2_5 = slope_q2.5,
      slope_q97_5 = slope_q97.5,
      intercept = intercept_estimate,
      intercept_sd = intercept_error,
      intercept_q2_5 = intercept_q2.5,
      intercept_q97_5 = intercept_q97_5,
      activation_energy_cal_per_mol = activation_energy,
      activation_energy_sd_cal_per_mol = activation_energy_err,
      activation_energy_q2_5_cal_per_mol = activation_energy_low,
      activation_energy_q97_5_cal_per_mol = activation_energy_high
    )

    result <- list(
      series_id = sid,
      status = "completed",
      reason = NULL,
      params = params,
      diagnostics = diagnostics,
      artifacts = list()
    )
    series_results[[length(series_results) + 1]] <- result
    results_map[[sid]] <- result
  }
}

if (length(invalid_entries) > 0) {
  for (item in invalid_entries) {
    # Already captured earlier when length(valid_entries) == 0 handled,
    # but ensure duplicates aren't added twice.
    if (!is.null(results_map[[item$series_id]])) {
      next
    }
    result <- list(
      series_id = item$series_id,
      status = "failed",
      reason = item$reason %||% "Invalid input series.",
      params = list(),
      diagnostics = list(status = "failed", reason = item$reason),
      artifacts = list()
    )
    series_results[[length(series_results) + 1]] <- result
    results_map[[item$series_id]] <- result
  }
}

if (length(series_entries) > 0) {
  ordered_results <- vector("list", length(series_entries))
  for (idx in seq_along(series_entries)) {
    entry <- series_entries[[idx]]
    sid <- entry$series_id %||% ""
    result <- results_map[[sid]]
    if (is.null(result)) {
      reason <- "Series missing from Bayesian engine output."
      result <- list(
        series_id = sid,
        status = "failed",
        reason = reason,
        params = list(),
        diagnostics = list(status = "failed", reason = reason),
        artifacts = list()
      )
    }
    ordered_results[[idx]] <- result
  }
  series_results <- ordered_results
}

metadata <- payload$metadata %||% list()
metadata$mode <- payload$mode %||% metadata$mode %||% "bayesian"
metadata$formula <- formula_info$string
metadata$backend <- backend
metadata$chains <- chains
metadata$iter <- iter
metadata$warmup <- warmup
metadata$random_effects <- use_random_effects
metadata$has_measurement_error <- has_std_err

artifacts <- list()
if (length(warnings_captured) > 0) {
  artifacts$warnings <- unique(warnings_captured)
}

artifacts$diagnostics <- list(
  rhat_max = rhat_max %||% NA_real_,
  ess_bulk_min = ess_bulk_min %||% NA_real_,
  ess_tail_min = ess_tail_min %||% NA_real_,
  ndraws = n_draws_value %||% NA_integer_
)

global_params <- list()
if (!is.null(global_slope)) {
  global_params$fixed_slope <- list(
    estimate = global_slope["Estimate"],
    est_error = global_slope["Est.Error"],
    q2_5 = global_slope["Q2.5"],
    q97_5 = global_slope["Q97.5"]
  )
  global_params$fixed_activation_energy_cal_per_mol <- -global_slope["Estimate"] * R_GAS_CONSTANT
}
if (!is.null(global_intercept)) {
  global_params$fixed_intercept <- list(
    estimate = global_intercept["Estimate"],
    est_error = global_intercept["Est.Error"],
    q2_5 = global_intercept["Q2.5"],
    q97_5 = global_intercept["Q97.5"]
  )
}

output <- list(
  metadata = metadata,
  global_params = global_params,
  artifacts = artifacts,
  series = series_results
)

json_text <- toJSON(output, auto_unbox = TRUE, pretty = TRUE, digits = NA, null = "null")
writeLines(json_text, con = output_path)
