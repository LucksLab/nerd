#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(jsonlite)
  library(nlme)
})

`%||%` <- function(x, y) if (is.null(x) || length(x) == 0) y else x

safe_log <- function(x, floor = 1e-12) {
  base::log(pmax(as.numeric(x), floor))
}

fmod_model <- function(timepoint, log_kappa, log_kdeg, log_fmod0) {
  kappa <- exp(log_kappa)
  kdeg <- exp(log_kdeg)
  fmod0 <- exp(log_fmod0)
  1 - exp(-kappa * (1 - exp(-kdeg * timepoint))) + fmod0
}

fmod_model_constrained <- function(timepoint, log_kappa, log_kdeg_fixed, log_fmod0) {
  kappa <- exp(log_kappa)
  kdeg <- exp(log_kdeg_fixed)
  fmod0 <- exp(log_fmod0)
  1 - exp(-kappa * (1 - exp(-kdeg * timepoint))) + fmod0
}

build_series_df <- function(series_entry) {
  times <- as.numeric(series_entry$timepoints %||% numeric())
  fmods <- as.numeric(series_entry$fmod_values %||% numeric())
  n <- min(length(times), length(fmods))
  if (n == 0) {
    return(data.frame())
  }
  times <- times[seq_len(n)]
  fmods <- fmods[seq_len(n)]
  meta <- series_entry$metadata %||% list()
  data.frame(
    series_id = rep(series_entry$series_id, n),
    nt_id = rep(series_entry$nt_id, n),
    valtype = rep(series_entry$valtype %||% "", n),
    timepoint = times,
    fmod = fmods,
    site = rep(meta$site %||% NA_integer_, n),
    base = rep(meta$base %||% NA_character_, n),
    stringsAsFactors = FALSE
  )
}

compute_diagnostics <- function(df, fitted_vals, residuals, n_params) {
  chisq <- sum(residuals^2)
  ss_tot <- sum((df$fmod - mean(df$fmod))^2)
  r2 <- if (abs(ss_tot) < .Machine$double.eps) NA_real_ else 1 - chisq / ss_tot
  list(
    r2 = r2,
    chisq = chisq,
    reduced_chisq = if (nrow(df) > n_params) chisq / (nrow(df) - n_params) else NA_real_,
    time_min = min(df$timepoint),
    time_max = max(df$timepoint),
    ndata = nrow(df),
    nfree = max(nrow(df) - n_params, 0),
    success = TRUE,
    status = "completed"
  )
}

fit_single_series <- function(series_entry, initial_log_kdeg = NA_real_, fixed_log_kdeg = NA_real_) {
  df <- build_series_df(series_entry)
  if (nrow(df) < 3) {
    return(list(
      status = "failed",
      reason = "At least three timepoints are required for a fit.",
      params = list(),
      diagnostics = list(status = "failed")
    ))
  }
  if (all(abs(df$fmod) < .Machine$double.eps)) {
    return(list(
      status = "failed",
      reason = "Non-zero fmod values are required for a fit.",
      params = list(),
      diagnostics = list(status = "failed")
    ))
  }

  log_kappa_start <- safe_log(
    max(1e-6, -log(max(1e-6, 1 - min(0.999, max(df$fmod)))))
  )
  log_fmod_start <- safe_log(max(1e-6, min(df$fmod)))

  if (is.na(initial_log_kdeg) || !is.finite(initial_log_kdeg)) {
    initial_log_kdeg <- safe_log(1e-3)
  }

  ctrl <- nls.control(maxiter = 200, warnOnly = TRUE)
  tryCatch({
    if (!is.na(fixed_log_kdeg) && is.finite(fixed_log_kdeg)) {
      df$log_kdeg_fixed <- fixed_log_kdeg
      fit <- nls(
        fmod ~ fmod_model_constrained(timepoint, log_kappa, log_kdeg_fixed, log_fmod0),
        data = df,
        start = list(
          log_kappa = log_kappa_start,
          log_fmod0 = log_fmod_start
        ),
        algorithm = "port",
        lower = c(log_kappa = log(1e-8), log_fmod0 = log(1e-12)),
        upper = c(log_kappa = log(1e4), log_fmod0 = log(1)),
        control = ctrl
      )
    } else {
      fit <- nls(
        fmod ~ fmod_model(timepoint, log_kappa, log_kdeg, log_fmod0),
        data = df,
        start = list(
          log_kappa = log_kappa_start,
          log_kdeg = initial_log_kdeg,
          log_fmod0 = log_fmod_start
        ),
        algorithm = "port",
        lower = c(log_kappa = log(1e-8), log_kdeg = log(1e-8), log_fmod0 = log(1e-12)),
        upper = c(log_kappa = log(1e4), log_kdeg = log(1e4), log_fmod0 = log(1)),
        control = ctrl
      )
    }

    coefs <- coef(fit)
    stderr <- tryCatch({
      se <- sqrt(diag(vcov(fit)))
      as.list(se)
    }, error = function(e) {
      list()
    })

    if (!is.na(fixed_log_kdeg) && is.finite(fixed_log_kdeg)) {
      log_kdeg_val <- fixed_log_kdeg
      n_params <- 2
    } else {
      log_kdeg_val <- coefs[["log_kdeg"]]
      n_params <- 3
    }

    log_kappa_val <- coefs[["log_kappa"]]
    log_fmod_val <- coefs[["log_fmod0"]]

    fitted_vals <- predict(fit)
    resid_vals <- residuals(fit)
    diagnostics <- compute_diagnostics(df, fitted_vals, resid_vals, n_params)

    params <- list(
      log_kobs = log_kappa_val,
      log_kdeg = log_kdeg_val,
      log_fmod0 = log_fmod_val,
      kobs = exp(log_kappa_val),
      kdeg = exp(log_kdeg_val),
      fmod0 = exp(log_fmod_val),
      metadata = series_entry$metadata %||% list()
    )
    if (!is.null(stderr$log_kappa)) {
      params$log_kobs_err <- stderr$log_kappa
    }
    if (!is.na(log_kdeg_val) && !is.null(stderr$log_kdeg)) {
      params$log_kdeg_err <- stderr$log_kdeg
    }
    if (!is.null(stderr$log_fmod0)) {
      params$log_fmod0_err <- stderr$log_fmod0
    }

    list(
      status = "completed",
      reason = NULL,
      params = params,
      diagnostics = diagnostics
    )
  }, error = function(err) {
    list(
      status = "failed",
      reason = conditionMessage(err),
      params = list(),
      diagnostics = list(status = "failed", reason = conditionMessage(err))
    )
  })
}

select_series_for_global <- function(series_results, global_selection, r2_threshold) {
  if (length(series_results) == 0) {
    return(character())
  }
  allowed_bases <- NULL
  mode <- tolower(trimws(global_selection %||% ""))
  if (mode %in% c("ac_only", "ac", "a_c")) {
    allowed_bases <- c("A", "C")
  } else if (mode %in% c("acg_only", "acg")) {
    allowed_bases <- c("A", "C", "G")
  }

  selected <- character()
  for (sid in names(series_results)) {
    record <- series_results[[sid]]
    if (is.null(record) || record$status != "completed") {
      next
    }
    meta <- record$params$metadata %||% list()
    base_val <- toupper(meta$base %||% "")
    if (!is.null(allowed_bases) && nzchar(base_val) && !(base_val %in% allowed_bases)) {
      next
    }
    if (!is.null(r2_threshold)) {
      r2 <- record$diagnostics$r2
      if (is.na(r2) || (!is.na(r2_threshold) && r2 < r2_threshold)) {
        next
      }
    }
    selected <- c(selected, sid)
  }
  unique(selected)
}

assemble_round_payload <- function(round_id, status, per_nt, global_params, qc_metrics, notes = NULL) {
  list(
    round_id = round_id,
    status = status,
    per_nt = per_nt,
    global_params = global_params %||% list(),
    qc_metrics = qc_metrics %||% list(),
    notes = notes
  )
}

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) {
  stop("Expected input and output JSON paths.")
}
input_path <- args[[1]]
output_path <- args[[2]]

json_text <- NULL
payload <- tryCatch({
  info <- file.info(input_path)
  size <- info$size
  if (is.na(size) || size <= 0) {
    json_lines <- readLines(input_path, warn = FALSE)
    json_text <<- paste(json_lines, collapse = "\n")
  } else {
    json_text <<- readChar(input_path, size, useBytes = TRUE)
  }
  fromJSON(json_text, simplifyVector = FALSE)
}, error = function(err) {
  tail_len <- 240
  tail_section <- ""
  text_len <- if (!is.null(json_text)) nchar(json_text, type = "bytes") else 0
  if (!is.null(json_text) && text_len > 0) {
    start_idx <- max(1, text_len - tail_len + 1)
    tail_section <- substr(json_text, start_idx, text_len)
    cat(
      sprintf(
        "JSON parse failed. Total bytes: %s. Last %d bytes:\n%s\n",
        text_len,
        min(tail_len, text_len),
        tail_section
      ),
      file = stderr()
    )
  } else {
    cat("JSON parse failed: input appears empty.\n", file = stderr())
  }
  stop(sprintf("Failed to parse input JSON '%s': %s", input_path, conditionMessage(err)))
})
series_list <- payload$series %||% list()
rounds_requested <- payload$rounds %||% list()
options <- payload$options %||% list()
resolved <- payload$resolved %||% list()

initial_log_kdeg <- resolved$initial_log_kdeg %||% NA_real_
fallback_constrained_log_kdeg <- resolved$constrained_log_kdeg %||% NA_real_

series_results <- list()
per_nt_round1 <- list()
successes_round1 <- 0

for (series_entry in series_list) {
  series_id <- series_entry$series_id
  fit <- fit_single_series(series_entry, initial_log_kdeg = initial_log_kdeg, fixed_log_kdeg = NA_real_)
  series_results[[series_id]] <- fit
  payload_entry <- list(
    nt_id = series_entry$nt_id,
    valtype = series_entry$valtype %||% "",
    params = fit$params %||% list(),
    diagnostics = fit$diagnostics %||% list()
  )
  payload_entry$diagnostics$status <- fit$status
  if (!is.null(fit$reason)) {
    payload_entry$diagnostics$reason <- fit$reason
  }
  per_nt_round1[[length(per_nt_round1) + 1]] <- payload_entry
  if (fit$status == "completed") {
    successes_round1 <- successes_round1 + 1
  }
}

rounds_output <- list()
round1_status <- if (successes_round1 > 0) "completed" else "failed"
round1_qc <- list(
  n_total = length(series_list),
  n_success = successes_round1,
  success_rate = if (length(series_list) > 0) successes_round1 / length(series_list) else 0
)
if ("round1_free" %in% rounds_requested) {
  rounds_output[[length(rounds_output) + 1]] <- assemble_round_payload(
    "round1_free",
    round1_status,
    per_nt_round1,
    global_params = list(),
    qc_metrics = round1_qc
  )
}

global_round_status <- "skipped"
global_params <- list()
global_qc <- list()
global_notes <- NULL
global_log_kdeg <- NA_real_

run_global <- "round2_global" %in% rounds_requested
if (run_global) {
  r2_threshold <- options$global_filters$r2_threshold %||% NULL
  if (!is.null(r2_threshold)) {
    r2_threshold <- as.numeric(r2_threshold)
  }

  selected_ids <- select_series_for_global(
    series_results,
    options$global_selection %||% "",
    r2_threshold
  )

  if (length(selected_ids) == 0) {
    global_round_status <- "skipped"
    global_notes <- "No nucleotides satisfied the selection criteria for global fitting."
  } else if (length(selected_ids) < 2) {
    global_round_status <- "skipped"
    global_notes <- "Global fit skipped: requires at least two nucleotides after filtering."
  } else {
    frame_map <- list()
    kept_ids <- character()

    for (sid in selected_ids) {
      entry <- NULL
      for (candidate in series_list) {
        if (identical(candidate$series_id, sid)) {
          entry <- candidate
          break
        }
      }
      if (is.null(entry)) {
        next
      }

      frame <- build_series_df(entry)
      if (!("fmod" %in% names(frame))) {
        frame$fmod <- as.numeric(entry$fmod_values %||% numeric())
      }
      if (!("timepoint" %in% names(frame))) {
        frame$timepoint <- as.numeric(entry$timepoints %||% numeric())
      }
      if (nrow(frame) < 3) {
        next
      }

      frame_map[[sid]] <- frame
      kept_ids <- c(kept_ids, sid)
    }

    kept_ids <- unique(kept_ids)

    if (length(kept_ids) < 2) {
      global_round_status <- "skipped"
      global_notes <- "Global fit skipped: insufficient usable nucleotides after filtering."
    } else {
      selected_ids <- kept_ids
      global_df <- do.call(rbind, frame_map[selected_ids])
      global_df <- as.data.frame(global_df)
      if (!("fmod" %in% names(global_df))) {
        stop(
          "Global dataset missing 'fmod' column; columns: ",
          paste(names(global_df), collapse = ", ")
        )
      }
      global_df$series_factor <- factor(global_df$series_id)
      global_df <- groupedData(fmod ~ timepoint | series_factor, data = global_df)

      starts_kappa <- sapply(selected_ids, function(sid) {
        record <- series_results[[sid]]
        if (!is.null(record$params$log_kobs)) {
          return(as.numeric(record$params$log_kobs))
        }
        safe_log(1e-3)
      })
      names(starts_kappa) <- paste0("log_kappa.", levels(global_df$series_factor))

      starts_fmod <- sapply(selected_ids, function(sid) {
        record <- series_results[[sid]]
        if (!is.null(record$params$log_fmod0)) {
          return(as.numeric(record$params$log_fmod0))
        }
        safe_log(1e-6)
      })
      names(starts_fmod) <- paste0("log_fmod0.", levels(global_df$series_factor))

      starts_kdeg <- sapply(selected_ids, function(sid) {
        record <- series_results[[sid]]
        if (!is.null(record$params$log_kdeg)) {
          return(as.numeric(record$params$log_kdeg))
        }
        safe_log(1e-3)
      })
      starts_kdeg <- starts_kdeg[is.finite(starts_kdeg)]

      if (is.na(initial_log_kdeg) || !is.finite(initial_log_kdeg)) {
        if (length(starts_kdeg) > 0) {
          initial_log_kdeg <- mean(starts_kdeg)
        }
        if (!is.finite(initial_log_kdeg)) {
          initial_log_kdeg <- safe_log(1e-3)
        }
      }

      start_kappa_vals <- starts_kappa[is.finite(starts_kappa)]
      log_kappa_start_global <- if (length(start_kappa_vals) > 0) {
        mean(start_kappa_vals)
      } else {
        safe_log(1e-3)
      }

      start_fmod_vals <- starts_fmod[is.finite(starts_fmod)]
      log_fmod_start_global <- if (length(start_fmod_vals) > 0) {
        mean(start_fmod_vals)
      } else {
        safe_log(1e-6)
      }

      global_fit <- tryCatch(
        {
          nlme(
            fmod ~ fmod_model(timepoint, log_kappa, log_kdeg, log_fmod0),
            data = global_df,
            fixed = log_kappa + log_kdeg + log_fmod0 ~ 1,
            random = pdDiag(list(log_kappa ~ 1, log_fmod0 ~ 1)),
            groups = ~series_factor,
            start = c(
              log_kappa = log_kappa_start_global,
              log_kdeg = initial_log_kdeg,
              log_fmod0 = log_fmod_start_global
            ),
            control = nlmeControl(
              maxIter = 200,
              pnlsMaxIter = 200,
              msMaxIter = 200,
              tolerance = 1e-6
            )
          )
        },
        error = function(err) {
          err
        }
      )

      if (inherits(global_fit, "error")) {
        msg <- conditionMessage(global_fit)
        if (grepl("contrasts not defined for -1 degrees of freedom", msg, fixed = TRUE)) {
          global_round_status <- "skipped"
          global_notes <- "Global fit skipped: insufficient contrast after filtering."
        } else {
          global_round_status <- "failed"
          global_notes <- paste0("Global fit failed: ", msg)
        }
      } else {
        fixed_params <- fixef(global_fit)
        log_kdeg_raw <- fixed_params[["log_kdeg"]]
        if (is.null(log_kdeg_raw) || length(log_kdeg_raw) == 0) {
          global_round_status <- "failed"
          global_notes <- "Global fit failed: log_kdeg coefficient missing in result."
          global_params <- list()
          global_qc <- list()
        } else {
          log_kdeg <- suppressWarnings(as.numeric(log_kdeg_raw[1]))
          if (is.na(log_kdeg) || !is.finite(log_kdeg)) {
            global_round_status <- "failed"
            global_notes <- "Global fit failed: log_kdeg coefficient non-finite."
            global_params <- list()
            global_qc <- list()
          } else {
            global_log_kdeg <- log_kdeg

            fitted_vals <- fitted(global_fit)
            resid_vals <- residuals(global_fit)
            chisq <- sum(resid_vals^2)
            ss_tot <- sum((global_df$fmod - mean(global_df$fmod))^2)
            r2 <- if (abs(ss_tot) < .Machine$double.eps) NA_real_ else 1 - chisq / ss_tot

            se_table <- tryCatch(
              {
                summary(global_fit)$tTable
              },
              error = function(err) NULL
            )
            log_kdeg_err <- if (!is.null(se_table) && "log_kdeg" %in% rownames(se_table)) {
              as.numeric(se_table["log_kdeg", "Std.Error"])
            } else {
              NA_real_
            }

            global_round_status <- "completed"
            global_params <- list(
              log_kdeg = log_kdeg,
              kdeg = exp(log_kdeg)
            )
            if (!is.na(log_kdeg_err) && is.finite(log_kdeg_err)) {
              global_params$log_kdeg_err <- log_kdeg_err
            }
            global_qc <- list(
              chisq = chisq,
              r2 = r2,
              ndata = nrow(global_df),
              nfree = max(nrow(global_df) - length(fixed_params), 0),
              n_sites = length(selected_ids)
            )
          }
        }
      }
    }
  }
}

if ("round2_global" %in% rounds_requested) {
  rounds_output[[length(rounds_output) + 1]] <- assemble_round_payload(
    "round2_global",
    global_round_status,
    per_nt = list(),
    global_params = global_params,
    qc_metrics = global_qc,
    notes = global_notes
  )
}

run_constrained <- "round3_constrained" %in% rounds_requested

constrained_log_kdeg <- NA_real_
if (global_round_status == "completed" && is.finite(global_log_kdeg)) {
  constrained_log_kdeg <- global_log_kdeg
} else if (!is.na(fallback_constrained_log_kdeg) && is.finite(fallback_constrained_log_kdeg)) {
  constrained_log_kdeg <- fallback_constrained_log_kdeg
}

if (run_constrained) {
  if (is.na(constrained_log_kdeg) || !is.finite(constrained_log_kdeg)) {
    rounds_output[[length(rounds_output) + 1]] <- assemble_round_payload(
      "round3_constrained",
      "skipped",
      per_nt = list(),
      global_params = list(),
      qc_metrics = list(),
      notes = "No constrained kdeg value supplied or produced by round 2."
    )
  } else {
    per_nt_round3 <- list()
    successes_round3 <- 0
    for (series_entry in series_list) {
      fit <- fit_single_series(
        series_entry,
        initial_log_kdeg = constrained_log_kdeg,
        fixed_log_kdeg = constrained_log_kdeg
      )
      payload_entry <- list(
        nt_id = series_entry$nt_id,
        valtype = series_entry$valtype %||% "",
        params = fit$params %||% list(),
        diagnostics = fit$diagnostics %||% list()
      )
      payload_entry$diagnostics$status <- fit$status
      if (!is.null(fit$reason)) {
        payload_entry$diagnostics$reason <- fit$reason
      }
      per_nt_round3[[length(per_nt_round3) + 1]] <- payload_entry
      if (fit$status == "completed") {
        successes_round3 <- successes_round3 + 1
      }
    }
    round3_status <- if (successes_round3 > 0) "completed" else "failed"
    qc <- list(
      n_total = length(series_list),
      n_success = successes_round3,
      success_rate = if (length(series_list) > 0) successes_round3 / length(series_list) else 0
    )
    rounds_output[[length(rounds_output) + 1]] <- assemble_round_payload(
      "round3_constrained",
      round3_status,
      per_nt_round3,
      global_params = list(log_kdeg = constrained_log_kdeg, kdeg = exp(constrained_log_kdeg)),
      qc_metrics = qc,
      notes = NULL
    )
  }
}

metadata <- payload$global_metadata %||% list()
metadata$rg_id <- payload$rg_id
metadata$engine <- "r_integration"

output <- list(
  metadata = metadata,
  rounds = rounds_output,
  artifacts = list()
)

json <- toJSON(output, pretty = TRUE, auto_unbox = TRUE, digits = 8, na = "null")
write(json, output_path)
