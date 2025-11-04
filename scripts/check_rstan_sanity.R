#!/usr/bin/env Rscript

# Quick sanity check that cmdstanr and posterior can compile and run a simple Bayesian model.
# Fit a normal model to synthetic data and print posterior diagnostics.

suppressPackageStartupMessages({
  library(cmdstanr)
  library(posterior)
})

cmdstan_ready <- function() {
  path_try <- try(cmdstanr::cmdstan_path(), silent = TRUE)
  if (inherits(path_try, "try-error")) {
    return(FALSE)
  }
  dir.exists(path_try)
}

if (!cmdstan_ready()) {
  message("CmdStan not detected; attempting to install.")
  cmdstanr::install_cmdstan()
}

cmdstan_path <- cmdstanr::cmdstan_path()
message(sprintf("Using CmdStan at: %s", cmdstan_path))

options(mc.cores = max(1L, parallel::detectCores(logical = FALSE)))

stan_model_code <- "
data {
  int<lower = 0> N;
  vector[N] y;
}
parameters {
  real mu;
  real<lower = 0> sigma;
}
model {
  mu ~ normal(0, 5);
  sigma ~ exponential(1);
  y ~ normal(mu, sigma);
}
"

set.seed(2024)
standata <- list(
  N = 30L,
  y = rnorm(30L, mean = 0.5, sd = 0.75)
)

message("Compiling Stan model with cmdstanr...")
stan_file <- cmdstanr::write_stan_file(stan_model_code)
model <- cmdstanr::cmdstan_model(stan_file)

fit <- model$sample(
  data = standata,
  chains = 2,
  parallel_chains = 2,
  iter_warmup = 500,
  iter_sampling = 500,
  refresh = 0,
  adapt_delta = 0.9
)

message("Sampling complete. Posterior summary:")
summary_tbl <- fit$summary(variables = c("mu", "sigma"))
print(summary_tbl, digits = 3)

draws <- fit$draws(variables = c("mu", "sigma"))
diag_summary <- summarise_draws(draws, default_summary_measures())
print(diag_summary, digits = 3)

rhat_vals <- summary_tbl$rhat
ess_bulk_vals <- summary_tbl$ess_bulk

message(sprintf("max R-hat: %.3f", max(rhat_vals, na.rm = TRUE)))
message(sprintf("min bulk-ESS: %.0f", min(ess_bulk_vals, na.rm = TRUE)))

if (any(is.nan(rhat_vals)) || any(rhat_vals > 1.1)) {
  stop("R-hat exceeds 1.1; investigate Stan installation.")
}

message("cmdstanr sanity check completed successfully.")
