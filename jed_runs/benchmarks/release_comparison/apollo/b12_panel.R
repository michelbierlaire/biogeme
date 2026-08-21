#!/usr/bin/env Rscript

library(apollo)

script_argument <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
script_file <- sub("^--file=", "", script_argument[[1]])
source(file.path(dirname(normalizePath(script_file)), "apollo_support.R"))

arguments <- parse_benchmark_arguments()
data_path <- arguments[["data"]]
output_path <- arguments[["output"]]
draws <- 5000L
draw_type <- "pmc"
model_name <- "b12_panel"

apollo_initialise()
database <- load_swissmetro(data_path, panel = TRUE)

apollo_beta <- c(
  b_cost = 0,
  b_time = 0,
  b_time_s = 1,
  asc_car = 0,
  asc_car_s = 1,
  asc_train = 0,
  asc_train_s = 1,
  asc_sm = 0,
  asc_sm_s = 1
)
apollo_fixed <- character()

apollo_control <- list(
  modelName = paste0(model_name, "_apollo"),
  modelDescr = "Apollo reproduction of the Biogeme b12 panel benchmark",
  indivID = "ID",
  panelData = TRUE,
  mixing = TRUE,
  nCores = 1,
  seed = 1223,
  analyticGrad = TRUE,
  noDiagnostics = TRUE,
  workInLogs = FALSE,
  outputDirectory = file.path(dirname(output_path), "apollo_files")
)

apollo_draws <- list(
  interDrawsType = draw_type,
  interNDraws = draws,
  interNormDraws = c(
    "draw_b_time",
    "draw_asc_car",
    "draw_asc_train",
    "draw_asc_sm"
  ),
  interUnifDraws = character(),
  intraDrawsType = draw_type,
  intraNDraws = 0L,
  intraNormDraws = character(),
  intraUnifDraws = character()
)

apollo_randCoeff <- function(apollo_beta, apollo_inputs) {
  # See the corresponding note in b05a_normal_mixture.R.  The random
  # coefficients are written directly in the conditional utilities below.
  return(list())
}

apollo_inputs <- apollo_validateInputs(silent = TRUE)

apollo_probabilities <- function(apollo_beta, apollo_inputs, functionality = "estimate") {
  apollo_attach(apollo_beta, apollo_inputs)
  on.exit(apollo_detach(apollo_beta, apollo_inputs))
  P <- list()

  V <- list(
    train = (asc_train + asc_train_s * draw_asc_train) +
      (b_time + b_time_s * draw_b_time) * TRAIN_TT_SCALED + b_cost * TRAIN_COST_SCALED,
    swissmetro = (asc_sm + asc_sm_s * draw_asc_sm) +
      (b_time + b_time_s * draw_b_time) * SM_TT_SCALED + b_cost * SM_COST_SCALED,
    car = (asc_car + asc_car_s * draw_asc_car) +
      (b_time + b_time_s * draw_b_time) * CAR_TT_SCALED + b_cost * CAR_CO_SCALED
  )
  mnl_settings <- list(
    alternatives = c(train = 1, swissmetro = 2, car = 3),
    avail = list(train = TRAIN_AV_SP, swissmetro = SM_AV, car = CAR_AV_SP),
    choiceVar = CHOICE,
    utilities = V
  )
  P[["model"]] <- apollo_mnl(mnl_settings, functionality)
  P <- apollo_panelProd(P, apollo_inputs, functionality)
  P <- apollo_avgInterDraws(P, apollo_inputs, functionality)
  P <- apollo_prepareProb(P, apollo_inputs, functionality)
  return(P)
}

estimate_settings <- list(
  estimationRoutine = "bgw",
  hessianRoutine = "none",
  maxIterations = 1000,
  printLevel = 0,
  silent = TRUE,
  writeIter = FALSE,
  scaleAfterConvergence = FALSE
)

started <- proc.time()[["elapsed"]]
model <- apollo_estimate(
  apollo_beta,
  apollo_fixed,
  apollo_probabilities,
  apollo_inputs,
  estimate_settings
)
elapsed <- proc.time()[["elapsed"]] - started

write_benchmark_result(
  output_path = output_path,
  model_name = model_name,
  data_path = data_path,
  draws = draws,
  draw_type = draw_type,
  hessian_routine = estimate_settings$hessianRoutine,
  model = model,
  elapsed_seconds = elapsed,
  panel = TRUE,
  rows = nrow(database)
)

cat(sprintf("Apollo %s: %.6f seconds, converged=%s\n", model_name, elapsed, isTRUE(model$successfulEstimation)))
