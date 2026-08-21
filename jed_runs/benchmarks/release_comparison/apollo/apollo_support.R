# Shared helpers for the Apollo release-comparison benchmark.

parse_benchmark_arguments <- function() {
  values <- commandArgs(trailingOnly = TRUE)
  if (length(values) %% 2 != 0) {
    stop("Arguments must be supplied as --name value pairs.")
  }
  result <- list()
  if (length(values) == 0) return(result)
  for (i in seq(1, length(values), by = 2)) {
    key <- values[[i]]
    if (!startsWith(key, "--")) stop("Invalid argument: ", key)
    result[[substring(key, 3)]] <- values[[i + 1]]
  }
  required <- c("data", "output")
  missing <- required[!required %in% names(result)]
  if (length(missing) > 0) {
    stop("Missing required argument(s): ", paste(paste0("--", missing), collapse = ", "))
  }
  result
}

load_swissmetro <- function(data_path, panel = FALSE) {
  database <- read.delim(
    data_path,
    sep = "\t",
    header = TRUE,
    stringsAsFactors = FALSE,
    check.names = FALSE
  )

  keep <- database$PURPOSE %in% c(1, 3) & database$CHOICE != 0
  database <- database[keep, , drop = FALSE]

  database$SM_COST <- database$SM_CO * (database$GA == 0)
  database$TRAIN_COST <- database$TRAIN_CO * (database$GA == 0)
  database$CAR_AV_SP <- database$CAR_AV * (database$SP != 0)
  database$TRAIN_AV_SP <- database$TRAIN_AV * (database$SP != 0)
  database$TRAIN_TT_SCALED <- database$TRAIN_TT / 100
  database$TRAIN_COST_SCALED <- database$TRAIN_COST / 100
  database$SM_TT_SCALED <- database$SM_TT / 100
  database$SM_COST_SCALED <- database$SM_COST / 100
  database$CAR_TT_SCALED <- database$CAR_TT / 100
  database$CAR_CO_SCALED <- database$CAR_CO / 100

  if (!panel) {
    # Apollo requires an indivID even for cross-sectional models.  The
    # Biogeme b05a specification is not panel: every filtered row is its own
    # decision maker, even though the source data contain repeated IDs.
    database$apollo_row_id <- seq_len(nrow(database))
  }
  database
}

write_benchmark_result <- function(
    output_path,
    model_name,
    data_path,
    draws,
    draw_type,
    hessian_routine,
    model,
    elapsed_seconds,
    panel,
    rows
) {
  if (!requireNamespace("jsonlite", quietly = TRUE)) {
    stop("The jsonlite package is required to write benchmark results.")
  }
  estimate <- model$estimate
  if (is.null(estimate)) estimate <- numeric()
  estimated_parameters <- as.list(as.numeric(estimate))
  names(estimated_parameters) <- names(estimate)

  result <- list(
    package = "apollo",
    apollo_version = as.character(utils::packageVersion("apollo")),
    model = model_name,
    data_path = normalizePath(data_path, mustWork = FALSE),
    rows = rows,
    panel = panel,
    draws = draws,
    draw_type = draw_type,
    hessian_routine = hessian_routine,
    seed = 1223,
    wall_time_seconds = unname(elapsed_seconds),
    final_log_likelihood = if (!is.null(model$maximum)) as.numeric(model$maximum) else NA_real_,
    estimated_parameters = estimated_parameters,
    successful_estimation = isTRUE(model$successfulEstimation),
    optimizer_code = if (!is.null(model$code)) as.numeric(model$code) else NA_real_
  )
  dir.create(dirname(output_path), recursive = TRUE, showWarnings = FALSE)
  jsonlite::write_json(
    result,
    output_path,
    auto_unbox = TRUE,
    pretty = TRUE,
    digits = 17,
    na = "null"
  )
}
