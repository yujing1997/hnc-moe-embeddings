#!/usr/bin/env Rscript
# =============================================================================
# pmsampsize_riley_clean.R
# =============================================================================
# Clean Riley et al. sample-size calculation for Cox prediction models.
#
# Purpose
# -------
# Recompute required sample size with the pmsampsize package using the same
# simple pattern as the package documentation for survival outcomes:
#
#   pmsampsize(type = "s", csrsquared = ..., parameters = ...,
#              rate = ..., timepoint = ..., meanfup = ...)
#
# Study-specific choices
# ----------------------
# * type        = "s" for survival / Cox prediction models.
# * parameters  = 6 for the baseline model, 9 for enhanced models.
#   - Baseline: 4 clinical covariates + 2 missingness indicators.
#   - Enhanced: baseline + 3 LLM features OR baseline + 3 embedding PCs.
# * shrinkage   = 0.9, the pmsampsize default.
# * timepoint   = 2 years, matching the 24-month IBS / calibration horizon.
# * rate        = observed incidence rate = events / total follow-up time.
# * meanfup     = observed mean follow-up time, in years.
# * csrsquared  = derived from an assumed C-statistic of 0.70 using the
#   Riley/Royston equations shown in the Riley sample-size papers.  The package
#   example value 0.051 is retained as a sensitivity comparator, not primary.
#
# Notes on arguments that do NOT apply here
# ----------------------------------------
# * prevalence and cstatistic are binary-outcome options in pmsampsize docs.
# * intercept, sd, and mmoe are continuous-outcome options.
# * For survival models, use rate, timepoint, meanfup, parameters, shrinkage,
#   and csrsquared or nagrsquared.
#
# Output
# ------
# Writes:
#   src/analysis/pmsampsize_riley_clean_results.csv
#   src/analysis/pmsampsize_riley_clean_results.txt
#   src/analysis/pmsampsize_riley_clean_results.md
# =============================================================================

.libPaths(c("~/R/library", .libPaths()))
suppressPackageStartupMessages({
  library(pmsampsize)
})

BASE_DIR <- "/Data/Yujing/HNC_OutcomePred/Reports_Agents/Revision1/Downstream_Analysis"
OUTCOMES_PATH <- file.path(BASE_DIR, "outputs_all_fields", "all_fields_processed.csv")
OUT_DIR <- file.path(BASE_DIR, "moe_embeddings_comp", "src", "analysis")

TIMEPOINT_YEARS <- 2
SHRINKAGE <- 0.90
# Use either a single C-statistic, e.g. 0.68, or a grid, e.g.
# seq(0.60, 0.70, by = 0.02) or c(0.60, 0.64, 0.68, 0.70).
ASSUMED_CSTAT <- seq(0.60, 0.70, by = 0.02) # or a single value, e.g. 0.68
ASSUMED_CSTATS <- sort(unique(as.numeric(ASSUMED_CSTAT)))
if (length(ASSUMED_CSTATS) == 0 || any(is.na(ASSUMED_CSTATS)) ||
    any(ASSUMED_CSTATS <= 0.5) || any(ASSUMED_CSTATS >= 1)) {
  stop("ASSUMED_CSTAT must contain one or more numeric values between 0.5 and 1.")
}

# Main-paper assumption. Keep this value in ASSUMED_CSTAT if you want the
# corresponding table generated in Section 5.
MAIN_PAPER_CSTAT <- 0.68
CSTAT_DIAGNOSTIC <- 0.60
ASSUMED_CSRSQ_LABELS <- sprintf("assumed_C%.2f_Riley_R2", ASSUMED_CSTATS)
PRIMARY_CSRSQ_LABEL <- ASSUMED_CSRSQ_LABELS[1]
DIAGNOSTIC_CSRSQ_LABEL <- sprintf("diagnostic_C%.2f_Riley_R2", CSTAT_DIAGNOSTIC)

# Documentation example comparator, not a universal/default value.
DOC_EXAMPLE_CSRSQUARED <- 0.051

# Small sensitivity grid around plausible Cox-Snell R² values.
CSRSQ_SENSITIVITY <- c(0.020, DOC_EXAMPLE_CSRSQUARED, 0.075, 0.100)

OUTCOMES <- list(
  OS     = list(time = "OS",     event = "Death"),
  DFS    = list(time = "DFS",    event = "Failure"),
  LFS    = list(time = "LFS",    event = "Locoregional_Failure"),
  DistFS = list(time = "DistFS", event = "distant failure")
)

PARAMETER_SCENARIOS <- list(
  baseline_p6 = 6,
  enhanced_p9 = 9
)

as_num <- function(x) suppressWarnings(as.numeric(x))

round_int <- function(x) {
  if (is.na(x)) return(NA_integer_)
  as.integer(round(x))
}

# Riley/Royston C-statistic -> Cox-Snell R² conversion used for survival
# prediction model sample size calculations.
#
# Mirrors the equations in the Riley worked examples / colleague note:
#   Eq 22: D = 5.5(C-0.5) + 10.26(C-0.5)^3
#   Eq 21: R2_D = [(pi/8)D^2] / [pi^2/6 + (pi/8)D^2]
#   Eq 20: Royston R2 = [-(pi^2/6)R2_D] / [(pi^2/6)R2_D - 1]
#   Eq 18: LR = -E * log(1 - Royston R2)
#   Eq  6: R2_CS = 1 - exp(-LR / N)
cstat_to_csrsq_riley <- function(cstat, n, events) {
  d <- 5.5 * (cstat - 0.5) + 10.26 * (cstat - 0.5)^3
  r2_d <- ((pi / 8) * d^2) / ((pi^2 / 6) + (pi / 8) * d^2)
  royston_r2 <- (-(pi^2 / 6) * r2_d) / (((pi^2 / 6) * r2_d) - 1)
  lr <- -events * log(1 - royston_r2)
  1 - exp(-lr / n)
}

cstat_conversion_components <- function(cstat) {
  d <- 5.5 * (cstat - 0.5) + 10.26 * (cstat - 0.5)^3
  r2_d <- ((pi / 8) * d^2) / ((pi^2 / 6) + (pi / 8) * d^2)
  royston_r2 <- (-(pi^2 / 6) * r2_d) / (((pi^2 / 6) * r2_d) - 1)

  c(d = d, r2_d = r2_d, royston_r2 = royston_r2)
}

fmt_num <- function(x, digits = 3) {
  ifelse(is.na(x), "NA", formatC(x, format = "f", digits = digits))
}

fmt_int <- function(x) {
  ifelse(is.na(x), "NA", formatC(as.integer(round(x)), format = "d", big.mark = ","))
}

fmt_yes_no <- function(x) {
  ifelse(is.na(x), "NA", ifelse(x, "Yes", "No"))
}

scenario_label <- function(x) {
  ifelse(x == "baseline_p6", "Baseline", ifelse(x == "enhanced_p9", "Enhanced", x))
}

md_table <- function(headers, rows) {
  align <- ifelse(grepl("^(N|Events|p|Observed|Required|Riley|C=|Doc|R\\^2|Event|Incidence|Mean|\\$)", headers), "---:", "---")
  c(
    paste0("| ", paste(headers, collapse = " | "), " |"),
    paste0("| ", paste(align, collapse = " | "), " |"),
    apply(rows, 1, function(row) paste0("| ", paste(row, collapse = " | "), " |"))
  )
}

collapse_cstats <- function(cstats) {
  paste(fmt_num(cstats, 2), collapse = ", ")
}

write_markdown_summary <- function(outcome_stats, results, csv_path, txt_path, md_path) {
  assumed <- results[results$csrsq_label %in% ASSUMED_CSRSQ_LABELS, ]

  conversion_rows <- do.call(rbind, lapply(ASSUMED_CSTATS, function(cstat) {
    conv <- cstat_conversion_components(cstat)
    c(fmt_num(cstat, 2), fmt_num(conv[["d"]], 5),
      fmt_num(conv[["r2_d"]], 5), fmt_num(conv[["royston_r2"]], 5))
  }))

  outcome_lr_rows <- do.call(rbind, lapply(ASSUMED_CSTATS, function(cstat) {
    conv <- cstat_conversion_components(cstat)
    lr <- -outcome_stats$events_observed * log(1 - conv[["royston_r2"]])
    cbind(
      outcome_stats$outcome,
      fmt_num(cstat, 2),
      fmt_int(outcome_stats$n_observed),
      fmt_int(outcome_stats$events_observed),
      fmt_num(lr, 2),
      fmt_num(cstat_to_csrsq_riley(cstat, outcome_stats$n_observed, outcome_stats$events_observed), 4)
    )
  }))

  outcome_rows <- cbind(
    outcome_stats$outcome,
    fmt_int(outcome_stats$n_observed),
    fmt_int(outcome_stats$events_observed),
    fmt_num(outcome_stats$event_fraction, 3),
    fmt_num(outcome_stats$incidence_rate_per_year, 4),
    fmt_num(outcome_stats$mean_followup_years, 2)
  )

  main_c_in_grid <- any(abs(ASSUMED_CSTATS - MAIN_PAPER_CSTAT) < 1e-10)
  main_c_note <- if (main_c_in_grid) {
    paste0("For the main paper, use the table for assumed C = ",
           fmt_num(MAIN_PAPER_CSTAT, 2),
           ". The remaining assumed-C tables are intended for the supplement/sensitivity analysis.")
  } else {
    paste0("MAIN_PAPER_CSTAT is set to ", fmt_num(MAIN_PAPER_CSTAT, 2),
           ", but that value is not included in ASSUMED_CSTAT; add it to ASSUMED_CSTAT if you want a main-paper table generated.")
  }
  cstat_increment_text <- if (length(ASSUMED_CSTATS) > 1) {
    fmt_num(min(diff(ASSUMED_CSTATS)), 2)
  } else {
    "not applicable"
  }

  assumed_table_lines <- unlist(lapply(ASSUMED_CSTATS, function(cstat) {
    subset <- assumed[abs(assumed$assumed_cstat - cstat) < 1e-10, ]
    table_rows <- cbind(
      subset$outcome,
      scenario_label(subset$model_scenario),
      fmt_int(subset$parameters),
      fmt_int(subset$n_observed),
      fmt_int(subset$events_observed),
      fmt_num(subset$csrsquared, 3),
      fmt_int(subset$required_n),
      fmt_int(subset$required_events),
      fmt_num(subset$EPP_obs, 2),
      fmt_num(subset$required_epp, 2),
      fmt_yes_no(subset$adequate_by_n),
      fmt_yes_no(subset$adequate_by_events),
      fmt_yes_no(subset$adequate_by_EPP)
    )
    c(
      paste0("### Assumed C = ", fmt_num(cstat, 2),
             ifelse(abs(cstat - MAIN_PAPER_CSTAT) < 1e-10, " — main-paper assumption", " — supplemental sensitivity")),
      "",
      md_table(c("Outcome", "Scenario", "p", "Observed N", "Observed events",
                 "R²_CS", "Required N", "Required events", "Observed EPP",
                 "Riley EPP", "N adequate?", "Events adequate?", "EPP adequate?"),
               table_rows),
      ""
    )
  }))

  fixed_sens_labels <- c("sensitivity_R2_0.020", "doc_example_R2_0.051",
                         "sensitivity_R2_0.075", "sensitivity_R2_0.100")
  sens_labels <- c(ASSUMED_CSRSQ_LABELS, fixed_sens_labels)
  sens_headers <- c(
    "Outcome", "Scenario", "p",
    sprintf("C=%.2f Riley R²", ASSUMED_CSTATS),
    "R²=0.020", "Doc R²=0.051", "R²=0.075", "R²=0.100"
  )
  scenario_keys <- unique(assumed[, c("outcome", "model_scenario", "parameters")])
  sens_rows <- do.call(rbind, lapply(seq_len(nrow(scenario_keys)), function(i) {
    row <- scenario_keys[i, ]
    subset <- results[results$outcome == row$outcome &
                        results$model_scenario == row$model_scenario &
                        results$parameters == row$parameters, ]
    required <- sapply(sens_labels, function(lbl) {
      match_row <- subset[subset$csrsq_label == lbl, ]
      if (nrow(match_row) == 0) "NA" else fmt_int(match_row$required_n[1])
    })
    c(row$outcome, scenario_label(row$model_scenario), fmt_int(row$parameters), required)
  }))

  adequacy_rows <- do.call(rbind, lapply(ASSUMED_CSTATS, function(cstat) {
    subset <- assumed[abs(assumed$assumed_cstat - cstat) < 1e-10, ]
    pass_all <- subset$adequate_by_n & subset$adequate_by_events & subset$adequate_by_EPP
    c(fmt_num(cstat, 2), fmt_int(sum(pass_all, na.rm = TRUE)), fmt_int(nrow(subset)))
  }))

  lines <- c(
    "# Riley et al. `pmsampsize` calculation for HNC Cox prediction models",
    "",
    "<!-- Auto-generated by pmsampsize_riley_clean.R. Do not edit results by hand; change parameters in the R script and rerun it. -->",
    "",
    "This file summarizes the clean survival-model sample-size calculation produced by [`pmsampsize_riley_clean.R`](pmsampsize_riley_clean.R). The calculation uses the R package `pmsampsize` for **development of a multivariable prediction model with a survival outcome**.",
    "",
    paste0("Input data: `", OUTCOMES_PATH, "`  "),
    paste0("CSV output: [`", basename(csv_path), "`](", basename(csv_path), ")  "),
    paste0("Text output: [`", basename(txt_path), "`](", basename(txt_path), ")  "),
    paste0("Markdown output: [`", basename(md_path), "`](", basename(md_path), ")"),
    "",
    "---",
    "",
    "## 1. What is Riley's sample-size calculation?",
    "",
    "Riley et al. proposed a formal sample-size framework for developing clinical prediction models. For survival outcomes, `pmsampsize(type = \"s\")` estimates the minimum development sample size needed for a Cox prediction model so that overfitting is controlled and predicted risk at the target timepoint is estimated with acceptable precision.",
    "",
    "For Cox models, required sample size depends mainly on candidate predictor parameters, target global shrinkage, expected Cox-Snell R², event incidence rate, mean follow-up, and the prediction time horizon.",
    "",
    "## 2. Inputs used and rationale",
    "",
    paste0("- Survival model type: `pmsampsize(type = \"s\", ...)`"),
    paste0("- Prediction timepoint: ", TIMEPOINT_YEARS, " years"),
    paste0("- Shrinkage: ", SHRINKAGE),
    "- Baseline model: 6 parameters; enhanced model: 9 parameters.",
    paste0("- Assumed C-statistic value(s): ", collapse_cstats(ASSUMED_CSTATS)),
    paste0("- Main-paper assumed C-statistic: ", fmt_num(MAIN_PAPER_CSTAT, 2)),
    paste0("- Documentation comparator Cox-Snell R²: ", fmt_num(DOC_EXAMPLE_CSRSQUARED, 3)),
    "- Event rate: observed outcome incidence rate, events / total person-years.",
    "",
    "## 3. Primary Cox-Snell R² from assumed C-statistic",
    "",
    "The assumed C-statistic is converted to Cox-Snell R² using the Riley/Royston sequence:",
    "",
    "$$D = 5.5(C-0.5) + 10.26(C-0.5)^3$$",
    "",
    "$$R^2_D = \\frac{(\\pi/8)D^2}{\\pi^2/6 + (\\pi/8)D^2}$$",
    "",
    "$$R^2_{Royston} = \\frac{-(\\pi^2/6)R^2_D}{(\\pi^2/6)R^2_D - 1}$$",
    "",
    "$$LR = -E\\log(1-R^2_{Royston})$$",
    "",
    "$$R^2_{CS} = 1 - \\exp\\left(-\\frac{LR}{N}\\right)$$",
    "",
    md_table(c("Assumed C", "D", "R²_D", "Royston R²"), conversion_rows),
    "",
    md_table(c("Outcome", "Assumed C", "N", "Events", "LR", "C-derived R²_CS"), outcome_lr_rows),
    "",
    "## 4. Outcome-level quantities used as inputs",
    "",
    md_table(c("Outcome", "N", "Events", "Event fraction", "Incidence rate/year", "Mean follow-up, years"), outcome_rows),
    "",
    "Definitions: event fraction = events / N; incidence rate/year = events / total person-years of follow-up.",
    "",
    "## 5. Primary Riley `pmsampsize` results",
    "",
    paste0("Primary results use each assumed C value (", collapse_cstats(ASSUMED_CSTATS), ") converted to Cox-Snell R²."),
    "",
    main_c_note,
    "",
    "Each assumed C value is shown as a separate self-contained table so a single chosen assumption can be copied directly into the manuscript.",
    "",
    assumed_table_lines,
    "",
    "### Interpretation across assumed C values",
    "",
    "Scenarios passing all three checks by assumed C:",
    "",
    md_table(c("Assumed C", "Scenarios adequate by N/events/EPP", "Total scenarios"), adequacy_rows),
    "",
    "Observed EPP is `events_observed / parameters`. Riley EPP is the required events-per-parameter returned by `pmsampsize`.",
    "",
    "## 6. Sensitivity to assumed Cox-Snell R² / assumed C-statistic",
    "",
    "Cells show required N for each assumption.",
    "",
    md_table(sens_headers, sens_rows),
    "",
    "## 7. EPP/sample-size methods paragraph for manuscript",
    "",
    paste0("> We assessed whether the available cohort provided sufficient sample size and events for development of the Cox prediction models using the Riley et al. framework implemented in `pmsampsize` for survival outcomes. Calculations used a 2-year prediction horizon, target global shrinkage of ", fmt_num(SHRINKAGE, 2), ", and the observed outcome-specific event incidence rate, calculated as events divided by total person-years of follow-up, with observed mean follow-up entered in years. Candidate degrees of freedom were set to 6 for the baseline model and 9 for the enhanced models. Because `pmsampsize(type = \"s\")` requires an anticipated Cox-Snell R² rather than a C-statistic, we converted an assumed C-statistic to Cox-Snell R² using the Riley/Royston equations. The main analysis assumed C = ", fmt_num(MAIN_PAPER_CSTAT, 2), ", representing moderate expected discrimination; additional assumptions from C = ", fmt_num(min(ASSUMED_CSTATS), 2), " to C = ", fmt_num(max(ASSUMED_CSTATS), 2), " in increments of ", cstat_increment_text, " were reported as sensitivity analyses in the supplement. For each endpoint and model size, observed EPP was calculated as observed events divided by candidate parameters and compared with the Riley-required EPP, alongside the required total sample size and required number of events."),
    "",
    "## 8. Bottom-line manuscript framing",
    "",
    paste0("> For Riley et al. sample-size assessment, the main-paper assumption was target C = ", fmt_num(MAIN_PAPER_CSTAT, 2), ", converted to Cox-Snell R² using the Riley/Royston equations. Outcome-specific event rates were estimated from the cohort as events per person-year. Model adequacy was assessed against the Riley-required sample size, required event count, and required EPP. Sensitivity analyses over assumed C values of ", collapse_cstats(ASSUMED_CSTATS), " can be reported in the supplement."),
    ""
  )

  writeLines(lines, md_path)
}

summarise_outcome <- function(df, outcome_name, time_col, event_col) {
  if (!time_col %in% names(df)) stop("Missing time column: ", time_col)
  if (!event_col %in% names(df)) stop("Missing event column: ", event_col)

  time_months <- as_num(df[[time_col]])
  event <- as_num(df[[event_col]])
  keep <- !is.na(time_months) & !is.na(event) & time_months > 0 &
    time_months <= 240 & event %in% c(0, 1)

  time_years <- time_months[keep] / 12
  event <- event[keep]
  n <- length(event)
  events <- sum(event == 1)
  total_fup <- sum(time_years)
  mean_fup <- mean(time_years)
  rate <- events / total_fup
  event_fraction <- events / n

  data.frame(
    outcome = outcome_name,
    n_observed = n,
    events_observed = events,
    event_fraction = event_fraction,
    total_followup_years = total_fup,
    mean_followup_years = mean_fup,
    incidence_rate_per_year = rate,
    assumed_cstats = collapse_cstats(ASSUMED_CSTATS),
    first_assumed_cstat = ASSUMED_CSTATS[1],
    first_assumed_cstat_csrsquared = cstat_to_csrsq_riley(ASSUMED_CSTATS[1], n, events),
    cstat_diagnostic = CSTAT_DIAGNOSTIC,
    cstat_diagnostic_csrsquared = cstat_to_csrsq_riley(CSTAT_DIAGNOSTIC, n, events),
    stringsAsFactors = FALSE
  )
}

run_pmsampsize <- function(outcome_row, scenario_name, parameters, csrsq_label,
                           csrsq_value, assumed_cstat = NA_real_) {
  res <- tryCatch({
    pmsampsize(
      type = "s",
      csrsquared = csrsq_value,
      parameters = parameters,
      shrinkage = SHRINKAGE,
      rate = outcome_row$incidence_rate_per_year,
      timepoint = TIMEPOINT_YEARS,
      meanfup = outcome_row$mean_followup_years
    )
  }, error = function(e) e)

  if (inherits(res, "error")) {
    return(data.frame(
      outcome = outcome_row$outcome,
      model_scenario = scenario_name,
      parameters = parameters,
      csrsq_label = csrsq_label,
      assumed_cstat = assumed_cstat,
      csrsquared = csrsq_value,
      n_observed = outcome_row$n_observed,
      events_observed = outcome_row$events_observed,
      event_fraction = outcome_row$event_fraction,
      incidence_rate_per_year = outcome_row$incidence_rate_per_year,
      mean_followup_years = outcome_row$mean_followup_years,
      timepoint_years = TIMEPOINT_YEARS,
      EPP_obs = outcome_row$events_observed / parameters,
      required_n = NA_integer_,
      required_events = NA_integer_,
      required_person_time = NA_real_,
      required_epp = NA_real_,
      adequate_by_n = NA,
      adequate_by_events = NA,
      adequate_by_EPP = NA,
      binding_criterion = NA_character_,
      error = conditionMessage(res),
      stringsAsFactors = FALSE
    ))
  }

  ss <- res$results_table
  binding <- rownames(ss)[which.max(ss[, "Samp_size"])]
  required_n <- round_int(res$sample_size)
  required_events <- round_int(res$events)

  data.frame(
    outcome = outcome_row$outcome,
    model_scenario = scenario_name,
    parameters = parameters,
    csrsq_label = csrsq_label,
    assumed_cstat = assumed_cstat,
    csrsquared = csrsq_value,
    n_observed = outcome_row$n_observed,
    events_observed = outcome_row$events_observed,
    event_fraction = outcome_row$event_fraction,
    incidence_rate_per_year = outcome_row$incidence_rate_per_year,
    mean_followup_years = outcome_row$mean_followup_years,
    timepoint_years = TIMEPOINT_YEARS,
    EPP_obs = outcome_row$events_observed / parameters,
    required_n = required_n,
    required_events = required_events,
    required_person_time = res$tot_per_yrs_final,
    required_epp = res$EPP,
    adequate_by_n = outcome_row$n_observed >= required_n,
    adequate_by_events = outcome_row$events_observed >= required_events,
    adequate_by_EPP = (outcome_row$events_observed / parameters) >= res$EPP,
    binding_criterion = binding,
    error = NA_character_,
    stringsAsFactors = FALSE
  )
}

main <- function() {
  df <- read.csv(OUTCOMES_PATH, check.names = FALSE)

  outcome_stats <- do.call(rbind, lapply(names(OUTCOMES), function(nm) {
    spec <- OUTCOMES[[nm]]
    summarise_outcome(df, nm, spec$time, spec$event)
  }))

  rows <- list()
  i <- 1L
  for (r in seq_len(nrow(outcome_stats))) {
    oc <- outcome_stats[r, ]

    assumed_csrsq_values <- setNames(
      sapply(ASSUMED_CSTATS, function(cstat) {
        cstat_to_csrsq_riley(cstat, oc$n_observed, oc$events_observed)
      }),
      ASSUMED_CSRSQ_LABELS
    )
    fixed_csrsq_values <- c(
      sensitivity_R2_0.020 = 0.020,
      doc_example_R2_0.051 = DOC_EXAMPLE_CSRSQUARED,
      sensitivity_R2_0.075 = 0.075,
      sensitivity_R2_0.100 = 0.100
    )
    diagnostic_csrsq_value <- if (any(abs(ASSUMED_CSTATS - CSTAT_DIAGNOSTIC) < 1e-10)) {
      numeric(0)
    } else {
      setNames(oc$cstat_diagnostic_csrsquared, DIAGNOSTIC_CSRSQ_LABEL)
    }

    csrsq_values <- c(assumed_csrsq_values, fixed_csrsq_values, diagnostic_csrsq_value)
    assumed_cstat_lookup <- setNames(rep(NA_real_, length(csrsq_values)), names(csrsq_values))
    assumed_cstat_lookup[ASSUMED_CSRSQ_LABELS] <- ASSUMED_CSTATS

    # Ensure the primary row appears first and avoid duplicate 0.051 if users
    # add it to the sensitivity vector later.
    csrsq_values <- csrsq_values[!duplicated(names(csrsq_values))]

    for (sc_name in names(PARAMETER_SCENARIOS)) {
      p <- PARAMETER_SCENARIOS[[sc_name]]
      for (cs_label in names(csrsq_values)) {
        rows[[i]] <- run_pmsampsize(
          oc, sc_name, p, cs_label, unname(csrsq_values[[cs_label]]),
          assumed_cstat = unname(assumed_cstat_lookup[[cs_label]])
        )
        i <- i + 1L
      }
    }
  }

  results <- do.call(rbind, rows)

  csv_path <- file.path(OUT_DIR, "pmsampsize_riley_clean_results.csv")
  txt_path <- file.path(OUT_DIR, "pmsampsize_riley_clean_results.txt")
  md_path <- file.path(OUT_DIR, "pmsampsize_riley_clean_results.md")
  write.csv(results, csv_path, row.names = FALSE)
  write_markdown_summary(outcome_stats, results, csv_path, txt_path, md_path)

  sink(txt_path)
  on.exit(sink(), add = TRUE)

  cat(strrep("=", 78), "\n")
  cat("Clean pmsampsize Riley calculation for HNC Cox models\n")
  cat(strrep("=", 78), "\n\n")
  cat("Input outcome data:", OUTCOMES_PATH, "\n")
  cat("pmsampsize type      : survival ('s')\n")
  cat("timepoint            :", TIMEPOINT_YEARS, "years\n")
  cat("shrinkage            :", SHRINKAGE, "\n")
  cat("assumed C-stat values:", collapse_cstats(ASSUMED_CSTATS), "\n")
  cat("assumed csrsquared   : derived from each assumed C using Riley/Royston equations\n")
  cat("doc example R²       :", DOC_EXAMPLE_CSRSQUARED, "(sensitivity comparator, not primary)\n")
  cat("rate input           : observed outcome incidence rate = events / total person-years\n\n")

  cat("Outcome-level quantities estimated from all_fields_processed.csv\n")
  cat("----------------------------------------------------------------\n")
  print(outcome_stats[, c("outcome", "n_observed", "events_observed", "event_fraction",
                         "incidence_rate_per_year", "mean_followup_years",
                         "assumed_cstats", "first_assumed_cstat_csrsquared",
                         "cstat_diagnostic", "cstat_diagnostic_csrsquared")], row.names = FALSE)

  cat("\nPrimary pmsampsize results using assumed C value(s) =", collapse_cstats(ASSUMED_CSTATS), "converted to Cox-Snell R²\n")
  cat("----------------------------------------------------------------\n")
  primary <- results[results$csrsq_label %in% ASSUMED_CSRSQ_LABELS, ]
  print(primary[, c("outcome", "model_scenario", "parameters", "n_observed",
                   "events_observed", "assumed_cstat", "csrsquared",
                   "incidence_rate_per_year", "mean_followup_years",
                   "required_n", "required_events", "EPP_obs", "required_epp",
                   "adequate_by_n", "adequate_by_events", "adequate_by_EPP",
                   "binding_criterion")],
        row.names = FALSE)

  cat("\nSensitivity results\n")
  cat("----------------------------------------------------------------\n")
  print(results[, c("outcome", "model_scenario", "parameters", "csrsq_label",
                   "assumed_cstat", "csrsquared", "required_n", "required_events",
                   "EPP_obs", "required_epp", "adequate_by_EPP",
                   "adequate_by_n", "binding_criterion")], row.names = FALSE)

  cat("\nInterpretation notes\n")
  cat("----------------------------------------------------------------\n")
  cat("1. For type='s', pmsampsize needs csrsquared or nagrsquared.\n")
  cat("   The cstatistic argument in the package documentation is for binary outcomes,\n")
  cat("   so this script does not pass cstatistic to pmsampsize.\n")
  cat(sprintf("2. Primary csrsquared values are derived from assumed C value(s) %s using\n", collapse_cstats(ASSUMED_CSTATS)))
  cat("   the Riley/Royston conversion equations, then passed to pmsampsize(type='s').\n")
  cat("3. The pmsampsize documentation example value 0.051 is retained as a\n")
  cat("   sensitivity comparator only; it is not treated as universal/default.\n")
  cat("4. prevalence, intercept, sd, and mmoe are not survival-model inputs here.\n")

  cat("\nSaved CSV:", csv_path, "\n")
  cat("Saved TXT:", txt_path, "\n")
  cat("Saved MD :", md_path, "\n")
}

main()
