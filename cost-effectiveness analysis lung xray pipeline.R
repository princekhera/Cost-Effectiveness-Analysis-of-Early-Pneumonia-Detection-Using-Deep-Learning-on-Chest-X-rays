##########################################################################
# CEA: AI-assisted pneumonia detection (Decision tree + PSA)
# - Deterministic results + 5000-run PSA
# - Outputs: expected cost/QALY per arm, ICER, CE plane, CEAC, tornado-like OWSA
# Author: (you) - adapt parameters & sources
##########################################################################
# -----------------------------
# 0. Setup: packages & helpers
# -----------------------------
required_pkgs <- c("dplyr", "tibble", "ggplot2", "purrr", "readr", "tidyr")
inst <- required_pkgs[!(required_pkgs %in% installed.packages()[,"Package"])]
if(length(inst)) install.packages(inst, repos = "https://cloud.r-project.org")

library(dplyr); library(tibble); library(ggplot2); library(purrr); library(readr); library(tidyr)

# Helper: convert mean+sd to Beta distribution parameters
beta_params_from_mean_sd <- function(mean, sd){
  var <- sd^2
  # ensure valid
  if(mean <= 0 | mean >= 1) stop("mean must be between 0 and 1 for Beta")
  tmp <- (mean * (1 - mean) / var) - 1
  alpha <- mean * tmp
  beta  <- (1 - mean) * tmp
  if(alpha <= 0 | beta <= 0) stop("Invalid sd for Beta (too large). Adjust sd or use bounds.")
  return(list(alpha = alpha, beta = beta))
}

# Helper: convert mean+sd to Gamma parameters (shape, rate)
gamma_params_from_mean_sd <- function(mean, sd){
  var <- sd^2
  shape <- (mean^2) / var
  rate  <- mean / var
  if(shape <= 0 | rate <= 0) stop("Invalid mean/sd for Gamma")
  return(list(shape = shape, rate = rate))
}

# -----------------------------
# 1. Base parameters (edit these)
# -----------------------------
# Perspective: NHS payer. Time horizon: short (days to months). No discounting.
params <- tibble::tribble(
  ~name,            ~value,    ~sd,     ~type,       ~notes,
  "prevalence",     0.10,      0.02,    "prob",     "Prevalence in screened pop (example)",
  "Se_AI",          0.92,      0.03,    "prob",     "AI sensitivity",
  "Sp_AI",          0.88,      0.03,    "prob",     "AI specificity",
  "Se_std",         0.80,      0.04,    "prob",     "Standard sensitivity",
  "Sp_std",         0.90,      0.03,    "prob",     "Standard specificity",
  "cost_screen_AI", 50,        10,      "cost",     "AI per-screen cost (inference + infra amort.)",
  "cost_screen_std",30,        8,       "cost",     "Standard per-screen cost (radiologist/clinic)",
  "cost_early",     500,       100,     "cost",     "Cost of early-treated pneumonia",
  "cost_late",      4000,      800,     "cost",     "Cost of late/missed pneumonia (hospital/ICU)",
  "cost_fp_follow", 100,       30,      "cost",     "Follow-up cost for FP",
  "u_mild",         0.75,      0.05,    "util",     "Utility during mild pneumonia",
  "u_sev",          0.50,      0.07,    "util",     "Utility during severe pneumonia",
  "d_mild_days",    14,        2,       "time",     "Duration mild (days)",
  "d_sev_days",     30,        5,       "time",     "Duration severe (days)"
)

# Turn into named list for deterministic use
param_list <- params %>% select(name, value) %>% deframe()

# Useful conversions
d_mild <- param_list["d_mild_days"] / 365
d_sev  <- param_list["d_sev_days"] / 365

# -----------------------------
# 2. Deterministic expected cost & QALY (per screened patient)
# -----------------------------
calc_expected <- function(p){
  # p is a named numeric vector of base values
  P <- p["prevalence"]
  Se_AI <- p["Se_AI"]; Sp_AI <- p["Sp_AI"]
  Se_std <- p["Se_std"]; Sp_std <- p["Sp_std"]
  
  # Branch probs AI
  TP_AI <- Se_AI * P
  FN_AI <- (1 - Se_AI) * P
  TN_AI <- Sp_AI * (1 - P)
  FP_AI <- (1 - Sp_AI) * (1 - P)
  
  # Branch probs Standard
  TP_std <- Se_std * P
  FN_std <- (1 - Se_std) * P
  TN_std <- Sp_std * (1 - P)
  FP_std <- (1 - Sp_std) * (1 - P)
  
  # Costs
  c_ai <- p["cost_screen_AI"]; c_std <- p["cost_screen_std"]
  c_early <- p["cost_early"]; c_late <- p["cost_late"]; c_fp <- p["cost_fp_follow"]
  
  # Utilities
  u_mild <- p["u_mild"]; u_sev <- p["u_sev"]
  q_mild <- u_mild * d_mild
  q_sev  <- u_sev  * d_sev
  
  # Expected costs
  E_cost_AI <- TP_AI * (c_ai + c_early) +
    FN_AI * (c_ai + c_late) +
    TN_AI * (c_ai) +
    FP_AI * (c_ai + c_fp)
  
  E_cost_std <- TP_std * (c_std + c_early) +
    FN_std * (c_std + c_late) +
    TN_std * (c_std) +
    FP_std * (c_std + c_fp)
  
  # Expected QALYs (short-horizon loss/gain due to illness)
  E_qaly_AI <- TP_AI * q_mild + FN_AI * q_sev
  E_qaly_std <- TP_std * q_mild + FN_std * q_sev
  
  # Output
  tibble::tibble(
    arm = c("AI","Standard"),
    expected_cost = c(E_cost_AI, E_cost_std),
    expected_qaly = c(E_qaly_AI, E_qaly_std)
  )
}

det_results <- calc_expected(param_list)
det_results
# ICER
delta_cost <- det_results$expected_cost[det_results$arm=="AI"] - det_results$expected_cost[det_results$arm=="Standard"]
delta_qaly <- det_results$expected_qaly[det_results$arm=="AI"] - det_results$expected_qaly[det_results$arm=="Standard"]
icer_det <- delta_cost / delta_qaly
cat("\nDeterministic results:\n")
print(det_results)
cat("\nDelta cost:", round(delta_cost,2), "Delta QALY:", signif(delta_qaly,6), "\n")
cat("ICER (deterministic) = ", ifelse(is.finite(icer_det), round(icer_det,2), "NA (dominated)"), "\n\n")

# -----------------------------
# 3. PSA: Monte Carlo simulation
# -----------------------------
set.seed(12345)
n_sim <- 5000

# Build sampling distributions from mean+sd (use helper functions)
# We'll sample probabilities with Beta, costs with Gamma, utilities with Beta
# Define functions to safe-get alpha/beta or shape/rate; if sd is zero or too small, use small jitter.

sample_parameters <- function(params_tbl, n){
  out <- tibble::tibble(sim = 1:n)
  
  for(i in seq_len(nrow(params_tbl))){
    nm <- params_tbl$name[i]
    m  <- params_tbl$value[i]
    sd <- params_tbl$sd[i]
    typ <- params_tbl$type[i]
    
    if(typ == "prob" | typ == "util"){
      # Beta
      # if sd is zero or NA, make a small sd
      if(is.na(sd) || sd <= 0) sd <- max(0.01, 0.1 * m)
      pr <- tryCatch(beta_params_from_mean_sd(m, sd),
                     error = function(e){
                       # fallback: small variance
                       s2 <- (m*(1-m))/1000
                       beta_params_from_mean_sd(m, sqrt(s2))
                     })
      alpha <- pr$alpha; beta <- pr$beta
      samp <- rbeta(n, shape1 = alpha, shape2 = beta)
    } else if(typ == "cost"){
      # Gamma
      if(is.na(sd) || sd <= 0) sd <- max(1, 0.2 * m)
      gp <- gamma_params_from_mean_sd(m, sd)
      samp <- rgamma(n, shape = gp$shape, rate = gp$rate)
    } else if(typ == "time"){
      # keep time fixed (deterministic). Convert to numeric for consistency.
      samp <- rep(m, n)
    } else {
      samp <- rep(m, n)
    }
    out <- bind_cols(out, tibble::tibble(!!nm := samp))
  }
  return(out)
}

psa_inputs <- sample_parameters(params, n_sim)

# Compute expected cost and QALY per simulation
compute_per_sim <- function(row){
  pvec <- as.numeric(row)
  names(pvec) <- names(row)
  # filter out sim column if present
  pvec <- pvec[!(names(pvec) == "sim")]
  calc_expected(pvec) %>%
    mutate(sim = row["sim"])
}

# Vectorized calculation: use dplyr rowwise approach
psa_results <- psa_inputs %>%
  rowwise() %>%
  do({
    row <- as.list(.)
    p_named <- row
    p_named$sim <- NULL
    names(p_named) <- names(psa_inputs)[names(psa_inputs) != "sim"]
    # convert to numeric named vector
    pnum <- unlist(p_named)
    calc_expected(pnum)
  }) %>%
  ungroup() %>%
  mutate(sim = rep(psa_inputs$sim, each = 2)) # each sim has two rows

# Wide format for delta calculations
psa_wide <- psa_results %>%
  pivot_wider(names_from = arm, values_from = c(expected_cost, expected_qaly), names_sep = ".") %>%
  mutate(delta_cost = expected_cost.AI - expected_cost.Standard,
         delta_qaly = expected_qaly.AI - expected_qaly.Standard,
         icer = delta_cost / delta_qaly)

# Basic PSA summaries
mean_delta_cost <- mean(psa_wide$delta_cost, na.rm=TRUE)
mean_delta_qaly <- mean(psa_wide$delta_qaly, na.rm=TRUE)
mean_icer_psa <- mean(psa_wide$icer[is.finite(psa_wide$icer)], na.rm=TRUE)

cat("PSA means: mean delta cost =", round(mean_delta_cost,2),
    "mean delta QALY =", signif(mean_delta_qaly,6), "\n")
cat("Mean ICER across finite simulations (not decision rule) =", round(mean_icer_psa,2), "\n\n")

# -----------------------------
# 4. CE plane
# -----------------------------
ce_plane <- ggplot(psa_wide, aes(x = delta_qaly, y = delta_cost)) +
  geom_point(alpha = 0.3, size = 0.8) +
  geom_hline(yintercept = 0, linetype = "dashed") +
  geom_vline(xintercept = 0, linetype = "dashed") +
  labs(title = "Cost-effectiveness plane (AI vs Standard)",
       x = "Δ QALYs (AI - Standard)",
       y = "Δ Cost (AI - Standard)") +
  theme_minimal()
print(ce_plane)

# -----------------------------
# 5. CEAC
# -----------------------------
# For a set of willingness-to-pay thresholds, compute probability AI is cost-effective
thresholds <- seq(0, 200000, by = 1000)
ceac_df <- tibble::tibble(threshold = thresholds) %>%
  mutate(p_ce = map_dbl(threshold, function(k){
    # net monetary benefit: NMB = k * delta_qaly - delta_cost ; AI is CE if NMB > 0
    mean(psa_wide$delta_qaly * k - psa_wide$delta_cost > 0, na.rm = TRUE)
  }))

ceac_plot <- ggplot(ceac_df, aes(x = threshold, y = p_ce)) +
  geom_line() +
  labs(title = "CEAC: Probability AI is cost-effective",
       x = "Willingness-to-pay per QALY (£)",
       y = "Probability AI is cost-effective") +
  theme_minimal()
print(ceac_plot)

# -----------------------------
# 6. Simple tornado-like OWSA (one-way)
# -----------------------------
# Choose key parameters to vary by +/- 25% or within plausible limits
ow_params <- c("Se_AI","Sp_AI","Se_std","Sp_std","cost_screen_AI","cost_screen_std","cost_early","cost_late","cost_fp_follow","u_mild","u_sev")
base <- param_list

owsa_results <- map_dfr(ow_params, function(pn){
  val <- base[pn]
  low <- ifelse(params$type[params$name==pn] %in% c("prob","util"), max(0.001, val * 0.75), val * 0.75)
  high <- ifelse(params$type[params$name==pn] %in% c("prob","util"), min(0.999, val * 1.25), val * 1.25)
  p_low <- base; p_high <- base
  p_low[pn] <- low; p_high[pn] <- high
  res_low <- calc_expected(p_low)
  res_high <- calc_expected(p_high)
  delta_low <- res_low$expected_cost[res_low$arm=="AI"] - res_low$expected_cost[res_low$arm=="Standard"]
  q_low <- res_low$expected_qaly[res_low$arm=="AI"] - res_low$expected_qaly[res_low$arm=="Standard"]
  icer_low <- delta_low / q_low
  
  delta_high <- res_high$expected_cost[res_high$arm=="AI"] - res_high$expected_cost[res_high$arm=="Standard"]
  q_high <- res_high$expected_qaly[res_high$arm=="AI"] - res_high$expected_qaly[res_high$arm=="Standard"]
  icer_high <- delta_high / q_high
  
  tibble::tibble(param = pn,
                 low = low, high = high,
                 delta_cost_low = delta_low, delta_qaly_low = q_low, icer_low = icer_low,
                 delta_cost_high = delta_high, delta_qaly_high = q_high, icer_high = icer_high)
})

# Simplified tornado: show absolute change in ICER between low and high
owsa_plot_df <- owsa_results %>%
  mutate(icer_low_num = ifelse(is.finite(icer_low), icer_low, NA_real_),
         icer_high_num = ifelse(is.finite(icer_high), icer_high, NA_real_),
         icer_change = icer_high_num - icer_low_num) %>%
  arrange(desc(abs(icer_change))) %>%
  slice(1:10)

# Plot horizontal bars of change
tornado_df <- owsa_plot_df %>%
  select(param, icer_low_num, icer_high_num) %>%
  pivot_longer(cols = starts_with("icer"), names_to = "bound", values_to = "icer_val") %>%
  mutate(bound = ifelse(bound=="icer_low_num","Low","High"))

ggplot(tornado_df, aes(x = param, y = icer_val, fill = bound)) +
  geom_col(position = position_dodge(width = 0.8), width = 0.7) +
  coord_flip() +
  labs(title = "One-way sensitivity (ICER at low & high parameter values)",
       x = "Parameter",
       y = "ICER (£ per QALY)") +
  theme_minimal()

# -----------------------------
# 7. Save results (optional)
# -----------------------------
# write.csv(psa_wide, "psa_results.csv", row.names = FALSE)
# write.csv(det_results, "deterministic_results.csv", row.names = FALSE)
# ggsave("ce_plane.png", ce_plane, width=7, height=5, dpi=300)
# ggsave("ceac_plot.png", ceac_plot, width=7, height=5, dpi=300)

# -----------------------------
# 8. Quick printed summary
# -----------------------------
cat("==== SUMMARY ====\n")
cat("Deterministic expected cost (AI): ", round(det_results$expected_cost[det_results$arm=="AI"],2),
    " Standard: ", round(det_results$expected_cost[det_results$arm=="Standard"],2), "\n")
cat("Deterministic expected QALY (AI): ", signif(det_results$expected_qaly[det_results$arm=="AI"],6),
    " Standard: ", signif(det_results$expected_qaly[det_results$arm=="Standard"],6), "\n")
cat("Deterministic ICER:", ifelse(is.finite(icer_det), round(icer_det,2), "NA/dominated"), "\n")
cat("PSA mean delta cost:", round(mean_delta_cost,2), " mean delta QALY:", signif(mean_delta_qaly,6), "\n")
cat("Proportion of PSA sims where AI is cost-effective at £20,000/QALY:",
    mean(psa_wide$delta_qaly * 20000 - psa_wide$delta_cost > 0, na.rm = TRUE), "\n")
cat("==== End ====\n")
