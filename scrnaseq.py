# =====================================================================
# Differential Gene Expression with Pangolin
# =====================================================================
#
# Goal: infer which genes change expression between two conditions
# (e.g. control vs. treatment) from RNA-seq count data.
#
# Model (Negative Binomial via Gamma-Poisson mixture):
#
#   For gene g, cell c in condition x_c ∈ {0, 1}:
#
#     alpha_g  ~ Normal(3, 2)          # baseline log-expression
#     beta_g   ~ Normal(0, 1)          # log fold-change (LFC), key quantity
#     phi_g    ~ LogNormal(0, 1)       # overdispersion (>0)
#
#     mu_gc    = exp(alpha_g + beta_g * x_c)   # mean count
#     lambda_gc ~ Gamma(phi_g, phi_g / mu_gc)  # NB as Gamma-Poisson mixture
#     y_gc     ~ Poisson(lambda_gc)
#
#   Marginalizing lambda_gc over the Gamma prior yields exactly the
#   Negative Binomial distribution NB(mu_gc, phi_g), but keeping the
#   mixture explicit lets Pangolin handle it natively.
#
# Inference question: given observed counts Y, what is the posterior
# distribution of beta_g?  Is P(beta_g > 0) high (upregulated)?
# =====================================================================

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pangolin as pg
from pangolin import interface as pi

# =====================================================================
# 1. Simulate toy RNA-seq count data
# =====================================================================
rng = np.random.default_rng(42)

gene_names = ["GeneA", "GeneB", "GeneC"]
G = len(gene_names)

# True parameters used to generate data
true_alpha = [2.5, 3.0, 2.0]   # baseline log-expression per gene
true_beta  = [1.5, -1.0, 0.0]  # LFC: upregulated, downregulated, unchanged
true_phi   = [5.0, 4.0, 8.0]   # overdispersion (higher = less overdispersed)

N_ctrl = N_treat = 15           # cells per condition
x_cond = np.array([0] * N_ctrl + [1] * N_treat, dtype=float)
N = len(x_cond)

# Simulate counts: y ~ Gamma-Poisson(mu, phi)
Y_obs = []
for g in range(G):
    mu_g  = np.exp(true_alpha[g] + true_beta[g] * x_cond)
    lam_g = rng.gamma(shape=true_phi[g], scale=mu_g / true_phi[g])
    Y_obs.append(rng.poisson(lam_g))

print("Simulated mean counts (control | treatment):")
for g, name in enumerate(gene_names):
    ctrl_mean  = Y_obs[g][:N_ctrl].mean()
    treat_mean = Y_obs[g][N_ctrl:].mean()
    print(f"  {name}: {ctrl_mean:.1f} | {treat_mean:.1f}  (true LFC = {true_beta[g]:+.1f})")

# =====================================================================
# 2. Define the Pangolin model
# =====================================================================
alphas   = [pi.normal(3.0, 2.0) for _ in range(G)]
betas    = [pi.normal(0.0, 1.0) for _ in range(G)]
log_phis = [pi.normal(0.0, 1.0) for _ in range(G)]
phis     = [pi.exp(lp) for lp in log_phis]

ys = []
for g in range(G):
    # log-linear mean: mu = exp(alpha + beta * condition)
    mu_g  = pi.exp(alphas[g] + betas[g] * x_cond)
    # Gamma mixing distribution (encodes NB overdispersion)
    lam_g = pi.gamma(phis[g], phis[g] / mu_g)
    # Poisson likelihood given the Gamma rate
    ys.append(pi.poisson(lam_g))

# =====================================================================
# 3. Inference
# =====================================================================
# Name the parameters we want to track in the posterior
params = {}
for g, name in enumerate(gene_names):
    params[f"alpha_{name}"] = alphas[g]
    params[f"beta_{name}"]  = betas[g]
    params[f"phi_{name}"]   = phis[g]

# Condition on observed counts and draw posterior samples
idata = pg.blackjax.sample_arviz(params, ys, Y_obs, niter=2000)

# =====================================================================
# 4. Posterior summary
# =====================================================================
beta_names = [f"beta_{name}" for name in gene_names]
summary = az.summary(idata, var_names=beta_names)
print("\nPosterior summary of log fold-changes (beta):")
print(summary[["mean", "sd", "hdi_3%", "hdi_97%"]])

# Posterior probability of being up- or downregulated
print("\nPosterior DE probabilities:")
for g, name in enumerate(gene_names):
    beta_samples = idata.posterior[f"beta_{name}"].values.flatten()
    p_up   = np.mean(beta_samples > 0)
    p_down = np.mean(beta_samples < 0)
    print(
        f"  {name}: P(LFC>0) = {p_up:.2f}, P(LFC<0) = {p_down:.2f}"
        f"  [true LFC = {true_beta[g]:+.1f}]"
    )

# =====================================================================
# 5. Plots
# =====================================================================
# --- Forest plot of LFC posteriors ---
fig, ax = plt.subplots(figsize=(6, 3))
for g, name in enumerate(gene_names):
    beta_samples = idata.posterior[f"beta_{name}"].values.flatten()
    lo, hi = np.percentile(beta_samples, [3, 97])
    ax.plot([lo, hi], [g, g], color="steelblue", lw=2)
    ax.plot(np.mean(beta_samples), g, "o", color="steelblue")
    ax.plot(true_beta[g], g, "x", color="tomato", ms=8, mew=2, label="true" if g == 0 else "")

ax.axvline(0, color="gray", linestyle="--", lw=1)
ax.set_yticks(range(G))
ax.set_yticklabels(gene_names)
ax.set_xlabel("Log Fold-Change (beta)")
ax.set_title("Posterior LFC estimates (94% HDI)\nblue = posterior, red x = true value")
ax.legend()
plt.tight_layout()
plt.show()

# --- Trace plots for LFC parameters ---
az.plot_trace(idata, var_names=beta_names, compact=True)
plt.suptitle("MCMC traces for log fold-changes", y=1.01)
plt.tight_layout()
plt.show()

# --- Posterior distributions ---
az.plot_posterior(idata, var_names=beta_names, ref_val=0)
plt.suptitle("Posterior distributions of log fold-changes", y=1.01)
plt.tight_layout()
plt.show()
