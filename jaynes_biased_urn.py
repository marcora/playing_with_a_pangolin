"""
==========================================================================
Bayesian Network: The Biased Urn (Bernoulli Trials, Unknown Rate)
==========================================================================

E.T. Jaynes — "Probability Theory: The Logic of Science" (PTLOS), Ch. 6
"Elementary Parameter Estimation"

The Story
---------
An urn contains balls of two colours (red / white) in unknown proportions.
We draw N balls *with replacement* and record the colour each time.
After k red results in N draws, what should a rational agent believe about
θ — the true fraction of red balls?

Jaynes answers with Bayes' theorem.  Here we cast this as a Bayesian
Network (DAG) and use the Pangolin PPL + BlackJAX/NUTS to do exact
posterior inference, then verify against the analytical Beta posterior.

Bayesian Network (DAG)
----------------------

        ┌──────────────────────┐
        │  θ ~ Beta(α, β)      │  ← prior: our belief before any draws
        └───────────┬──────────┘
                    │  shared parameter
          ┌─────────┼─────────────┐
          ▼                       ▼
  ┌───────────────┐     ┌────────────────┐
  │ k ~ Bin(N,θ) │     │ y* ~ Bern(θ)  │
  │  (observed)  │     │  (predictive)  │
  └───────────────┘     └────────────────┘

  k is the *sufficient statistic* for θ (Jaynes PTLOS §6.2):
  conditioning on k is equivalent to conditioning on all N individual flips.

Analytical posterior (conjugate Beta-Binomial model):
    θ | k, N ~ Beta(α + k, β + N − k)

Laplace's Rule of Succession (PTLOS §6.6):
    P(y* = red | k, N) = E[θ | k, N] = (α + k) / (α + β + N)
    For α = β = 1 (uniform prior):  P = (k + 1) / (N + 2)

Key Jaynes insight (PTLOS §6.3):
    "The posterior is determined entirely by the likelihood and the prior;
     the order in which the data arrives is irrelevant."

==========================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats

import pangolin as pg
from pangolin import interface as pi
from pangolin import ir
from pangolin.jax_backend.bijectors import default_bijector_dict

# ── 0.  Style ──────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)

# ── 1.  Problem setup ──────────────────────────────────────────────────────
#  Prior  θ ~ Beta(α, β)
#  α = β = 1  →  uniform (Laplace's "principle of insufficient reason")
#  This expresses complete prior ignorance about θ.
ALPHA_PRIOR = 1.0
BETA_PRIOR  = 1.0

#  Data: N draws, k red outcomes
N = 14       # total draws
K = 10       # observed red (≈ 71 %)

#  Analytical (conjugate) posterior
ALPHA_POST = ALPHA_PRIOR + K
BETA_POST  = BETA_PRIOR  + (N - K)

prior_dist = stats.beta(ALPHA_PRIOR, BETA_PRIOR)
post_dist  = stats.beta(ALPHA_POST,  BETA_POST)

post_mean  = ALPHA_POST / (ALPHA_POST + BETA_POST)
post_mode  = (ALPHA_POST - 1) / (ALPHA_POST + BETA_POST - 2)  # valid for α,β > 1
post_std   = post_dist.std()

# Laplace's rule of succession
p_next_analytical = (ALPHA_PRIOR + K) / (ALPHA_PRIOR + BETA_PRIOR + N)

# MLE for comparison
theta_mle = K / N

# ── 2.  Pangolin model definition ──────────────────────────────────────────
#
#  Node 1:  θ ~ Beta(α, β)           — prior on urn composition
#  Node 2:  k ~ Binomial(N, θ)       — observed count (sufficient statistic)
#  Node 3:  y_new ~ Bernoulli(θ)     — predictive: next draw
#
theta   = pi.beta(ALPHA_PRIOR, BETA_PRIOR)     # continuous latent (→ NUTS ok)
k_obs   = pi.binomial(N, theta)                # discrete observed  (→ conditioning ok)
# NOTE: y_new ~ Bernoulli(θ) is discrete; BlackJAX/NUTS requires all latent variables
# to be continuous, so y_new is shown in the graph only and used post-hoc below.
y_new   = pi.bernoulli(theta)                  # for DAG display only

print("=" * 66)
print("   BAYESIAN NETWORK: BIASED URN  (Jaynes PTLOS Ch. 6)")
print("=" * 66)
print(f"\n  Prior       :  θ ~ Beta({ALPHA_PRIOR:.0f}, {BETA_PRIOR:.0f})")
print(f"  Experiment  :  N = {N} draws,  k = {K} red")
print(f"  Posterior   :  θ | k,N ~ Beta({ALPHA_POST:.0f}, {BETA_POST:.0f})\n")
print("  Pangolin upstream graph for k_obs:")
pi.print_upstream(k_obs)
print("\n  Pangolin upstream graph for y_new (predictive):")
pi.print_upstream(y_new)
del y_new    # not passed to NUTS — drawn post-hoc as posterior predictive (see below)

print(f"\n  Analytical posterior mean  = {post_mean:.4f}")
print(f"  Analytical posterior mode  = {post_mode:.4f}")
print(f"  Analytical posterior std   = {post_std:.4f}")
print(f"  MLE  θ̂ = k/N               = {theta_mle:.4f}")
print(f"  Laplace rule  P(y*=red)   = {p_next_analytical:.4f}")

# ── 3.  Posterior inference with BlackJAX / NUTS ───────────────────────────
NITER = 5_000
print(f"\n  Running NUTS  ({NITER} iterations) …")

# BlackJAX's constrain step looks up every distribution in the bijector dict.
# Discrete distributions are not in the default dict, so we extend it.
# Setting them to None tells the backend: "no bijection needed" (pass-through).
discrete_passthrough = {
    ir.Binomial:     None,
    ir.Bernoulli:    None,
    ir.BernoulliLogit: None,
    ir.Categorical:  None,
    ir.Poisson:      None,
    ir.BetaBinomial: None,
}
extended_bijector_dict = {**default_bijector_dict, **discrete_passthrough}

#  Step 1 — sample θ from posterior p(θ | k=K) with NUTS (continuous latent only)
theta_samps = pg.blackjax.sample(
    theta,                    # query: continuous latent (→ NUTS ok)
    [k_obs],                  # evidence variable
    [K],                      # observed value
    niter=NITER,
    bijector_dict=extended_bijector_dict,
)
theta_samps = np.asarray(theta_samps)
mcmc_mean   = theta_samps.mean()
mcmc_std    = theta_samps.std()

#  Step 2 — posterior predictive for y_new ~ Bernoulli(θ)
#  P(y*=red | k, N)  =  E[θ | k, N]   by the law of total expectation.
#  One Bernoulli draw per posterior θ sample gives the full predictive distribution.
rng          = np.random.default_rng(0)
y_new_samps  = rng.binomial(1, theta_samps)   # shape (NITER,)
p_next_mcmc  = y_new_samps.mean()             # ≈ E[θ|k]  =  Laplace's rule

print(f"\n  NUTS posterior mean        = {mcmc_mean:.4f}  "
      f"(analytical: {post_mean:.4f},  error: {abs(mcmc_mean - post_mean):.4f})")
print(f"  NUTS posterior std         = {mcmc_std:.4f}  "
      f"(analytical: {post_std:.4f},  error: {abs(mcmc_std - post_std):.4f})")
print(f"  NUTS P(y*=red)             = {p_next_mcmc:.4f}  "
      f"(analytical: {p_next_analytical:.4f},  error: {abs(p_next_mcmc - p_next_analytical):.4f})")

# ── 4.  Build analytical learning curves (no extra NUTS) ──────────────────
#
#  As we accumulate draws (k_i red in n_i total), the posterior evolves:
#      θ | k_i, n_i ~ Beta(α + k_i, β + n_i − k_i)
#
#  We simulate a run of N draws with the true θ = K/N (to keep it consistent)
#  and show how the posterior mean and 95 % CI tighten.
#
rng = np.random.default_rng(42)
draws = rng.binomial(1, K / N, size=N)   # simulated sequence of 0/1 results

ns       = np.arange(1, N + 1)
cum_k    = np.cumsum(draws)
alpha_n  = ALPHA_PRIOR + cum_k
beta_n   = BETA_PRIOR  + (ns - cum_k)

post_mean_n  = alpha_n / (alpha_n + beta_n)
post_lo_n    = stats.beta.ppf(0.025, alpha_n, beta_n)
post_hi_n    = stats.beta.ppf(0.975, alpha_n, beta_n)

# ── 5.  Visualisation ─────────────────────────────────────────────────────
theta_grid = np.linspace(1e-4, 1 - 1e-4, 400)
prior_pdf  = prior_dist.pdf(theta_grid)
post_pdf   = post_dist.pdf(theta_grid)

# Four-panel figure
fig = plt.figure(figsize=(14, 10))
gs  = gridspec.GridSpec(2, 2, hspace=0.42, wspace=0.35)
ax1 = fig.add_subplot(gs[0, 0])   # Bayesian update
ax2 = fig.add_subplot(gs[0, 1])   # Prior sensitivity
ax3 = fig.add_subplot(gs[1, 0])   # Learning curve
ax4 = fig.add_subplot(gs[1, 1])   # DAG diagram

fig.suptitle(
    "Bayesian Network — Biased Urn   "
    "(Jaynes PTLOS, Ch. 6)\n"
    f"Prior: Beta({ALPHA_PRIOR:.0f},{BETA_PRIOR:.0f})  ·  "
    f"Data: N={N}, k={K}  ·  "
    f"Posterior: Beta({ALPHA_POST:.0f},{BETA_POST:.0f})",
    fontsize=13, y=1.01,
)

# ── Panel A: Bayesian update ───────────────────────────────────────────────
ax1.plot(theta_grid, prior_pdf, lw=2, ls="--", color="steelblue",
         label=f"Prior  Beta({ALPHA_PRIOR:.0f},{BETA_PRIOR:.0f})")
ax1.plot(theta_grid, post_pdf,  lw=2.5, color="tomato",
         label=f"Posterior  Beta({ALPHA_POST:.0f},{BETA_POST:.0f})  [analytical]")
ax1.hist(theta_samps, bins=60, density=True, alpha=0.35, color="tomato",
         label=f"Posterior  [NUTS, n={NITER:,}]")
ax1.axvline(theta_mle, color="grey",     ls=":",  lw=1.5, label=f"MLE  θ̂={theta_mle:.2f}")
ax1.axvline(post_mean, color="tomato",   ls=":",  lw=1.5, label=f"Post. mean={post_mean:.2f}")
ax1.axvline(ALPHA_PRIOR / (ALPHA_PRIOR + BETA_PRIOR),
            color="steelblue", ls=":",  lw=1.5, label="Prior mean=0.50")

# Shade 95 % posterior credible interval
lo95 = post_dist.ppf(0.025)
hi95 = post_dist.ppf(0.975)
mask = (theta_grid >= lo95) & (theta_grid <= hi95)
ax1.fill_between(theta_grid[mask], 0, post_pdf[mask], color="tomato", alpha=0.12,
                 label=f"95 % CI  [{lo95:.2f}, {hi95:.2f}]")

ax1.set_xlabel("θ  (fraction of red balls)")
ax1.set_ylabel("Density")
ax1.set_title("A)  Bayesian Update: Prior → Posterior")
ax1.legend(fontsize=8, loc="upper left")

# ── Panel B: Prior sensitivity ────────────────────────────────────────────
prior_specs = [
    (1,  1,  "Uniform   Beta(1,1)",         "steelblue"),
    (2,  2,  "Mild      Beta(2,2)",         "mediumseagreen"),
    (5,  5,  "Informative  Beta(5,5)",      "darkorange"),
    (1,  5,  "Sceptical Beta(1,5)",         "mediumpurple"),
    (10, 10, "Strong     Beta(10,10)",      "crimson"),
]

for (a, b, lbl, col) in prior_specs:
    ap, bp = a + K, b + (N - K)
    ax2.plot(theta_grid, stats.beta.pdf(theta_grid, ap, bp), lw=2, color=col,
             label=f"{lbl}  →  Post({ap},{bp})")

ax2.axvline(theta_mle, color="grey", ls=":", lw=1.5, label=f"MLE={theta_mle:.2f}")
ax2.set_xlabel("θ")
ax2.set_ylabel("Posterior density")
ax2.set_title("B)  Posterior Sensitivity to Prior")
ax2.legend(fontsize=7.5)

# ── Panel C: Sequential belief update (learning curve) ────────────────────
ax3.fill_between(ns, post_lo_n, post_hi_n, alpha=0.25, color="tomato",
                 label="95 % credible interval")
ax3.plot(ns, post_mean_n, lw=2, color="tomato", label="Posterior mean  E[θ|data]")
ax3.axhline(K / N, color="grey",   ls=":",  lw=1.5, label=f"True rate  k/N = {K/N:.2f}")
ax3.axhline(0.5,   color="steelblue", ls="--", lw=1,   label="Prior mean = 0.50")

ax3.set_xlabel("Number of draws  n")
ax3.set_ylabel("Belief about  θ")
ax3.set_title("C)  Sequential Updating (Jaynes: order doesn't matter)")
ax3.legend(fontsize=8)
ax3.set_xlim(1, N)
ax3.set_ylim(0, 1)

# ── Panel D: DAG diagram (hand-drawn with patches) ────────────────────────
ax4.set_xlim(0, 10)
ax4.set_ylim(0, 10)
ax4.set_aspect("equal")
ax4.axis("off")
ax4.set_title("D)  Bayesian Network (DAG)", pad=8)

from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

def dag_node(ax, cx, cy, text, color, radius=0.8):
    circ = plt.Circle((cx, cy), radius, color=color, ec="white", lw=2, zorder=3)
    ax.add_patch(circ)
    ax.text(cx, cy, text, ha="center", va="center", fontsize=10, fontweight="bold",
            color="white", zorder=4)

def dag_arrow(ax, x0, y0, x1, y1):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="->", lw=1.8, color="#444444"),
                zorder=2)

# θ node (top center)
dag_node(ax4, 5, 7.8, "θ",         "#2a6496", radius=0.75)
# k node (lower left — observed, grey background)
dag_node(ax4, 2.5, 4.5, "k",       "#888888", radius=0.75)
# y* node (lower right — predictive, green)
dag_node(ax4, 7.5, 4.5, "y*",      "#2ca089", radius=0.75)

dag_arrow(ax4, 4.35, 7.3, 3.1,  5.2)
dag_arrow(ax4, 5.65, 7.3, 6.9,  5.2)

# Labels on nodes
ax4.text(2.5, 3.35,  "Bin(N, θ)\nobserved",  ha="center", va="top",  fontsize=8.5,
         color="#555555")
ax4.text(7.5, 3.35,  "Bern(θ)\npredictive", ha="center", va="top",  fontsize=8.5,
         color="#555555")
ax4.text(5.0, 9.1,   "Beta(α, β)\nprior",     ha="center", va="bottom", fontsize=8.5,
         color="#555555")

# Observed evidence box around k node
rect = FancyBboxPatch((1.4, 3.4), 2.2, 2.2, boxstyle="round,pad=0.15",
                      fill=False, ec="#888888", lw=1.5, ls="--", zorder=1)
ax4.add_patch(rect)
ax4.text(2.5, 2.95, "(observed)", ha="center", va="top", fontsize=7.5, color="#888888")

# Summary box
summary = (
    f"N = {N} draws,  k = {K} red\n"
    f"α={ALPHA_PRIOR:.0f}, β={BETA_PRIOR:.0f}  →  α'={ALPHA_POST:.0f}, β'={BETA_POST:.0f}\n"
    f"E[θ|k] = {post_mean:.3f}   (MLE = {theta_mle:.3f})\n"
    f"P(y*=red|k) = {p_next_analytical:.3f}  [Laplace rule]\n"
    f"NUTS:  E[θ|k] = {mcmc_mean:.3f},  P(y*) = {p_next_mcmc:.3f}"
)
ax4.text(5, 1.5, summary, ha="center", va="center", fontsize=8.5,
         bbox=dict(boxstyle="round,pad=0.5", fc="#f5f5f5", ec="#cccccc", lw=1),
         family="monospace")

plt.savefig("jaynes_biased_urn.png", dpi=150, bbox_inches="tight")
plt.show()
print("\n  Figure saved → jaynes_biased_urn.png")
print("=" * 66)
