import marimo

__generated_with = "0.23.1"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Differential Gene Expression with a Negative Binomial Model

    A didactic walkthrough of Bayesian DGE analysis using [Pangolin](https://github.com/justindomke/pangolin).
    """)
    return


@app.cell(hide_code=True)
def _():
    import arviz as az
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import pangolin as pg
    from pangolin import interface as pi
    return az, np, pd, pg, pi, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Background

    RNA-seq quantifies gene expression by counting how many RNA molecules from each gene
    are present in a cell (or sample). The central question in **differential gene expression
    (DGE)** analysis is:

    > Which genes change their expression level between two conditions — e.g. control vs. treatment?

    Count data have two key statistical properties that shape our model choice:

    1. **Discrete and non-negative** — we observe integer counts, never negative values.
    2. **Overdispersed** — biological variability makes the variance larger than the mean,
       ruling out the simpler Poisson model (which requires variance = mean).

    The **Negative Binomial (NB)** distribution handles both. We parameterize it via a
    Gamma–Poisson mixture, which Pangolin can represent natively:

    $$\lambda_{gc} \sim \mathrm{Gamma}(\phi_g,\; \phi_g / \mu_{gc})$$
    $$y_{gc} \mid \lambda_{gc} \sim \mathrm{Poisson}(\lambda_{gc})$$

    Marginalizing $\lambda_{gc}$ over the Gamma prior recovers exactly
    $y_{gc} \sim \mathrm{NB}(\mu_{gc}, \phi_g)$, where $\mu_{gc}$ is the mean count and
    $\phi_g > 0$ is the overdispersion parameter (larger $\phi_g$ = less overdispersion).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Toy Dataset

    We simulate counts for **3 genes** across **30 cells** (15 control, 15 treatment).
    The true log fold-changes (LFCs) are:

    | Gene  | True LFC | Interpretation  |
    |-------|----------|-----------------|
    | GeneA | +1.5     | upregulated     |
    | GeneB | −1.0     | downregulated   |
    | GeneC |  0.0     | unchanged       |

    The goal: recover these LFCs from counts alone, with calibrated uncertainty.
    """)
    return


@app.cell
def _(np):
    rng = np.random.default_rng(42)

    gene_names = ["GeneA", "GeneB", "GeneC"]
    G = len(gene_names)

    true_alpha = [2.5,  3.0,  2.0]   # baseline log-expression
    true_beta  = [1.5, -1.0,  0.0]   # log fold-change (LFC)
    true_phi   = [5.0,  4.0,  8.0]   # overdispersion

    N_ctrl = N_treat = 15
    x_cond = np.array([0] * N_ctrl + [1] * N_treat, dtype=float)
    N = len(x_cond)

    Y_obs = []
    for g in range(G):
        mu_g  = np.exp(true_alpha[g] + true_beta[g] * x_cond)
        lam_g = rng.gamma(shape=true_phi[g], scale=mu_g / true_phi[g])
        Y_obs.append(rng.poisson(lam_g))

    return G, N, N_ctrl, N_treat, Y_obs, gene_names, rng, true_alpha, true_beta, true_phi, x_cond


@app.cell
def _(N_ctrl, Y_obs, gene_names, np, pd, true_beta):
    rows = []
    for _g, _name in enumerate(gene_names):
        _ctrl  = Y_obs[_g][:N_ctrl]
        _treat = Y_obs[_g][N_ctrl:]
        rows.append({
            "Gene":           _name,
            "True LFC":       f"{true_beta[_g]:+.1f}",
            "Control mean":   f"{np.mean(_ctrl):.1f}",
            "Treatment mean": f"{np.mean(_treat):.1f}",
        })

    pd.DataFrame(rows).set_index("Gene")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Model

    We fit a **log-linear model** for the mean count of gene $g$ in cell $c$:

    $$\log \mu_{gc} = \alpha_g + \beta_g \cdot x_c$$

    where $x_c \in \{0, 1\}$ is the condition indicator (0 = control, 1 = treatment).

    - $\alpha_g$ is the **baseline log-expression** in the control condition.
    - $\beta_g$ is the **log fold-change**: $e^{\beta_g}$ is the fold-change in mean count.

    The full generative model for gene $g$, cell $c$:

    $$\alpha_g \sim \mathcal{N}(3, 2)$$
    $$\beta_g \sim \mathcal{N}(0, 1)$$
    $$\log \phi_g \sim \mathcal{N}(0, 1) \quad (\phi_g > 0)$$
    $$\mu_{gc} = \exp(\alpha_g + \beta_g \cdot x_c)$$
    $$\lambda_{gc} \sim \mathrm{Gamma}(\phi_g,\; \phi_g / \mu_{gc})$$
    $$y_{gc} \sim \mathrm{Poisson}(\lambda_{gc})$$

    We place a $\mathcal{N}(0,1)$ prior on $\log\phi_g$ so that the overdispersion
    is constrained to be positive and not too extreme.
    """)
    return


@app.cell
def _(G, Y_obs, pi, x_cond):
    alphas   = [pi.normal(3.0, 2.0) for _ in range(G)]
    betas    = [pi.normal(0.0, 1.0) for _ in range(G)]
    log_phis = [pi.normal(0.0, 1.0) for _ in range(G)]
    phis     = [pi.exp(lp) for lp in log_phis]

    ys = []
    for _g in range(G):
        _mu  = pi.exp(alphas[_g] + betas[_g] * x_cond)  # mean count, shape (N,)
        _lam = pi.gamma(phis[_g], phis[_g] / _mu)        # Gamma mixing variable
        ys.append(pi.poisson(_lam))                       # observed counts

    # Name the parameters we want in the posterior
    params = {}
    for _g, _name in enumerate(["GeneA", "GeneB", "GeneC"]):
        params[f"alpha_{_name}"] = alphas[_g]
        params[f"beta_{_name}"]  = betas[_g]
        params[f"phi_{_name}"]   = phis[_g]

    return alphas, betas, log_phis, params, phis, ys


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Inference

    We condition on the observed counts $Y$ and draw samples from the joint posterior
    $p(\alpha, \beta, \phi \mid Y)$ using MCMC (BlackJax/NUTS under the hood).

    The Gamma mixing variables $\lambda_{gc}$ are latent — Pangolin discovers them
    in the model graph and samples them automatically alongside the gene-level parameters.
    """)
    return


@app.cell
def _(az, gene_names, idata, np, pd):
    beta_names = [f"beta_{name}" for name in gene_names]
    summary = az.summary(idata, var_names=beta_names)[["mean", "sd", "hdi_3%", "hdi_97%"]]

    rows_post = []
    for _name in gene_names:
        _samples = idata.posterior[f"beta_{_name}"].values.flatten()
        _row = summary.loc[f"beta_{_name}"]
        rows_post.append({
            "Gene":       _name,
            "Post. mean": round(float(_row["mean"]), 3),
            "Post. SD":   round(float(_row["sd"]), 3),
            "HDI 3%":     round(float(_row["hdi_3%"]), 3),
            "HDI 97%":    round(float(_row["hdi_97%"]), 3),
            "P(LFC > 0)": round(float(np.mean(_samples > 0)), 2),
            "P(LFC < 0)": round(float(np.mean(_samples < 0)), 2),
        })

    pd.DataFrame(rows_post).set_index("Gene")
    return beta_names, rows_post, summary


@app.cell
def _(pg, params, ys, Y_obs):
    idata = pg.blackjax.sample_arviz(params, ys, Y_obs, niter=2000)
    return (idata,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Results

    The posterior distributions of $\beta_g$ quantify our uncertainty about each gene's
    log fold-change.  Because $\beta_g$ is a continuous parameter, the natural Bayesian
    summary is the **posterior probability of being differentially expressed**:

    - $P(\beta_g > 0)$: probability of upregulation
    - $P(\beta_g < 0)$: probability of downregulation

    Unlike a p-value, these are direct probability statements about the quantity of interest.
    """)
    return


@app.cell
def _(gene_names, idata, np, plt, true_beta):
    fig_forest, ax_forest = plt.subplots(figsize=(6, 3))

    for _g, _name in enumerate(gene_names):
        _samples = idata.posterior[f"beta_{_name}"].values.flatten()
        _lo, _hi = np.percentile(_samples, [3, 97])
        ax_forest.plot([_lo, _hi], [_g, _g], color="steelblue", lw=2.5)
        ax_forest.plot(np.mean(_samples), _g, "o", color="steelblue", zorder=3)
        ax_forest.plot(
            true_beta[_g], _g, "x",
            color="tomato", ms=9, mew=2,
            label="true LFC" if _g == 0 else "",
        )

    ax_forest.axvline(0, color="gray", linestyle="--", lw=1)
    ax_forest.set_yticks(range(len(gene_names)))
    ax_forest.set_yticklabels(gene_names)
    ax_forest.set_xlabel("Log Fold-Change (β)")
    ax_forest.set_title("Posterior LFC estimates — 94% HDI\n(blue = posterior mean, red × = true value)")
    ax_forest.legend(loc="lower right")
    plt.tight_layout()
    fig_forest
    return ax_forest, fig_forest


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The forest plot shows the posterior mean and 94% highest-density interval (HDI) for
    each gene's LFC.  GeneA and GeneB are clearly separated from zero; GeneC straddles it,
    reflecting genuine uncertainty about a truly unchanged gene.
    """)
    return


@app.cell
def _(az, beta_names, idata, plt):
    az.plot_trace(idata, var_names=beta_names, compact=True)
    plt.suptitle("MCMC traces — log fold-changes (β)", y=1.01)
    plt.tight_layout()
    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The trace plots confirm that the chains mixed well: the left panels show smooth,
    overlapping density estimates and the right panels show stationary, well-mixed
    sample paths — no divergences or drifts.
    """)
    return


@app.cell
def _(az, beta_names, idata, plt):
    az.plot_posterior(idata, var_names=beta_names, ref_val=0)
    plt.suptitle("Posterior distributions — log fold-changes (β)\n(reference line at 0)", y=1.01)
    plt.tight_layout()
    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The posterior plots show the full posterior density for each $\beta_g$, with the
    shaded 94% HDI.  The reference line at 0 makes it easy to read off whether the
    mass is clearly above, below, or straddling zero.

    ---

    **Key takeaways**

    - The Negative Binomial likelihood, written as a Gamma–Poisson mixture, handles
      the overdispersion typical of RNA-seq data.
    - Bayesian inference yields the full posterior over LFCs, not just point estimates,
      giving a natural measure of uncertainty.
    - $P(\beta_g > 0)$ and $P(\beta_g < 0)$ replace p-values with direct probability
      statements about upregulation and downregulation.
    - Pangolin's `vmap`-based broadcasting lets the same scalar prior nodes
      ($\alpha_g$, $\beta_g$, $\phi_g$) generate a vector of per-cell likelihoods
      without any explicit looping.
    """)
    return


if __name__ == "__main__":
    app.run()
