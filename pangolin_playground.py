# /// script
# dependencies = ["jax", "jaxlib", "pangolin @ git+https://github.com/justindomke/pangolin.git"]
# ///

import marimo

__generated_with = "0.23.1"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Setup environment
    """)
    return


@app.cell
def _():
    import numpy as np
    import pandas as pd
    import pangolin as pg
    import pangolin.interface as pi
    import matplotlib.pyplot as plt
    import seaborn as sns
    import arviz as az

    return az, pg, pi, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Simulate data
    """)
    return


@app.cell
def _():
    N = 100
    return (N,)


@app.cell
def _(N, pi):
    α = pi.constant(10)
    β = pi.constant(2)
    σ = pi.constant(5)
    x = [pi.constant(i) for i in range(N)]
    μ = [α + β * x[i] for i in range(N)]
    y = [pi.normal(μ[i], σ) for i in range(N)]
    return (y,)


@app.cell
def _(pg, y):
    y_sim = pg.blackjax.sample(y)
    return (y_sim,)


@app.cell
def _(N, az, plt, y_sim):
    y_sim_az = az.from_dict(dict(zip(range(N), y_sim)))
    az.plot_trace(y_sim_az)
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Prior
    """)
    return


@app.cell
def _(N, pi):
    α_1 = pi.normal(0, 100)
    β_1 = pi.normal(0, 10)
    σ_1 = pi.exponential(0.1)
    x_1 = [pi.constant(i) for i in range(N)]
    μ_1 = [α_1 + β_1 * x_1[i] for i in range(N)]
    y_1 = [pi.normal(μ_1[i], σ_1) for i in range(N)]
    y_pred = [pi.normal(μ_1[i], σ_1) for i in range(N)]
    return y_1, y_pred, α_1, β_1, σ_1


@app.cell
def _(pg, α_1, β_1, σ_1):
    prior = pg.blackjax.sample([α_1, β_1, σ_1])
    return (prior,)


@app.cell
def _(az, plt, prior):
    prior_az = az.from_dict(dict(zip(["alpha", "beta", "sigma"], prior)))
    az.plot_trace(prior_az)
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Prior predictive
    """)
    return


@app.cell
def _(pg, y_pred):
    y_prior_pred = pg.blackjax.sample(y_pred)
    return (y_prior_pred,)


@app.cell
def _(N, az, plt, y_prior_pred):
    y_prior_pred_az = az.from_dict(dict(zip(range(N), y_prior_pred)))
    az.plot_trace(y_prior_pred_az)
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Posterior
    """)
    return


@app.cell
def _(N, y_sim):
    y_obs = [y_sim[i][0] for i in range(N)]
    return (y_obs,)


@app.cell
def _(pg, y_1, y_obs, α_1, β_1, σ_1):
    post = pg.blackjax.sample([α_1, β_1, σ_1], y_1, y_obs)
    return (post,)


@app.cell
def _(az, plt, post):
    post_az = az.from_dict(dict(zip(["alpha", "beta", "sigma"], post)))
    az.plot_trace(post_az)
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Posterior predictive
    """)
    return


@app.cell
def _(pg, y_1, y_obs, y_pred):
    y_post_pred = pg.blackjax.sample(y_pred, y_1, y_obs)
    return (y_post_pred,)


@app.cell
def _(N, az, plt, y_post_pred):
    y_post_pred_az = az.from_dict(dict(zip(range(N), y_post_pred)))
    az.plot_trace(y_post_pred_az)
    plt.tight_layout()
    plt.show()
    return


if __name__ == "__main__":
    app.run()
