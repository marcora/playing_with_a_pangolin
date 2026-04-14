# /// script
# [tool.marimo.runtime]
# auto_instantiate = false
# ///

import marimo

__generated_with = "0.23.1"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return


@app.cell
def _():
    import numpy as np
    import seaborn as sns
    import pangolin as pg
    from pangolin import interface as pi

    return np, pg, pi, sns


@app.cell
def _(np):
    # synthetic data
    np.random.seed(67)
    z_true = 0.7
    N = 20
    x_obs = np.random.binomial(1, z_true, N)
    z_obs = np.mean(x_obs)
    z_obs
    return N, x_obs


@app.cell
def _(N, pg, pi, x_obs):
    # model spec
    z = pi.beta(2, 2)
    x = [pi.bernoulli(z) for i in range(N)]
    # do inference
    z_post = pg.blackjax.sample(z, x, x_obs.tolist()) # p(z | x = x_obs)
    return (z_post,)


@app.cell
def _(sns, z_post):
    # plot
    ax = sns.kdeplot(z_post)
    ax.set_xlabel('θ')
    ax.set_xlim(0, 1)
    ax
    return


if __name__ == "__main__":
    app.run()
