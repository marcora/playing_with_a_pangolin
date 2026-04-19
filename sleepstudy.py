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
    # sim obs
    np.random.seed(67)
    z_true = 0.7
    N = 20
    x_obs = np.random.binomial(1, z_true, N)
    z_obs = np.mean(x_obs)
    z_obs
    return N, x_obs


@app.cell
def _(N, pi):
    # spec model
    z = pi.beta(2, 2)
    x = pi.vmap(pi.bernoulli, None, N)(z)
    return x, z


@app.cell
def _(pg, z):
    # sample prior
    z_prior = pg.blackjax.sample(z) # p(z)
    return (z_prior,)


@app.cell
def _(sns, z_prior):
    # plot prior
    ax_prior = sns.kdeplot(z_prior)
    ax_prior.set_xlabel('θ')
    ax_prior.set_xlim(0, 1)
    ax_prior
    return


@app.cell
def _(pg, x, x_obs, z):
    # sample posterior
    z_post = pg.blackjax.sample(z, x, x_obs) # p(z | x = x_obs)
    return (z_post,)


@app.cell
def _(sns, z_post):
    # plot posterior
    ax_post = sns.kdeplot(z_post)
    ax_post.set_xlabel('θ')
    ax_post.set_xlim(0, 1)
    ax_post
    return


if __name__ == "__main__":
    app.run()
