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
    ## Setup environment
    """)
    return


@app.cell
def _():
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import arviz as az
    import pangolin as pg
    from pangolin import interface as pi

    return pg, pi


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Linear model
    """)
    return


@app.cell
def _(pi):
    α_true = pi.constant(10)
    β_true = pi.constant(2)
    σ_true = pi.constant(5)
    x_true = pi.constant(range(0, 101))
    y_true = α_true + β_true * x_true + pi.normal(0, σ_true)
    return (y_true,)


@app.cell
def _(pg, y_true):
    y_obs = pg.blackjax.sample(y_true, niter = 1).flatten().tolist()
    return


@app.cell
def _(pi):
    α = pi.normal(0, 100)
    β = pi.normal(0, 10)
    σ = pi.exponential(10)
    x = pi.constant(range(0, 101))
    y = α + β * x + pi.normal(0, σ)
    return α, β, σ


@app.cell
def _(pg, α, β, σ):
    pg.blackjax.sample([α, β, σ])
    return


if __name__ == "__main__":
    app.run()
