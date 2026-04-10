# /// script
# [tool.marimo.runtime]
# auto_instantiate = false
# ///

import marimo

__generated_with = "0.23.0"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return


@app.cell
def _():
    # synthetic data
    import numpy as np
    np.random.seed(67)
    z_true = 0.7
    N = 20
    x_obs = np.random.binomial(1, z_true, N)

    # create model
    import pangolin
    from pangolin import interface as pi
    z = pi.beta(2,2)
    x = pi.vfor(pi.bernoulli, None, N)(z)

    # do inference
    z_samps = pangolin.blackjax.sample(z, x, x_obs) # p(z | x = x_obs)

    # plot
    import seaborn as sns
    sns.histplot(z_samps, binrange=[0,1])
    return


if __name__ == "__main__":
    app.run()
