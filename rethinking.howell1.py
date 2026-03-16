from uuid import NAMESPACE_DNS

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pangolin as pg
from pangolin import interface as pi

M = 10000

# -----------------------------
# 1. Create toy dataset
# -----------------------------

np.random.seed(123)

N = 50

# centered height (predictor)
height_c = np.random.normal(0, 1, N)

# true parameters (for simulation)
alpha_true = 45
beta_true = 0.6
sigma_true = 4

# generate weights
weight = alpha_true + beta_true * height_c + np.random.normal(0, sigma_true, N)

# -----------------------------
# 2. Define model
# -----------------------------

alpha = pi.normal(50, 25)
beta = pi.uniform(0, 1.5)
sigma = pi.exponential(0.1)

mu = alpha + beta * height_c
y = pi.normal(mu, sigma)

# -----------------------------
# 3. Run inference
# -----------------------------

draws = pg.blackjax.sample([alpha, beta, sigma], y, weight, niter=M)

# draws is a tuple/list aligned with the parameters
alpha_draws, beta_draws, sigma_draws = draws

print("Posterior means:")
print("alpha:", np.mean(alpha_draws))
print("beta :", np.mean(beta_draws))
print("sigma:", np.mean(sigma_draws))

# -----------------------------
# 3b. Read, prep, and analyze Howell1 (real data)
# -----------------------------

# read Howell1 data
url = "https://raw.githubusercontent.com/rmcelreath/rethinking/master/data/Howell1.csv"
d = pd.read_csv(url, sep=";")

# keep adults only
d = d[d["age"] >= 18].copy()

# create centered height
d["height_c"] = d["height"] - d["height"].mean()

# extract vectors
height_c = d["height_c"].to_numpy()
weight = d["weight"].to_numpy()

print("N adults:", len(d))
print(d[["height", "height_c", "weight"]].head())

# rebuild the likelihood using the real data vectors
mu = alpha + beta * height_c
y = pi.normal(mu, sigma)

# run inference using the already-defined priors/model pieces
draws = pg.blackjax.sample([alpha, beta, sigma], y, weight, niter=M)

alpha_draws, beta_draws, sigma_draws = draws
print("Posterior means:")
print("alpha:", np.mean(alpha_draws))
print("beta :", np.mean(beta_draws))
print("sigma:", np.mean(sigma_draws))

names = ["alpha", "beta", "sigma"]
idata = {
    name: np.array(samples).reshape(4, -1)
    for name, samples in dict(zip(names, draws)).items()
}

print(az.summary(idata))
az.plot_trace(idata)
plt.show()
