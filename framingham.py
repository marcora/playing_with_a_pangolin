# =====================================
# Imports
# =====================================
import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pangolin as pg
from pangolin import interface as pi

# =====================================
# 1. Load Framingham dataset (MIT OCW)
# =====================================
url = "https://ocw.mit.edu/courses/15-071-the-analytics-edge-spring-2017/5d689a024551e672313f7fd7eb1bee8d_framingham.csv"
df = pd.read_csv(url)

# =====================================
# 2. Preprocess data
# =====================================
df = df.rename(columns={"TenYearCHD": "out_bin", "male": "sex_bin", "age": "age"})

df = df[["out_bin", "sex_bin", "age"]].dropna()

# Standardize age
age_mean = df["age"].mean()
age_sd = df["age"].std()
df["age_std"] = (df["age"] - age_mean) / age_sd

# Extract arrays
y_obs = df["out_bin"].astype(int).values
age_std = df["age_std"].values
sex = df["sex_bin"].astype(int).values

n = len(y_obs)

print(f"n = {n}")
print(f"Observed prevalence = {y_obs.mean():.3f}")

# =====================================
# 3. Pangolin model
# =====================================
beta0 = pi.normal(0.0, 1.0)
beta1 = pi.normal(0.0, 1.0)
beta2 = pi.normal(0.0, 1.0)

eta = beta0 + beta1 * age_std + beta2 * sex

prob = pi.sigmoid(eta)

y = pi.bernoulli(prob)

# =====================================
# 4. Inference
# =====================================
post = pg.blackjax.sample([beta0, beta1, beta2], y, y_obs, niter=3000)

b0, b1, b2 = post

# =====================================
# 5. ArviZ conversion
# =====================================
idata = az.from_dict(posterior={"b0": b0, "b1": b1, "b2": b2})

print("\nArviZ summary:")
print(az.summary(idata))

az.plot_trace(idata)
plt.show()

az.plot_posterior(idata)
plt.show()


# =====================================
# 6. Derived quantities
# =====================================
def ilogit(x):
    return 1 / (1 + np.exp(-x))


or_age = np.exp(b1)
or_male = np.exp(b2)

print("\nPosterior probabilities:")
print("P(age increases risk) =", np.mean(or_age > 1))
print("P(male higher risk) =", np.mean(or_male > 1))
