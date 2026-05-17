import numpy as np
import matplotlib.pyplot as plt

import pangolin as pg
import pangolin.interface as pi


# --- 1) Simulate data from a GP prior ---
rng = np.random.default_rng(7)

n = 25
x_obs = np.linspace(0.0, 4.0, n)

amp_true = 1.2
ell_true = 1.0
sigma_true = 0.15  # fixed observation noise for this demo


def rbf_np(x1: np.ndarray, x2: np.ndarray, amp: float, ell: float) -> np.ndarray:
    sqdist = ((x1[:, None] - x2[None, :]) / ell) ** 2
    return (amp**2) * np.exp(-0.5 * sqdist)


K_true = rbf_np(x_obs, x_obs, amp_true, ell_true)
f_true = rng.multivariate_normal(np.zeros(n), K_true + 1e-6 * np.eye(n))
y_obs = f_true + rng.normal(0.0, sigma_true, size=n)


# --- 2) Build GP regression model in Pangolin ---
x = pi.constant(x_obs)

# Positive priors via log-parameterization
amp = pi.exp(pi.normal(0.0, 1.0))
ell = pi.exp(pi.normal(0.0, 0.5))


def kernel_row(x_i, x_all, a, l):
    sqdist = ((x_all - x_i) / l) ** 2
    return (a**2) * pi.exp(-0.5 * sqdist)


# K[i, j] = k(x_i, x_j)
K = pi.vmap(kernel_row, in_axes=[0, None, None, None])(x, x, amp, ell)
K = K + 1e-6 * pi.constant(np.eye(n))  # numerical jitter

f = pi.multi_normal(pi.constant(np.zeros(n)), K)
y = pi.normal(f, sigma_true)


# --- 3) Posterior sampling ---
# sample([latent vars], observed node, observed value)
amp_s, ell_s, f_s = pg.blackjax.sample([amp, ell, f], y, y_obs, niter=1200)

amp_s = np.asarray(amp_s)
ell_s = np.asarray(ell_s)
f_s = np.asarray(f_s)

print("Posterior means")
print(f"  amp ≈ {amp_s.mean():.3f} (true {amp_true})")
print(f"  ell ≈ {ell_s.mean():.3f} (true {ell_true})")


# --- 4) Plot posterior latent function ---
f_mean = f_s.mean(axis=0)
f_lo, f_hi = np.quantile(f_s, [0.05, 0.95], axis=0)

plt.figure(figsize=(9, 5))
plt.scatter(x_obs, y_obs, s=25, alpha=0.7, label="observed y")
plt.plot(x_obs, f_true, "k--", lw=1.5, label="true latent f")
plt.plot(x_obs, f_mean, color="C1", lw=2, label="posterior mean f")
plt.fill_between(x_obs, f_lo, f_hi, color="C1", alpha=0.25, label="90% CI")
plt.title("Gaussian Process regression in Pangolin")
plt.xlabel("x")
plt.ylabel("y / f(x)")
plt.legend()
plt.tight_layout()
plt.show()
