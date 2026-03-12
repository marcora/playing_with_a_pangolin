import pangolin as pg
from pangolin import interface as pi

# data for 8 schools model
num_schools = 8
observed = [28, 8, -3, 7, -1, 1, 18, 12]
stddevs = [15, 10, 16, 11, 9, 11, 10, 18]

# define model
mu = pi.normal(0, 10)  # μ ~ normal(0,10)
tau = pi.exp(pi.normal(5, 1))  # τ ~ lognormal(5,1)
theta = [pi.normal(mu, tau) for i in range(num_schools)]  # θ[i] ~ normal(μ,τ)
y = [
    pi.normal(theta[i], stddevs[i]) for i in range(num_schools)
]  # y[i] ~ normal(θ[i],stddevs[i])

# do inference / sample from p(theta | y=observed)
theta_samps = pg.blackjax.sample(theta, y, observed, niter=10000)


# plot results (no pangolin here!)
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

sns.swarmplot(np.array(theta_samps)[:, ::50].T, s=2, zorder=0)
plt.xlabel("school")
plt.ylabel("treatment effect")
