import pangolin as pg
from pangolin import interface as pi

alpha = 2
beta = 20
n = 1000
y_obs = [80, 60, 90, 70, 50]

fn = 0.10
fp = 0.05

π = [pi.beta(alpha, beta) for i in range(len(y_obs))]
π̃ = [fp + π[i] * (1 - fn - fp) for i in range(len(y_obs))]
y = [pi.binomial(n, π̃[i]) for i in range(len(y_obs))]

samples = pg.blackjax.sample(π, y, y_obs, niter=10000)
