import pangolin as pg
from pangolin import interface as pi

alpha = 2
beta = 20
n = 1000
y_obs = 80

π = pi.beta(alpha, beta)
y = pi.binomial(n, π)

samples = pg.blackjax.sample(π, y, y_obs, niter=10000)

print(samples.mean())
