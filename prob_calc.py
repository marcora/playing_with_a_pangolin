from pangolin import interface as pi
from pangolin.blackjax import E

x = pi.normal(0,2) # x ~ normal(0,2)
y = pi.normal(x,6) # y ~ normal(x,6)
print(E(x,y,-2.0)) # E[x|y=-2] (close to -0.2)
