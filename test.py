import pangolin as pg
from pangolin import interface as pi

x = pi.normal(0, 1)
pi.print_upstream(x)
pg.blackjax.sample(x)
