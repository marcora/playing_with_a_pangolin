## python simulation

import random

doors = {1, 2, 3}

win_keep = 0
win_switch = 0

M = 1000

for _ in range(M):
    car = random.choice(list(doors))
    pick = random.choice(list(doors))
    monty = random.choice(list(doors - {car, pick}))
    if car == pick:
        win_keep += 1
    else:
        win_switch += 1

print(f"Keep: {win_keep}, Switch: {win_switch}")

## pangolin implementation

import pangolin as pg
from pangolin import blackjax as bx
from pangolin import interface as pi

car = pi.categorical([1 / 3, 1 / 3, 1 / 3])
pick = pi.categorical([1 / 3, 1 / 3, 1 / 3])
monty = pi.categorical([1 / 3, 1 / 3, 1 / 3])
