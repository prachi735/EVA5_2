import math


def cyclic_lr(it, stepsize, min_lr=3e-4, max_lr=3e-3):
    cycle = math.floor(1 + it / (2 * stepsize))
    x = abs(it / stepsize - 2 * cycle + 1)
    lr_lambda = max(0, (1-x))
    lr = min_lr + (max_lr - min_lr) * lr_lambda
    return lr
