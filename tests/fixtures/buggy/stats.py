"""Small statistics helpers."""


def mean(xs):
    return sum(xs) / len(xs)


def median(xs):
    n = len(xs)
    mid = n // 2
    if n % 2:
        return xs[mid]
    return (xs[mid - 1] + xs[mid]) / 2


def variance(xs):
    m = mean(xs)
    return sum((x - m) ** 2 for x in xs) / len(xs) - 1
