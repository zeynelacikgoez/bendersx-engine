import builtins
import math
import random as _random


def array(obj):
    if isinstance(obj, list):
        return obj
    return list(obj)


def ones(shape):
    if isinstance(shape, int):
        return [1.0 for _ in range(shape)]
    rows, cols = shape
    return [[1.0 for _ in range(cols)] for _ in range(rows)]


def zeros(shape):
    if isinstance(shape, int):
        return [0.0 for _ in range(shape)]
    rows, cols = shape
    return [[0.0 for _ in range(cols)] for _ in range(rows)]


def full(shape, value):
    if isinstance(shape, int):
        return [value for _ in range(shape)]
    rows, cols = shape
    return [[value for _ in range(cols)] for _ in range(rows)]


def any(iterable):
    return builtins.any(iterable)


def isnan(x):
    return math.isnan(x)


class random:
    @staticmethod
    def random(size=None):
        if size is None:
            return _random.random()
        if isinstance(size, int):
            return [_random.random() for _ in range(size)]
        rows, cols = size
        return [[_random.random() for _ in range(cols)] for _ in range(rows)]
