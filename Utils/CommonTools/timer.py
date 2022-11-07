import time
import numpy as np


def timer(func):
    def run(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        time_cost = time.time() - start_time
        return result, np.float32(time_cost)

    return run


def timer_print(func):
    def run(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        time_cost = time.time() - start_time
        print(str(func) + '    ' + str(np.float32(time_cost)))
        return result

    return run
