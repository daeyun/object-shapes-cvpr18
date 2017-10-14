from timeit import default_timer as timer
import time
from collections import defaultdict
from contextlib import contextmanager

times = defaultdict(float)
counts = defaultdict(int)


class timeit:
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        name = self.name
        times[name] += self.end - self.start
        counts[name] += 1


def print_elapsed():
    for k, seconds in times.items():
        count = counts[k]
        print('{}: {:.6}'.format(k, seconds / count))
