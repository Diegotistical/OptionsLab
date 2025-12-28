import time
from functools import wraps


def timeit(fn):
    """Decorator to measure execution time of a function in milliseconds."""

    @wraps(fn)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = fn(*args, **kwargs)
        dt_ms = (time.perf_counter() - start) * 1000.0
        print(f"[timing] {fn.__name__}: {dt_ms:.2f} ms")
        return result

    return wrapper
