try:
    from numba import jit
except ImportError:
    def jit(*args, **kwargs):
        return lambda f: f


@jit(nopython=True)
def f():
    ...