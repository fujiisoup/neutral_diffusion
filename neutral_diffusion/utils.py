import numpy as np
import sparse


def vec2coo(x):
    if isinstance(x, sparse.COO):
        return x
    # convert dense vector x to sparse.COO
    return sparse.COO([np.arange(len(x)), ], x, shape=(len(x), ))
