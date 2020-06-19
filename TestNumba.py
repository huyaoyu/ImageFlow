
import numba
import numpy as np
import sys

@numba.jit(nopython=True)
def test_zeros(H, W):
    # array = np.zeros((H, W, 3), dtype=np.int) # Will cause jit error.
    array = np.zeros((H, W, 3), dtype=np.int64) # OK.
    array = np.zeros((H, W, 3), dtype=np.int32) # OK.

def main():
    test_zeros(100, 200)
    return 0

if __name__ == "__main__":
    sys.exit(main())