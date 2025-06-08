"""Definition of constants for odd finite difference (up to 5)."""

import numpy as np

# All constants are for forward finite difference method.
_FIRST_ORDER_COEFFICIENTS = np.array((-1, 1))
_THIRD_ORDER_COEFFICIENTS = np.array((-11 / 6, 3, -3 / 2, 1 / 3))
_FIFTH_ORDER_COEFFICIENTS = np.array((-137 / 60, 5, -5, 10 / 3, -5 / 4, 1 / 5))
