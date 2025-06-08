"""Internal implementation of finite difference methods for softadaptx."""

import numpy as np
from findiff import coefficients

from softadaptx.constants._finite_difference_constants import (
    _FIFTH_ORDER_COEFFICIENTS,
    _FIRST_ORDER_COEFFICIENTS,
    _THIRD_ORDER_COEFFICIENTS,
)
from softadaptx.utilities.logging import get_logger

# Get the logger
logger = get_logger()

# Constants
_MAX_NON_EVEN_ORDER = 5
_MIN_HIGHER_ORDER = 3


def _get_finite_difference(
    input_array: np.array,
    order: int | None = None,
    verbose: bool = True,
) -> float:
    """Internal utility method for estimating rate of change.

    This function aims to approximate the rate of change for a loss function,
    which is used for the 'LossWeighted' and 'Normalized' variants of softadaptx.

    For even accuracy orders, we take advantage of the `findiff` package
    (https://findiff.readthedocs.io/en/latest/source/examples-basic.html).
    Accuracy orders of 1 (trivial), 3, and 5 are retrieved from an internal
    constants file. Due to the underlying mathematics of computing the
    coefficients, all accuracy orders higher than 5 must be an even number.

    Args:
        input_array: An array of floats containing loss evaluations at the
          previous 'n' points (as many points as the order) of the finite
          difference method.
        order: An integer indicating the order of the finite difference method
          we want to use. The function will use the length of the 'input_array'
          array if no values is provided.
        verbose: Whether we want the function to print out information about
          computations or not.

    Returns:
        A float which is the approximated rate of change between the loss
        points.

    Raises:
        ValueError: If the number of points in the `input_array` array is
          smaller than the order of accuracy we desire.
        ValueError: If the order of accuracy is higher than 5 and it is not an
          even number.
    """
    # First, we want to check the order and the number of loss points we are
    # given
    if order is None:
        order = len(input_array) - 1
        if verbose:
            msg = (
                "Interpreting finite difference order as {} since no explicit order was specified."
            )
            msg = msg.format(order)
            logger.info(msg)
    else:
        if order > len(input_array):
            error_msg = (
                "The order of finite difference computations cannot be larger than the "
                "number of loss points. Please check the order argument or wait until "
                "enough points have been stored before calling the method."
            )
            raise ValueError(error_msg)

        if order + 1 < len(input_array):
            if verbose:
                msg = (
                    f"There are more points than 'order' + 1 ({order + 1}) points "
                    f"(array contains {len(input_array)} values). "
                    f"Function will use the last {order + 1} elements of loss points "
                    f"for computations."
                )
                logger.info(msg)
            input_array = input_array[(-1 * order - 1) :]

    order_is_even = order % 2 == 0
    # Next, we want to retrieve the correct coefficients based on the order
    if order > _MAX_NON_EVEN_ORDER and not order_is_even:
        raise ValueError(
            f"Accuracy orders larger than {_MAX_NON_EVEN_ORDER} must be even.",
        )

    if order_is_even:
        constants = coefficients(deriv=1, acc=order)["forward"]["coefficients"]
    elif order == 1:
        constants = _FIRST_ORDER_COEFFICIENTS
    elif order == _MIN_HIGHER_ORDER:
        constants = _THIRD_ORDER_COEFFICIENTS
    else:
        constants = _FIFTH_ORDER_COEFFICIENTS

    pointwise_multiplication = [input_array[i] * constants[i] for i in range(len(constants))]
    return float(np.sum(pointwise_multiplication))
