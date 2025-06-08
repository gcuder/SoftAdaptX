"""Implementation of the base class for SoftAdaptX."""

import numpy as np

from softadaptx.constants._stability_constants import _EPSILON
from softadaptx.utilities._finite_difference import _get_finite_difference


class SoftAdaptBase:
    """Base model for any of the SoftAdaptX variants.

    Attributes:
        epsilon: A float which is added to the denominator of a division for
          numerical stability.

    """

    def __init__(self) -> None:
        """Initializer of the base method."""
        self.epsilon = _EPSILON

    def _softmax(
        self,
        input_tensor: np.ndarray,
        beta: float = 1,
        numerator_weights: np.ndarray = None,
        shift_by_max_value: bool = True,
    ) -> np.ndarray:
        """Implementation of SoftAdaptX's modified softmax function.

        Args:
            input_tensor: A numpy array of floats which will be used for computing
              the (modified) softmax function.
            beta: A float which is the scaling factor (as described in the
              manuscript).
            numerator_weights: A numpy array of weights which are the actual value of
              of the loss components. This option is used for the
              "loss-weighted" variant of SoftAdapt.
            shift_by_max_value: A boolean indicating whether we want the values
              in the input tensor to be shifted by the maximum value.

        Returns:
            A numpy array of floats that are the softmax results.

        Raises:
            None.

        """
        if shift_by_max_value:
            exp_of_input = np.exp(beta * (input_tensor - np.max(input_tensor)))
        else:
            exp_of_input = np.exp(beta * input_tensor)

        # This option will be used for the "loss-weighted" variant of SoftAdapt.
        if numerator_weights is not None:
            exp_of_input = np.multiply(numerator_weights, exp_of_input)

        return exp_of_input / (np.sum(exp_of_input) + self.epsilon)

    def _compute_rates_of_change(
        self,
        input_tensor: np.ndarray,
        order: int | None = 5,
        verbose: bool = True,
    ) -> float:
        """Base class method for computing loss functions rate of change.

        Args:
            input_tensor: A numpy array of floats containing loss evaluations at the
              previous 'n' points (as many points as the order) of the finite
              difference method.
            order: An integer indicating the order of the finite difference
              method we want to use. The function will use the length of the
              'input_array' array if no values is provided.
            verbose: Whether we want the function to print out information about
              computations or not.

        Returns:
            The approximated derivative as a float value.

        Raises:
            None.

        """
        return _get_finite_difference(input_array=input_tensor, order=order, verbose=verbose)
