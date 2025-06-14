"""Implementation of the original variant of SoftAdaptX."""

import numpy as np

from softadaptx.algorithms.base import SoftAdaptBase
from softadaptx.utilities.logging import get_logger

# Get the logger
logger = get_logger()


class SoftAdapt(SoftAdaptBase):
    """The original variant class.

    The original variant of SoftAdapt is described in section 3.1.1 of our
    manuscript (located at: https://arxiv.org/pdf/1912.12355.pdf).

    Attributes:
        beta: A float that is the 'beta' hyperparameter in our manuscript. If
          beta > 0, then softAdapt will pay more attention the worst performing
          loss component. If beta < 0, then SoftAdapt will assign higher weights
          to the better performing components. Beta==0 is the trivial case and
          all loss components will have coefficient 1.

        accuracy_order: An integer indicating the accuracy order of the finite
          volume approximation of each loss component's slope.

    """

    def __init__(
        self,
        epsilon: float | None = None,
        beta: float = 0.1,
        accuracy_order: int | None = None,
    ) -> None:
        """SoftAdaptX class initializer."""
        super().__init__(epsilon=epsilon)
        self.beta = beta
        # Passing "None" as the order of accuracy sets the highest possible
        # accuracy in the finite difference approximation.
        self.accuracy_order = accuracy_order

    def get_component_weights(
        self,
        *loss_component_values: tuple[np.ndarray | list],
        verbose: bool = True,
    ) -> np.ndarray:
        """Class method for SoftAdaptX weights.

        Args:
            loss_component_values: A tuple consisting of the values of the each
              loss component that have been stored for the past 'n' iterations
              or epochs (as described in the manuscript).
            verbose: A boolean indicating user preference for whether internal
              functions should print out information and warning about
              computations.

        Returns:
            The computed weights for each loss components. For example, if there
            were 5 loss components, say (l_1, l_2, l_3, l_4, l_5), then the
            return numpy array will be the weights (alpha_1, alpha_2, alpha_3,
            alpha_4, alpha_5) in the order of the loss components.

        Raises:
            None.

        """
        if len(loss_component_values) == 1 and verbose:
            logger.warning(
                "You have only passed on the values of one loss component, "
                "which will result in trivial weighting.",
            )

        rates_of_change = []

        for points in loss_component_values:
            # Convert to numpy array if not already
            if not isinstance(points, np.ndarray):
                points_array = np.array(points, dtype=float)
            elif points.dtype != float:
                points_array = points.astype(float)
            else:
                points_array = points

            # Compute the rates of change for each one of the loss components.
            rates_of_change.append(
                self._compute_rates_of_change(points_array, self.accuracy_order, verbose=verbose),
            )

        rates_of_change = np.array(rates_of_change)
        # Calculate the weight and return the values.
        return self._softmax(input_tensor=rates_of_change, beta=self.beta)
