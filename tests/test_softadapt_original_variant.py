"""Unit testing for the original SoftAdapt variant."""

import unittest

import numpy as np

from softadaptx import SoftAdapt


class TestSoftAdapt(unittest.TestCase):
    """Class for testing our finite difference implementation."""

    @classmethod
    def setUpClass(cls) -> None:
        """Set up class-level test configuration."""
        cls.decimal_place = 5

    # First starting with positive slope test cases.
    def test_beta_positive_three_components(self) -> None:
        """Test SoftAdapt with three positive-slope loss components."""
        loss_component1 = np.array([1, 2, 3, 4, 5])
        loss_component2 = np.array([150, 100, 50, 10, 0.1])
        loss_component3 = np.array([1500, 1000, 500, 100, 1])

        solutions = np.array([9.9343e-01, 6.5666e-03, 3.8908e-22])

        softadapt_object = SoftAdapt(beta=0.1)
        alpha_0, alpha_1, alpha_2 = softadapt_object.get_component_weights(
            tuple(loss_component1.tolist()),
            tuple(loss_component2.tolist()),
            tuple(loss_component3.tolist()),
            verbose=False,
        )
        assert abs(alpha_0 - solutions[0]) < 10**-self.decimal_place, (
            "Incorrect SoftAdapt calculation for simple 'dominant loss' case. "
            "The first loss component failed."
        )
        assert abs(alpha_1 - solutions[1]) < 10**-self.decimal_place, (
            "Incorrect SoftAdapt calculation for simple 'dominant loss' case. "
            "The second loss component failed."
        )
        assert abs(alpha_2 - solutions[2]) < 10**-self.decimal_place, (
            "Incorrect SoftAdapt calculation for simple 'dominant loss' case. "
            "The third loss component failed."
        )
