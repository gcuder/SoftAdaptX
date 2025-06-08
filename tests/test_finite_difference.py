"""Unit testing for calculating finite difference approximations."""

import unittest

from softadaptx.utilities.finite_difference import get_finite_difference


class TestFiniteDifference(unittest.TestCase):
    """Class for testing our finite difference implementation."""

    @classmethod
    def setUpClass(cls) -> None:
        """Set up class for testing finite difference implementation."""
        cls.decimal_place = 5

    # First starting with positive slope test cases.
    def test_first_order_positive_slope(self) -> None:
        """Test first order finite difference approximation for simple positive slope test case."""
        order = 2
        loss_points = [0, 1, 2, 3, 4, 5]
        approximation = get_finite_difference(loss_points, order)
        self.assertAlmostEqual(
            1.0,
            approximation,
            self.decimal_place,
            "Incorrect first order approximation for simple positive slope test case.",
        )

    def test_second_order_positive_slope(self) -> None:
        """Test second order finite difference approximation for simple positive slope test case."""
        order = 2
        loss_points = [0, 1, 2, 3, 4, 5]
        approximation = get_finite_difference(loss_points, order)
        self.assertAlmostEqual(
            1.0,
            approximation,
            self.decimal_place,
            "Incorrect second order approximation for simple negative slope test case.",
        )

    def test_third_order_positive_slope(self) -> None:
        """Test third order finite difference approximation for simple positive slope test case."""
        order = 3
        loss_points = [0, 2, 4, 6, 8, 10]
        approximation = get_finite_difference(loss_points, order)
        self.assertAlmostEqual(
            2.0,
            approximation,
            self.decimal_place,
            "Incorrect third order approximation for simple positive slope test case.",
        )

    def test_fourth_order_positive_slope(self) -> None:
        """Test fourth order finite difference approximation for simple positive slope test case."""
        order = 4
        loss_points = [0, 2, 4, 6, 8, 10]
        approximation = get_finite_difference(loss_points, order)
        self.assertAlmostEqual(
            2.0,
            approximation,
            self.decimal_place,
            "Incorrect fourth order approximation for simple positive slope test case.",
        )

    def test_fifth_order_positive_slope(self) -> None:
        """Test fifth order finite difference approximation for simple positive slope test case."""
        order = 5
        loss_points = [-5, -4, -3, -2, -1, 0]
        approximation = get_finite_difference(loss_points, order)
        self.assertAlmostEqual(
            1.0,
            approximation,
            self.decimal_place,
            "Incorrect fifth order approximation for simple positive slope test case.",
        )

    def test_tenth_order_positive_slope(self) -> None:
        """Test tenth order finite difference approximation for simple positive slope test case."""
        order = 10
        loss_points = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        approximation = get_finite_difference(loss_points, order)
        self.assertAlmostEqual(
            1.0,
            approximation,
            self.decimal_place,
            "Incorrect 10th order approximation for simple positive slope test case.",
        )

    # From here on we have negative slope test cases.
    def test_first_order_negative_slope(self) -> None:
        """Test first order finite difference approximation for simple negative slope test case."""
        order = 1
        loss_points = [15, 12, 9, 6, 3, 0]
        approximation = get_finite_difference(loss_points, order)
        self.assertAlmostEqual(
            -3.0,
            approximation,
            self.decimal_place,
            "Incorrect first order approximation for simple negative slope test case.",
        )

    def test_second_order_negative_slope(self) -> None:
        """Test second order finite difference approximation for simple negative slope test case."""
        order = 2
        loss_points = [5, 4, 3, 2, 1, 0]
        approximation = get_finite_difference(loss_points, order)
        self.assertAlmostEqual(
            -1.0,
            approximation,
            self.decimal_place,
            "Incorrect second order approximation for simple negative slope test case.",
        )

    def test_third_order_negative_slope(self) -> None:
        """Test third order finite difference approximation for simple negative slope test case."""
        order = 3
        loss_points = [20, 16, 12, 8, 4, 0]
        approximation = get_finite_difference(loss_points, order)
        self.assertAlmostEqual(
            -4.0,
            approximation,
            self.decimal_place,
            "Incorrect third order approximation for simple negative slope test case.",
        )

    def test_fourth_order_negative_slope(self) -> None:
        """Test fourth order finite difference approximation for simple negative slope test case."""
        order = 4
        loss_points = [5, 4, 3, 2, 1, 0]
        approximation = get_finite_difference(loss_points, order)
        self.assertAlmostEqual(
            -1.0,
            approximation,
            self.decimal_place,
            "Incorrect fourth order approximation for simple negative slope test case.",
        )

    def test_fifth_order_negative_slope(self) -> None:
        """Test fifth order finite difference approximation for simple negative slope test case."""
        order = 5
        loss_points = [5, 4, 3, 2, 1, 0]
        approximation = get_finite_difference(loss_points, order)
        self.assertAlmostEqual(
            -1.0,
            approximation,
            self.decimal_place,
            "Incorrect fifth order approximation for simple negative slope test case.",
        )

    def test_tenth_order_negative_slope(self) -> None:
        """Test tenth order finite difference approximation for simple negative slope test case."""
        order = 10
        loss_points = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
        approximation = get_finite_difference(loss_points, order)
        self.assertAlmostEqual(
            -1.0,
            approximation,
            self.decimal_place,
            "Incorrect 10th order approximation for simple negative slope test case.",
        )
