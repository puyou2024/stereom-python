import numpy as np
import pytest

from stereom.orientation import branch_orientation

AXES = ("L", "C", "R")


def test_pure_L_direction_theta_zero():
    theta, omega = branch_orientation((0, 0, 0), (100, 0, 0), AXES)
    assert np.isclose(theta, 0.0, atol=1e-6)


def test_pure_C_direction_theta_ninety_fixes_D8():
    # dl == 0: the original returned phi=0 here (bug D8). Correct answer is 90.
    theta, omega = branch_orientation((0, 0, 0), (0, 50, 0), AXES)
    assert np.isclose(theta, 90.0, atol=1e-6)
    assert np.isclose(omega, 90.0, atol=1e-6)


def test_pure_R_direction_theta_ninety_and_omega_zero():
    theta, omega = branch_orientation((0, 0, 0), (0, 0, 50), AXES)
    assert np.isclose(theta, 90.0, atol=1e-6)
    assert np.isclose(omega, 0.0, atol=1e-6)


def test_dr_zero_dc_nonzero_omega_fixes_D8():
    # dl != 0, dr == 0, dc != 0: the original returned theta(=omega)=0 here (bug D8).
    theta, omega = branch_orientation((0, 0, 0), (10, 20, 0), AXES)
    assert np.isclose(omega, 90.0, atol=1e-6)


def test_theta_known_triangle():
    theta, omega = branch_orientation((0, 0, 0), (3, 4, 0), AXES)
    assert np.isclose(theta, np.degrees(np.arctan2(4.0, 3.0)), atol=1e-6)


def test_theta_is_direction_independent():
    theta_ab, _ = branch_orientation((0, 0, 0), (10, 20, 30), AXES)
    theta_ba, _ = branch_orientation((10, 20, 30), (0, 0, 0), AXES)
    assert np.isclose(theta_ab, theta_ba, atol=1e-9)


def test_omega_is_direction_independent_when_dl_nonzero():
    _, omega_ab = branch_orientation((0, 0, 0), (10, 20, 30), AXES)
    _, omega_ba = branch_orientation((10, 20, 30), (0, 0, 0), AXES)
    assert np.isclose(omega_ab, omega_ba, atol=1e-9)


def test_theta_in_valid_range_for_random_vectors():
    rng = np.random.default_rng(1)
    for _ in range(200):
        a = rng.integers(-50, 50, size=3)
        b = rng.integers(-50, 50, size=3)
        if np.array_equal(a, b):
            continue
        theta, omega = branch_orientation(tuple(a), tuple(b), AXES)
        assert -1e-9 <= theta <= 90 + 1e-9
        assert -1e-9 <= omega < 360 + 1e-9
