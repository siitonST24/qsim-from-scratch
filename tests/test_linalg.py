import numpy as np

import qsim


def test_braket_orthonormal_basis():
    basis0 = np.array([1, 0])
    basis1 = np.array([0, 1])

    assert np.isclose(qsim.braket(basis0, basis0), 1)
    assert np.isclose(qsim.braket(basis0, basis1), 0)
    assert np.isclose(qsim.braket(basis1, basis0), 0)
    assert np.isclose(qsim.braket(basis1, basis1), 1)


def test_ketbra_basis_matrix():
    basis0 = np.array([1, 0])
    basis1 = np.array([0, 1])

    expected_00 = np.array([[1, 0],
                            [0, 0]])
    expected_01 = np.array([[0, 1],
                            [0, 0]])
    expected_10 = np.array([[0, 0],
                            [1, 0]])
    expected_11 = np.array([[0, 0],
                            [0, 1]])

    result_00 = qsim.ketbra(basis0, basis0)
    result_01 = qsim.ketbra(basis0, basis1)
    result_10 = qsim.ketbra(basis1, basis0)
    result_11 = qsim.ketbra(basis1, basis1)

    np.testing.assert_allclose(result_00, expected_00)
    np.testing.assert_allclose(result_01, expected_01)
    np.testing.assert_allclose(result_10, expected_10)
    np.testing.assert_allclose(result_11, expected_11)
