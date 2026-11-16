import numpy as np
import pytest

import qsim

PAULI_X = np.array([[0, 1],
                    [1, 0]], dtype=complex)

PAULI_Y = np.array([[0, -1j],
                    [1j, 0]], dtype=complex)

PAULI_Z = np.array([[1, 0],
                    [0, -1]], dtype=complex)


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


def test_pauli_matrices_are_hermitian_and_unitary():
    assert qsim.is_hermitian(PAULI_X)
    assert qsim.is_hermitian(PAULI_Y)
    assert qsim.is_hermitian(PAULI_Z)

    assert qsim.is_unitary(PAULI_X)
    assert qsim.is_unitary(PAULI_Y)
    assert qsim.is_unitary(PAULI_Z)


def test_identity_is_hermitian_and_unitary():
    A = np.identity(2)

    assert qsim.is_hermitian(A)
    assert qsim.is_unitary(A)


def test_hermitian_but_not_unitary():
    A = np.array([[1, 0],
                  [0, 2]])

    assert qsim.is_hermitian(A)
    assert not qsim.is_unitary(A)


def test_unitary_but_not_hermitian():
    A = np.array([[1, 0],
                  [0, 1j]])

    assert not qsim.is_hermitian(A)
    assert qsim.is_unitary(A)


def test_neither_hermitian_nor_unitary():
    A = np.array([[0, 1],
                  [0, 0]])

    assert not qsim.is_hermitian(A)
    assert not qsim.is_unitary(A)


def test_ketbra_projector_is_hermitian():
    length = 3

    rng = np.random.default_rng(seed=42)
    real_part = (rng.random(length) - 0.5) * 2
    imag_part = (rng.random(length) - 0.5) * 2
    vec_rand = real_part + 1j * imag_part
    A = qsim.ketbra(vec_rand, vec_rand)

    assert qsim.is_hermitian(A)


def test_unitary_larger_matrix():
    A = np.array([[0, 0, 1],
                  [1, 0, 0],
                  [0, 1, 0]])

    assert not qsim.is_hermitian(A)
    assert qsim.is_unitary(A)


def test_small_numerical_error_is_tolerated():
    X_err = np.copy(PAULI_X)
    X_err[0, 1] += 5.0 * 1e-12

    assert qsim.is_hermitian(X_err)
    assert qsim.is_unitary(X_err)


def test_large_error_is_not_tolerated():
    X_err = np.copy(PAULI_X)
    X_err[0, 1] += 5.0 * 1e-3

    assert not qsim.is_hermitian(X_err)
    assert not qsim.is_unitary(X_err)


def test_accepts_python_lists():
    X_list = [[0, 1],
         [1, 0]]

    assert qsim.is_hermitian(X_list)
    assert qsim.is_unitary(X_list)


def test_rejects_non_square_matrix():
    A = np.array([[1, 0, 1],
                  [0, 1, 0]])

    with pytest.raises(ValueError):
        qsim.is_hermitian(A)

    with pytest.raises(ValueError):
        qsim.is_unitary(A)


def test_rejects_1d_array():
    A = np.array([1, 1])

    with pytest.raises(ValueError):
        qsim.is_hermitian(A)

    with pytest.raises(ValueError):
        qsim.is_unitary(A)


def test_rejects_non_numeric():
    A = np.array([["AB", "CD"],
                  ["EF", "GH"]])

    with pytest.raises(TypeError):
        qsim.is_hermitian(A)

    with pytest.raises(TypeError):
        qsim.is_unitary(A)


def test_rejects_empty_matrix():
    A = np.array([])

    with pytest.raises(ValueError):
        qsim.is_hermitian(A)

    with pytest.raises(ValueError):
        qsim.is_unitary(A)
