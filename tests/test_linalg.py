import itertools

import numpy as np
import pytest

import qsim

PAULI_X = np.array([[0, 1],
                    [1, 0]], dtype=complex)

PAULI_Y = np.array([[0, -1j],
                    [1j, 0]], dtype=complex)

PAULI_Z = np.array([[1, 0],
                    [0, -1]], dtype=complex)


def random_complex_vector(rng: np.random.Generator, length: int) -> np.ndarray:
    realpart = rng.uniform(-1, 1, length)
    imagpart = rng.uniform(-1, 1, length)
    return realpart + 1j * imagpart


def test_braket_orthonormal_basis():
    basis0 = np.array([1, 0])
    basis1 = np.array([0, 1])

    assert np.isclose(qsim.braket(basis0, basis0), 1)
    assert np.isclose(qsim.braket(basis0, basis1), 0)
    assert np.isclose(qsim.braket(basis1, basis0), 0)
    assert np.isclose(qsim.braket(basis1, basis1), 1)


def test_braket_conjugates_first_argument():
    a = np.array([1, 1j, 0, 1+1j])
    b = np.array([1, 2, 3, 4j])
    expected_ans = 5 + 2j

    assert np.isclose(qsim.braket(a, b), expected_ans)


def test_braket_is_conjugate_symmetric():
    a = np.array([1, 1j, 0, 1+1j])
    b = np.array([1, 2, 3, 4j])

    assert np.isclose(qsim.braket(a, b), np.conj(qsim.braket(b, a)))


def test_braket_with_itself_is_real_squared_norm():
    a = np.array([1, 1j, 0, 1+1j])

    sum_abs2 = np.sum(np.abs(a) ** 2)
    result = qsim.braket(a, a)

    assert np.isclose(result.imag, 0)
    assert np.isclose(result, sum_abs2)


def test_braket_accepts_row_and_column_vectors():
    length = 5

    rng = np.random.default_rng(seed=42)
    vec1_rand = random_complex_vector(rng, length)
    vec2_rand = random_complex_vector(rng, length)

    vec1 = {"1d": vec1_rand, "column": vec1_rand.reshape(-1, 1),
            "row": vec1_rand.reshape(1, -1)}

    vec2 = {"1d": vec2_rand, "column": vec2_rand.reshape(-1, 1),
            "row": vec2_rand.reshape(1, -1)}

    expected_ans = np.vdot(vec1_rand, vec2_rand)

    for (key1, key2) in itertools.product(vec1.keys(), vec2.keys()):
        assert np.isclose(expected_ans, qsim.braket(vec1[key1], vec2[key2]))


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


def test_ketbra_accepts_row_and_column_vectors():
    length = 5

    rng = np.random.default_rng(seed=42)
    vec1_rand = random_complex_vector(rng, length)
    vec2_rand = random_complex_vector(rng, length)

    vec1 = {"1d": vec1_rand, "column": vec1_rand.reshape(-1, 1),
            "row": vec1_rand.reshape(1, -1)}

    vec2 = {"1d": vec2_rand, "column": vec2_rand.reshape(-1, 1),
            "row": vec2_rand.reshape(1, -1)}

    expected_ans = np.outer(vec1_rand, np.conj(vec2_rand))

    for (key1, key2) in itertools.product(vec1.keys(), vec2.keys()):
        result = qsim.ketbra(vec1[key1], vec2[key2])
        assert np.allclose(expected_ans, result)
        assert expected_ans.shape == result.shape


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
    vec_rand = random_complex_vector(rng, length)
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


def test_rejects_empty_vector():
    a = np.array([])

    with pytest.raises(ValueError):
        qsim.braket(a, a)

    with pytest.raises(ValueError):
        qsim.ketbra(a, a)


def test_rejects_non_numeric_vector():
    a = np.array(["ab", "cd", "ef", "gh"])

    with pytest.raises(TypeError):
        qsim.braket(a, a)

    with pytest.raises(TypeError):
        qsim.ketbra(a, a)


def test_rejects_wrong_ndim_vector():
    scalar = 10
    a_3d = np.array([[[1, 1], [1, 1]],
                     [[2, 2], [2, 2]]])

    with pytest.raises(ValueError):
        qsim.braket(scalar, scalar)

    with pytest.raises(ValueError):
        qsim.ketbra(scalar, scalar)

    with pytest.raises(ValueError):
        qsim.braket(a_3d, a_3d)

    with pytest.raises(ValueError):
        qsim.ketbra(a_3d, a_3d)


def test_rejects_2d_matrix_as_vector():
    a_2d = np.array([[1, 2],
                     [3, 4]])

    with pytest.raises(ValueError):
        qsim.braket(a_2d, a_2d)

    with pytest.raises(ValueError):
        qsim.ketbra(a_2d, a_2d)


def test_rejects_length_mismatch():
    a = np.array([1, 2, 3, 4])
    b = np.array([1, 2])

    with pytest.raises(ValueError):
        qsim.braket(a, b)


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
