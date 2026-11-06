"""Basic linear algebra utilities for quantum states and operators.

This module provides functions based on Dirac's bra-ket notation, such as
the inner product of state vectors. Vectors may be given as 1-D arrays,
row vectors of shape (1, n), or column vectors of shape (n, 1).
"""

import numpy as np
import numpy.typing as npt


def braket(a: npt.ArrayLike, b: npt.ArrayLike) -> complex:
    """Return the inner product <a|b> of two state vectors.

    The complex conjugate is taken on the first argument ``a`` (the bra),
    following the convention of quantum mechanics:
    ``<a|b> = sum_i conj(a_i) * b_i``.

    Parameters
    ----------
    a : array_like
        The vector on the bra side. A 1-D array, a row vector of shape
        (1, n), or a column vector of shape (n, 1).
    b : array_like
        The vector on the ket side, in the same format as ``a``.

    Returns
    -------
    complex
        The inner product <a|b>. The result is always a Python ``complex``,
        even when both inputs are real.

    Raises
    ------
    TypeError
        If ``a`` or ``b`` contains non-numeric values.
    ValueError
        If ``a`` or ``b`` is not a vector, is empty, or if the two vectors
        have different lengths.

    Examples
    --------
    >>> braket([1, 0], [0, 1])
    0j
    >>> braket([1j, 0], [1, 0])
    -1j
    >>> braket([1, 1j], [1, 1j])
    (2+0j)
    """
    a = _as_vector(a, "a")
    b = _as_vector(b, "b")

    _check_same_length(a, b)

    return complex(np.inner(np.conj(a), b))


def ketbra(a: npt.ArrayLike, b: npt.ArrayLike) -> np.ndarray:
    """Return the outer product |a><b| of two state vectors.

    The complex conjugate is taken on the second argument ``b`` (the bra),
    following the convention of quantum mechanics:
    ``(|a><b|)_ij = a_i * conj(b_j)``.

    Parameters
    ----------
    a : array_like
        The vector on the ket side, of length m. A 1-D array, a row vector
        of shape (1, m), or a column vector of shape (m, 1).
    b : array_like
        The vector on the bra side, of length n, in the same format as ``a``.

    Returns
    -------
    numpy.ndarray
        The outer product |a><b| as an array of shape (m, n) with dtype
        ``complex128``, even when both inputs are real.

    Raises
    ------
    TypeError
        If ``a`` or ``b`` contains non-numeric values.
    ValueError
        If ``a`` or ``b`` is not a vector or is empty.

    Examples
    --------
    >>> ketbra([1, 0], [0, 1])
    array([[0.+0.j, 1.+0.j],
           [0.+0.j, 0.+0.j]])
    >>> ketbra([1, 0], [1j, 0])
    array([[0.-1.j, 0.+0.j],
           [0.+0.j, 0.+0.j]])
    >>> ketbra([1, 0, 0], [1, 0]).shape
    (3, 2)
    """
    a = _as_vector(a, "a")
    b = _as_vector(b, "b")

    a = a.reshape(-1, 1)
    b = b.reshape(1, -1)

    return (a @ np.conj(b)).astype(np.complex128)


def _as_vector(x: npt.ArrayLike, name: str) -> np.ndarray:
    """Convert x to a 1-D numeric array, accepting (n,), (1, n), and (n, 1)."""
    arr = np.asarray(x)

    if not np.issubdtype(arr.dtype, np.number):
        raise TypeError(f"{name} must contain numbers, got dtype {arr.dtype}")

    if arr.ndim == 0 or arr.ndim > 2:
        raise ValueError(
            f"{name} must be a 1-D, row, or column vector, got shape {arr.shape}"
        )

    if arr.ndim == 2 and min(arr.shape) != 1:
        raise ValueError(
            f"{name} must be a row or column vector, got shape {arr.shape}"
        )

    if arr.size == 0:
        raise ValueError(f"{name} must not be empty")

    return arr.ravel()


def _check_same_length(a: np.ndarray, b: np.ndarray) -> None:
    """Raise ValueError if the 1-D arrays a and b have different lengths."""
    if a.size != b.size:
        raise ValueError(
            f"a and b must have the same length, got {a.size} and {b.size}"
        )
