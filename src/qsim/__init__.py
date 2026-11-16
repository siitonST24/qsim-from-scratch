"""A quantum circuit simulator built from scratch with NumPy."""

from .linalg import braket, is_hermitian, is_unitary, ketbra

__all__ = ["braket", "is_hermitian", "is_unitary", "ketbra"]

__version__ = "0.0.1"
