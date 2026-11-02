# qsim-from-scratch

A quantum circuit simulator built from scratch with NumPy, written while learning quantum information.

## Overview

This repository is my learning project for quantum information science.
Instead of using existing quantum computing libraries such as Qiskit, I implement everything myself with NumPy: quantum states, gates, measurement, circuits, algorithms, and noise.
The goal is not to build a fast or production-ready simulator, but to understand each concept by turning it into working, tested code.

Main textbook: S. Ishizaka et al., *Introduction to Quantum Information Science, 2nd ed.* (Kyoritsu Shuppan, 2024, in Japanese)

## Roadmap

- [ ] **Stage 0: Setup** — project structure, tests, CI, and basic linear algebra tools
- [ ] **Stage 1: Postulates of quantum mechanics** — states, measurement, time evolution, and composite systems
- [ ] **Stage 2: Mixed states** — density matrices, partial trace, and generalized measurement (POVM)
- [ ] **Stage 3: Quantum circuits** — circuit model, multi-qubit gates, teleportation, and superdense coding
- [ ] **Stage 4: Quantum algorithms** — Deutsch–Jozsa, Bernstein–Vazirani, Simon, Grover, QFT, phase estimation, and Shor
- [ ] **Stage 5: Quantum information theory** — noise channels, entropy, error correction, and BB84

## Qubit ordering convention

An n-qubit basis state is written as |q0 q1 ... q(n-1)⟩, where **the leftmost qubit q0 is the most significant bit**.
The basis state corresponds to the following index of the state vector:

```
index = q0 * 2^(n-1) + q1 * 2^(n-2) + ... + q(n-1) * 2^0
```

For example, with two qubits, |01⟩ is index 1 and |10⟩ is index 2.

> **Note:** Qiskit uses the opposite (little-endian) ordering, where the rightmost qubit is q0.
> When comparing results with Qiskit, the qubit order must be reversed.

## Installation

*Coming soon.*

## Usage

*Coming soon.* A short example (e.g. creating a Bell state) will be added once the `Circuit` class is implemented.

## Running tests

*Coming soon.*

## Learning log

| Date | What I did |
| --- | --- |
| 2026-11-02 | Created the repository and defined the qubit ordering convention |

## References

- S. Ishizaka, T. Ogawa, A. Kawachi, G. Kimura, M. Hayashi, *Introduction to Quantum Information Science, 2nd ed.*, Kyoritsu Shuppan (2024)

## License

This project is licensed under the MIT License.