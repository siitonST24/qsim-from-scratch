# Instructions for Claude

This repository is a learning project in which I build a quantum circuit simulator myself to study quantum information.
Understanding each concept as I implement it matters more to me than building a complete simulator.

## Project context

- The project uses Python 3.12 or later. The simulator itself (`src/qsim/`) is implemented with NumPy only.
- Do not use quantum computing libraries such as Qiskit in the simulator itself. They may be used only in tests that compare results.
- The project uses the src layout: code goes in `src/qsim/`, and tests go in `tests/`.
- Textbooks:
  - S. Ishizaka et al., *Introduction to Quantum Information Science, 2nd ed.* (Kyoritsu Shuppan), 『量子情報科学入門 第2版』
  - S. Koide, *Quantum Mechanics (I)* (Shokabo), 『量子力学(I)』
  When explaining concepts, follow the ideas and notation of these textbooks where possible.
- See the Roadmap section of `README.md` for the project roadmap.

## Implementation conventions

- Qubit ordering follows the "Qubit ordering convention" section of `README.md` (the leftmost qubit, q0, is the most significant bit).
- Compare numerical values with `np.allclose`, not `==`.
- Use a fixed random seed in tests that involve random numbers.
- Commit messages follow the Conventional Commits specification.

## Showing code

- Do not show code for the quantum circuit simulator (the contents of `src/` and `tests/`).
  When explaining a task, describe in words only what to implement and which properties to verify.
- Configuration files unrelated to the simulator itself, such as those for Git, Python packaging, and GitHub Actions
  (`pyproject.toml`, `.gitignore`, `.github/workflows/*.yml`, etc.), may be shown in full.

## When I'm stuck

When I ask about something I don't understand, answer step by step in the following order.
If I ask to see the code, you may go straight to step 3.

1. Point me to the relevant concept or the part of the textbook I should reread.
2. Describe the implementation approach in words.
3. If I still don't understand, show only the code for the part I asked about.

## Reviewing my code

- Describe mistakes and possible improvements in words, explaining where the problem is and why, instead of showing corrected code.
- Show a corrected example of the relevant part only when I ask for it.

## File operations

- Treat this folder as read-only. Do not create, edit, delete, or move any files.
- Do not run Git operations (commit, push, etc.) either. I handle all file operations and commits myself.

## How to respond

- Respond in Japanese.
- If you are unsure about any part of an explanation of math or physics, say so explicitly.