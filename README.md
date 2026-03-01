# Linear Equation Systems Solvers

A Python-based computational tool designed to analyze and compare different numerical methods for solving systems of linear equations. Developed for a Numerical Methods course, this project evaluates the performance, correctness, and computational complexity of both iterative and direct solving algorithms.



## Overview

The project focuses on solving the equation $Ax=b$, where A is a square banded matrix containing five distinct diagonals. These types of matrices frequently appear in real-world applications such as electronics, electrodynamics, mechanics, and fluid dynamics. 

The application implements three distinct numerical methods:
1. **Jacobi Method** (Iterative)
2. **Gauss-Seidel Method** (Iterative)
3. **LU Decomposition** (Direct factorization)

## Technologies

* **Language:** Python 3
* **Libraries:** `numpy` for highly optimized matrix operations, and `matplotlib.pyplot` for rendering performance graphs.

## Key Features

* **Custom Algorithm Implementations:** Calculates solutions without relying on built-in high-level solver functions (utilizing fundamental matrix operations instead).
* **Convergence Tracking:** Calculates the Euclidean norm of the residual vector ($r^{(k)}=Ax^{(k)}-b$) at each iteration to monitor the precision and define the stopping criterion for iterative methods.
* **Performance Profiling:** Measures execution time using the `timeit` module across increasingly large matrix sizes (from N=100 up to N=3000).
* **Data Visualization:** Generates logarithmic and linear charts to visually compare the execution times and residual convergence of the different algorithms.
* **In-depth Analysis:** **[Read the full analysis report here (MN_PR2_pdf.pdf)](./MN_PR2_pdf.pdf)**

## Project Conclusions

Based on the performance profiling and theoretical analysis:
* **Computational Complexity:** Iterative methods (Jacobi and Gauss-Seidel) demonstrate an $O(N^2)$ complexity, making them significantly faster for large systems. In contrast, the direct LU decomposition method exhibits $O(N^3)$ complexity.
* **Scalability:** For smaller systems ($N<500$), the execution times across all three methods are comparable. However, as the number of unknowns grows, LU decomposition becomes highly inefficient.
* **Iterative Efficiency:** Between the iterative approaches, the Gauss-Seidel method operates roughly twice as fast as the Jacobi method.

## Setup and Execution

1. Ensure you have Python installed along with the required libraries:
   `pip install numpy matplotlib`
2. Clone the repository containing the `main.py` script.
3. Run the analysis script from your terminal:
   `python main.py`
