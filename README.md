# Adaptive Filters Benchmark

This repository provides a **computational performance benchmark** of multiple implementations of **adaptive filter algorithms**, focusing on **execution speed**, **numerical equivalence**, and **convergence validation**.

The primary objective is to evaluate how different **NumPy vectorization strategies** impact runtime performance, while ensuring that algorithmic behavior remains consistent with the original loop-based implementations.

## Benchmark Scope

For each algorithm, the benchmark compares:

* Classical loop-based implementation
* NumPy vectorized implementation
* Execution time
* Numerical consistency between implementations
* Convergence behavior

These comparisons allow a clear assessment of the trade-offs between readability, algorithmic fidelity, and computational performance.

## Evaluated Algorithms

The repository includes benchmarks for the following adaptive filters and operating modes:

* **CMA (Constant Modulus Algorithm)**
  Widely used blind equalization algorithm.

* **RDE (Radius-Directed Equalizer)**
  Radius-directed equalizer suitable for multimodulus constellations.

* **NLMS (Normalized Least Mean Squares)**
  Normalized LMS variant with improved numerical stability.

* **Decision-Directed LMS (DD-LMS)**
  LMS operating in decision-directed mode.

* **Data-Aided RDE (DA-RDE)**
  RDE variant assisted by known reference data.

* **RLS (Recursive Least Squares)**
  Adaptive algorithm with fast convergence and higher computational complexity.

* **Decision-Directed RLS (DD-RLS)**
  RLS variant operating in decision-directed mode.

* **Static Mode**
  Non-adaptive operation used as a performance baseline.

## Methodology

For each filter, controlled experiments are conducted to:

* Measure the impact of vectorization on execution time
* Verify numerical equivalence between implementations
* Validate convergence under identical input conditions

The benchmark is designed to be reproducible, extensible, and easily adaptable to new algorithms or optimization strategies.

## Final Goal

This project serves as a reference for researchers, students, and engineers interested in:

* Optimizing adaptive algorithms in Python
* Evaluating computational performance using NumPy
* Ensuring algorithmic fidelity after performance optimizations

Contributions and extensions are welcome.
