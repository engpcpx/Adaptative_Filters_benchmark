# NLMS_benchmark

🔬 NLMS Benchmark — Performance and Convergence Analysis

This repository provides a computational performance benchmark of multiple implementations of the Normalized Least Mean Squares (NLMS) algorithm, focusing on execution speed, numerical equivalence, and convergence validation.

The primary objective is to evaluate how different vectorization strategies in NumPy impact runtime performance, while ensuring that algorithmic behavior remains consistent with the original loop-based implementation.

📌 Key Features

✅ Original NLMS implementation (explicit loops)

🚀 Vectorized implementations:

tensordot

einsum

optimized NumPy broadcasting

📊 Performance benchmark (time per iteration)

📉 Convergence validation (error × iteration)

📈 IEEE-style plots with confidence intervals

🔁 Fully reproducible experiments

🧠 Scope and Motivation

While NLMS is conceptually simple, naive implementations can become computational bottlenecks in real-time or high-dimensional systems. This project demonstrates how careful array shaping and broadcasting can yield significant performance gains without altering the mathematical formulation of the algorithm.

⚠️ Important
This repository is primarily a performance benchmark.
Convergence analysis is included only as a numerical validation tool, not as the main optimization target.

🧪 Experimental Setup

All benchmarks were conducted using the following configuration:

Parameter	Value
Number of modes (nModes)	8
Number of taps (nTaps)	64
Step size (μ)	0.7
Iterations	10,000
Data type	Complex-valued
Random seed	Fixed (reproducible)
📂 Repository Structure
nlms-benchmark/
│
├── nlms.py                 # NLMS implementations
├── benchmark.py            # Performance benchmark (timing only)
├── convergence.py          # Convergence validation (error × iteration)
├── plots/
│   ├── benchmark_nlms.png
│   └── convergencia_nlms.png
│
├── requirements.txt
└── README.md

▶️ How to Run
1️⃣ Install dependencies
pip install -r requirements.txt

2️⃣ Run performance benchmark
python benchmark.py


This will generate:

A bar chart showing average execution time per iteration for each NLMS implementation.

3️⃣ Run convergence validation
python convergence.py


This will generate:

A smoothed convergence plot (moving average)

Confidence intervals (±1 standard deviation)

Direct comparison between original and optimized implementations

📊 Example Results
⏱ Performance Benchmark

Optimized broadcasting version achieves >5× speedup

einsum and tensordot offer moderate acceleration

Loop-based implementation is the slowest

📉 Convergence Validation

Optimized version preserves convergence behavior

Error statistics match the original NLMS within numerical tolerance

No divergence or instability observed

📄 Academic Reference

If you use this code or results in academic work, please cite:

Benchmark Computacional de Implementações do Algoritmo NLMS
(Manuscript under preparation)

🔗 Full Paper and Figures

The IEEE-style manuscript associated with this repository includes:

Detailed benchmark methodology

Statistical convergence analysis

Annotated performance figures

Figures used in the paper are available in the plots/ directory.

🔁 Reproducibility

Fixed random seed

Identical input data across implementations

Strict separation between:

Timing measurements

Convergence analysis

📜 License

This project is provided for research and educational purposes.
You are free to use, modify, and redistribute with proper attribution.

✉️ Contact

For questions, feedback, or collaboration:

Author: Paulo Paixão
Email: eng.pcpx@gmail.com

If you want, I can also:

add a CITATION.cff

generate requirements.txt

create a GitHub Actions workflow for reproducibility

adapt the README for a public dataset / benchmark badge

Just say the word.




ChatGPT can make mistakes. Check important info. See Cookie Preferences.
