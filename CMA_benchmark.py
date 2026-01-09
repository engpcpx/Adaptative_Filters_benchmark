"""
CMA — Benchmark e Convergência
- Gráfico 1: Benchmark Temporal (µs/it)
- Gráfico 2: Convergência (Erro × Iteração)
"""

import numpy as np
import time
import matplotlib.pyplot as plt

# ----------------------------
# Configurações
# ----------------------------
N_ITER = 10_000
N_MODES = 8
N_TAPS = 64
MU = 1e-4
RUN_WL = True
SEED = 42

R_CMA = np.ones((1, N_MODES))


# -----------------------------------------------------
# CMA original (loop) — PRESERVADA
# -----------------------------------------------------
def cmaUp(x, R, outEq, mu, H, H_, nModes, runWL):
    indMode = np.arange(0, nModes)
    outEq = outEq.T
    err = R - np.abs(outEq) ** 2

    prodErrOut = np.diag(err[0]) @ np.diag(outEq[0])

    for N in range(nModes):
        indUpdTaps = indMode + N * nModes
        inAdapt = x[:, N].T
        inAdaptPar = inAdapt.repeat(nModes).reshape(len(x), -1).T

        H[indUpdTaps, :] += mu * prodErrOut @ np.conj(inAdaptPar)

        if runWL:
            H_[indUpdTaps, :] += mu * prodErrOut @ inAdaptPar

    return H, H_, np.abs(err) ** 2


# -----------------------------------------------------
# Vetorizado com tensordot
# -----------------------------------------------------
def cma_vectorized_tensordot(x, R, outEq, mu, H, H_, nModes, runWL):
    outEq = outEq.T
    err = R - np.abs(outEq) ** 2
    prodErrOut = np.diag(err[0]) @ np.diag(outEq[0])

    inA = x.T
    inAdaptPar = np.repeat(inA[:, np.newaxis, :], nModes, axis=1)

    delta = mu * np.transpose(
        np.tensordot(prodErrOut, np.conj(inAdaptPar), axes=([1], [1])),
        (1, 0, 2),
    )

    H += delta.reshape(nModes * nModes, x.shape[0])

    if runWL:
        delta_wl = mu * np.transpose(
            np.tensordot(prodErrOut, inAdaptPar, axes=([1], [1])),
            (1, 0, 2),
        )
        H_ += delta_wl.reshape(nModes * nModes, x.shape[0])

    return H, H_, np.abs(err) ** 2


# -----------------------------------------------------
# Vetorizado com einsum
# -----------------------------------------------------
def cma_vectorized_einsum(x, R, outEq, mu, H, H_, nModes, runWL):
    outEq = outEq.T
    err = R - np.abs(outEq) ** 2
    prodErrOut = np.diag(err[0]) @ np.diag(outEq[0])

    inA = x.T
    inAdaptPar = np.repeat(inA[:, np.newaxis, :], nModes, axis=1)

    delta = mu * np.einsum("ij,njp->nip", prodErrOut, np.conj(inAdaptPar))
    H += delta.reshape(nModes * nModes, x.shape[0])

    if runWL:
        delta_wl = mu * np.einsum("ij,njp->nip", prodErrOut, inAdaptPar)
        H_ += delta_wl.reshape(nModes * nModes, x.shape[0])

    return H, H_, np.abs(err) ** 2


# -----------------------------------------------------
# Vetorizado com broadcast
# -----------------------------------------------------
def cma_vectorized_broadcast(x, R, outEq, mu, H, H_, nModes, runWL):
    outEq = outEq.T
    err = R - np.abs(outEq) ** 2

    delta = (
        mu
        * err[0][None, :, None]
        * outEq[0][None, :, None]
        * np.conj(x.T[:, None, :])
    )

    H += delta.reshape(nModes * nModes, x.shape[0])

    if runWL:
        delta_wl = (
            mu
            * err[0][None, :, None]
            * outEq[0][None, :, None]
            * x.T[:, None, :]
        )
        H_ += delta_wl.reshape(nModes * nModes, x.shape[0])

    return H, H_, np.abs(err) ** 2


# -----------------------------------------------------
# Benchmark Temporal - (Gráfico 1)
# -----------------------------------------------------
def benchmark_cma_time(nIter, nModes, nTaps, mu, runWL, seed):
    rng = np.random.default_rng(seed)

    methods = [
        ("CMA Original (loop)", cmaUp),
        ("CMA Vectorized (tensordot)", cma_vectorized_tensordot),
        ("CMA Vectorized (einsum)", cma_vectorized_einsum),
        ("CMA Vectorized (broadcast)", cma_vectorized_broadcast),
    ]

    xs = rng.standard_normal((nIter, nTaps, nModes)) + 1j * rng.standard_normal((nIter, nTaps, nModes))
    outs = rng.standard_normal((nIter, nModes, 1)) + 1j * rng.standard_normal((nIter, nModes, 1))

    results = {}

    for name, func in methods:
        H = np.zeros((nModes * nModes, nTaps), dtype=np.complex128)
        H_ = np.zeros_like(H)

        t0 = time.perf_counter()
        for k in range(nIter):
            H, H_, _ = func(xs[k], R_CMA, outs[k], mu, H, H_, nModes, runWL)
        t = time.perf_counter() - t0

        results[name] = t / nIter
        print(f"{name:30s}: {results[name]*1e6:8.2f} µs/it")

    plt.figure(figsize=(10, 6))
    bars = plt.bar(results.keys(), [v * 1e6 for v in results.values()])
    plt.ylabel("Tempo médio por iteração (µs)")
    plt.title("Benchmark Temporal CMA — Tempo por Iteração")
    plt.grid(axis="y", alpha=0.3)

    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, h,
                 f"{h:.2f} µs", ha="center", va="bottom", fontweight="bold")

    plt.tight_layout()
    plt.show(block=False)


# -----------------------------------------------------
# Convergência CMA (Gráfico 2) — METODOLOGIA NLMS
# -----------------------------------------------------
def plot_cma_convergence(nIter, nModes, nTaps, mu, runWL, seed):
    rng = np.random.default_rng(seed)

    methods = [
        ("CMA Original (loop)", cmaUp),
        ("CMA Vectorized (broadcast)", cma_vectorized_broadcast),
    ]

    WINDOW = 200

    def moving_stats(x, w):
        mean = np.convolve(x, np.ones(w)/w, mode="valid")
        sq_mean = np.convolve(x**2, np.ones(w)/w, mode="valid")
        std = np.sqrt(np.maximum(sq_mean - mean**2, 0))
        return mean, std

    plt.figure(figsize=(10, 6))

    for label, func in methods:
        H = np.zeros((nModes * nModes, nTaps), dtype=np.complex128)
        H_ = np.zeros_like(H)
        err_trace = np.zeros(nIter)

        for k in range(nIter):
            x = rng.standard_normal((nTaps, nModes)) + 1j * rng.standard_normal((nTaps, nModes))
            outEq = rng.standard_normal((nModes, 1)) + 1j * rng.standard_normal((nModes, 1))

            H, H_, err = func(x, R_CMA, outEq, mu, H, H_, nModes, runWL)
            err_trace[k] = np.sum(err)

        err_mean, err_std = moving_stats(err_trace, WINDOW)

        err_db = 10 * np.log10(err_mean + 1e-12)
        err_db_up = 10 * np.log10(err_mean + err_std + 1e-12)
        err_db_dn = 10 * np.log10(np.maximum(err_mean - err_std, 1e-12))

        it_axis = np.arange(len(err_db))

        plt.plot(it_axis, err_db, label=label, linewidth=2)
        plt.fill_between(it_axis, err_db_dn, err_db_up, alpha=0.2)

    plt.xlabel("Iteração", fontsize=12)
    plt.ylabel("Erro médio CMA |e|² (dB)", fontsize=12)
    plt.title("Convergência do CMA — Validação Numérica", fontsize=13)
    plt.legend()
    plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
    plt.tight_layout()
    plt.show()


# -----------------------------------------------------
# MAIN
# -----------------------------------------------------
if __name__ == "__main__":
    print("CMA — Benchmark + Convergência\n")

    benchmark_cma_time(
        nIter=N_ITER,
        nModes=N_MODES,
        nTaps=N_TAPS,
        mu=MU,
        runWL=RUN_WL,
        seed=SEED,
    )

    plot_cma_convergence(
        nIter=N_ITER,
        nModes=N_MODES,
        nTaps=N_TAPS,
        mu=MU,
        runWL=RUN_WL,
        seed=SEED,
    )
