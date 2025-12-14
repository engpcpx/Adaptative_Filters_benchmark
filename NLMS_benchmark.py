"""
NLMS — Benchmark e Convergência
- Preserva TODAS as funções NLMS
- Gráfico 1: Tempo médio por iteração (com valor em cada barra)
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
MU = 0.7
RUN_WL = True
SEED = 42


# -----------------------------------------------------
# Função NLMS original (loop)
# -----------------------------------------------------
def nlms_original(x, dx, outEq, mu, H, H_, nModes, runWL):
    indMode = np.arange(nModes)
    err = dx - outEq.T
    errDiag = np.diag(err[0])

    for N in range(nModes):
        indUpdTaps = indMode + N * nModes
        denom = np.linalg.norm(x[:, N]) ** 2 + 1e-12
        inAdapt = x[:, N].T / denom
        inAdaptPar = inAdapt.repeat(nModes).reshape(len(x), -1).T

        H[indUpdTaps, :] += mu * (errDiag @ np.conj(inAdaptPar))

        if runWL:
            H_[indUpdTaps, :] += mu * (errDiag @ inAdaptPar)

    return H, H_, np.abs(err) ** 2


# -----------------------------------------------------
# Vetorizado com tensordot
# -----------------------------------------------------
def nlms_vectorized_tensordot(x, dx, outEq, mu, H, H_, nModes, runWL):
    err = dx - outEq.T
    errDiag = np.diag(err[0])

    norms_sq = np.sum(np.abs(x) ** 2, axis=0) + 1e-12
    inAdapt = x / norms_sq[np.newaxis, :]

    inA = inAdapt.T
    inAdaptPar_stack = np.repeat(inA[:, np.newaxis, :], repeats=nModes, axis=1)

    delta_all = mu * np.transpose(
        np.tensordot(errDiag, np.conj(inAdaptPar_stack), axes=([1], [1])),
        axes=(1, 0, 2),
    )

    H += delta_all.reshape(nModes * nModes, inAdapt.shape[0])

    if runWL:
        delta_all_wl = mu * np.transpose(
            np.tensordot(errDiag, inAdaptPar_stack, axes=([1], [1])),
            axes=(1, 0, 2),
        )
        H_ += delta_all_wl.reshape(nModes * nModes, inAdapt.shape[0])

    return H, H_, np.abs(err) ** 2


# -----------------------------------------------------
# Vetorizado com einsum
# -----------------------------------------------------
def nlms_vectorized_einsum(x, dx, outEq, mu, H, H_, nModes, runWL):
    err = dx - outEq.T
    errDiag = np.diag(err[0])

    norms_sq = np.sum(np.abs(x) ** 2, axis=0) + 1e-12
    inAdapt = x / norms_sq[np.newaxis, :]

    inA = inAdapt.T
    inAdaptPar_stack = np.repeat(inA[:, np.newaxis, :], nModes, axis=1)

    delta_all = mu * np.einsum("ij,njp->nip", errDiag, np.conj(inAdaptPar_stack))
    H += delta_all.reshape(nModes * nModes, inAdapt.shape[0])

    if runWL:
        delta_all_wl = mu * np.einsum("ij,njp->nip", errDiag, inAdaptPar_stack)
        H_ += delta_all_wl.reshape(nModes * nModes, inAdapt.shape[0])

    return H, H_, np.abs(err) ** 2


# -----------------------------------------------------
# Vetorizado rápido (broadcast)
# -----------------------------------------------------
def nlms_vectorized_fast(x, dx, outEq, mu, H, H_, nModes, runWL):
    err = (dx - outEq.T)[0]

    norms_sq = np.sum(np.abs(x) ** 2, axis=0) + 1e-12
    inAdapt = x / norms_sq[np.newaxis, :]

    delta_all = (
        mu
        * err[None, :, None]
        * np.conj(inAdapt.T[:, None, :])
    )

    H += delta_all.reshape(nModes * nModes, inAdapt.shape[0])

    if runWL:
        delta_all_wl = (
            mu
            * err[None, :, None]
            * inAdapt.T[:, None, :]
        )
        H_ += delta_all_wl.reshape(nModes * nModes, inAdapt.shape[0])

    return H, H_, np.abs(err) ** 2


# -----------------------------------------------------
# Benchmark — SOMENTE TEMPO
# -----------------------------------------------------
def benchmark_time_only(nIter, nModes, nTaps, mu, runWL, seed):
    rng = np.random.default_rng(seed)

    methods = [
        ("Original (loop)", nlms_original),
        ("Vectorized (tensordot)", nlms_vectorized_tensordot),
        ("Vectorized (einsum)", nlms_vectorized_einsum),
        ("Vectorized (broadcast-fast)", nlms_vectorized_fast),
    ]

    xs = rng.standard_normal((nIter, nTaps, nModes)) + 1j * rng.standard_normal((nIter, nTaps, nModes))
    dxs = rng.standard_normal((nIter, 1, nModes)) + 1j * rng.standard_normal((nIter, 1, nModes))
    outs = rng.standard_normal((nIter, 1, nModes)) + 1j * rng.standard_normal((nIter, 1, nModes))

    results = {}

    for name, func in methods:
        H = np.zeros((nModes * nModes, nTaps), dtype=np.complex128)
        H_ = np.zeros_like(H)

        t0 = time.perf_counter()
        for k in range(nIter):
            H, H_, _ = func(xs[k], dxs[k], outs[k], mu, H, H_, nModes, runWL)
        t = time.perf_counter() - t0

        results[name] = t / nIter
        print(f"{name:30s}: {results[name]*1e6:8.2f} µs/it")

    # Gráfico 1 — Tempo
    plt.figure(figsize=(10, 6))

    labels = list(results.keys())
    values_us = [v * 1e6 for v in results.values()]
    bars = plt.bar(labels, values_us)

    plt.ylabel("Tempo médio por iteração (µs)")
    plt.title(f"Benchmark NLMS — {nIter:,} iterações")
    plt.grid(axis="y", alpha=0.3)

    # Anota valores em cada barra
    for bar, val in zip(bars, values_us):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{val:.2f} µs",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    plt.tight_layout()
    plt.show(block=False)


# -----------------------------------------------------
# Benchmark — Média móvel
# -----------------------------------------------------
def plot_nlms_convergence(nIter, nModes, nTaps, mu, runWL, seed):
    rng = np.random.default_rng(seed)

    methods = [
        ("Original (loop)", nlms_original),
        ("Vectorized (broadcast-fast)", nlms_vectorized_fast),
    ]

    WINDOW = 200  # média móvel (ajuste fino: 50–300)

    def moving_stats(x, w):
        """ Média móvel + desvio padrão móvel """
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
            dx = rng.standard_normal((1, nModes)) + 1j * rng.standard_normal((1, nModes))
            outEq = rng.standard_normal((1, nModes)) + 1j * rng.standard_normal((1, nModes))

            H, H_, err = func(x, dx, outEq, mu, H, H_, nModes, runWL)
            err_trace[k] = np.sum(np.abs(err) ** 2)

        # Média móvel + desvio
        err_mean, err_std = moving_stats(err_trace, WINDOW)

        # Escala dB
        err_db = 10 * np.log10(err_mean + 1e-12)
        err_db_up = 10 * np.log10(err_mean + err_std + 1e-12)
        err_db_dn = 10 * np.log10(np.maximum(err_mean - err_std, 1e-12))

        it_axis = np.arange(len(err_db))

        # Curva principal
        plt.plot(it_axis, err_db, label=label, linewidth=2)

        # Intervalo de confiança
        plt.fill_between(
            it_axis,
            err_db_dn,
            err_db_up,
            alpha=0.2,
        )

    # Estilo IEEE
    plt.xlabel("Iteração", fontsize=12)
    plt.ylabel("Erro médio |e|² (dB)", fontsize=12)
    plt.title("Convergência do NLMS — Validação Numérica", fontsize=13)
    plt.legend()
    plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
    plt.tight_layout()

    plt.show()


# -----------------------------------------------------
# MAIN
# -----------------------------------------------------
if __name__ == "__main__":
    print("NLMS — Benchmark + Convergência\n")

    benchmark_time_only(
        nIter=N_ITER,
        nModes=N_MODES,
        nTaps=N_TAPS,
        mu=MU,
        runWL=RUN_WL,
        seed=SEED,
    )

    plot_nlms_convergence(
        nIter=N_ITER,
        nModes=N_MODES,
        nTaps=N_TAPS,
        mu=MU,
        runWL=RUN_WL,
        seed=SEED,
    )
