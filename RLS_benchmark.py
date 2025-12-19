"""
RLS — Benchmark e Convergência
- Preserva TODAS as funções RLS
- Gráfico 1: Tempo (µs/it)
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
LAMBDA = 0.99
SEED = 42


# -----------------------------------------------------
# Função RLS original (loop) — NÃO ALTERADA
# -----------------------------------------------------
def rls_original(x, dx, outEq, λ, H, Sd, nModes):
    nTaps = H.shape[1]
    indMode = np.arange(0, nModes)
    indTaps = np.arange(0, nTaps)

    err = dx - outEq.T
    errDiag = np.diag(err[0])

    for N in range(nModes):
        indUpdModes = indMode + N * nModes
        indUpdTaps = indTaps + N * nTaps

        Sd_ = Sd[indUpdTaps, :]
        inAdapt = np.conj(x[:, N]).reshape(-1, 1)

        inAdaptPar = (inAdapt.T).repeat(nModes).reshape(len(x), -1).T

        Sd_ = (1 / λ) * (
            Sd_
            - (Sd_ @ (inAdapt @ inAdapt.conj().T) @ Sd_)
            / (λ + inAdapt.conj().T @ Sd_ @ inAdapt)
        )

        H[indUpdModes, :] += errDiag @ (Sd_ @ inAdaptPar.T).T
        Sd[indUpdTaps, :] = Sd_

    return H, Sd, np.abs(err) ** 2


# -----------------------------------------------------
# Vetorizado com tensordot — EQUIVALENTE
# -----------------------------------------------------
def rls_vectorized_tensordot(x, dx, outEq, lam, H, Sd, nModes):
    err = (dx - outEq.T)[0]

    for m in range(nModes):
        rows = slice(m * nModes, (m + 1) * nModes)
        taps = slice(m * N_TAPS, (m + 1) * N_TAPS)

        P = Sd[taps, :]
        u = np.conj(x[:, m])

        Pu = P @ u
        den = lam + u.conj() @ Pu

        P = (P - np.tensordot(Pu, Pu.conj(), axes=0) / den) / lam
        g = P @ u

        H[rows, :] += np.tensordot(err, g, axes=0)
        Sd[taps, :] = P

    return H, Sd, np.abs(dx - outEq.T) ** 2


# -----------------------------------------------------
# Vetorizado com einsum — EQUIVALENTE
# -----------------------------------------------------
def rls_vectorized_einsum(x, dx, outEq, lam, H, Sd, nModes):
    err = (dx - outEq.T)[0]

    for m in range(nModes):
        rows = slice(m * nModes, (m + 1) * nModes)
        taps = slice(m * N_TAPS, (m + 1) * N_TAPS)

        P = Sd[taps, :]
        u = np.conj(x[:, m])

        Pu = P @ u
        den = lam + u.conj() @ Pu

        P = (P - np.einsum("i,j->ij", Pu, Pu.conj()) / den) / lam
        g = P @ u

        H[rows, :] += np.einsum("i,j->ij", err, g)
        Sd[taps, :] = P

    return H, Sd, np.abs(dx - outEq.T) ** 2


# -----------------------------------------------------
# Vetorizado rápido (broadcast) — EQUIVALENTE
# -----------------------------------------------------
def rls_vectorized_broadcast(x, dx, outEq, lam, H, Sd, nModes):
    err = (dx - outEq.T)[0]

    for m in range(nModes):
        rows = slice(m * nModes, (m + 1) * nModes)
        taps = slice(m * N_TAPS, (m + 1) * N_TAPS)

        P = Sd[taps, :]
        u = np.conj(x[:, m])

        Pu = P @ u
        den = lam + u.conj() @ Pu

        P = (P - (Pu[:, None] * Pu.conj()[None, :]) / den) / lam
        g = P @ u

        H[rows, :] += err[:, None] * g[None, :]
        Sd[taps, :] = P

    return H, Sd, np.abs(dx - outEq.T) ** 2


# -----------------------------------------------------
# Benchmark — Tempo (RLS)
# -----------------------------------------------------
def benchmark_time_only(nIter, nModes, nTaps, lam, seed):
    rng = np.random.default_rng(seed)

    methods = [
        ("Original (loop)", rls_original),
        ("Vectorized (tensordot)", rls_vectorized_tensordot),
        ("Vectorized (einsum)", rls_vectorized_einsum),
        ("Vectorized (broadcast)", rls_vectorized_broadcast),
    ]

    xs = (
        rng.standard_normal((nIter, nTaps, nModes))
        + 1j * rng.standard_normal((nIter, nTaps, nModes))
    )
    dxs = (
        rng.standard_normal((nIter, 1, nModes))
        + 1j * rng.standard_normal((nIter, 1, nModes))
    )
    outs = (
        rng.standard_normal((nIter, 1, nModes))
        + 1j * rng.standard_normal((nIter, 1, nModes))
    )

    results = {}

    for name, func in methods:
        # Inicialização idêntica ao NLMS benchmark
        H = np.zeros((nModes * nModes, nTaps), dtype=np.complex128)
        Sd = np.tile(np.eye(nTaps, dtype=np.complex128), (nModes, 1))

        t0 = time.perf_counter()
        for k in range(nIter):
            H, Sd, _ = func(
                xs[k],
                dxs[k],
                outs[k],
                lam,
                H,
                Sd,
                nModes,
            )
        t = time.perf_counter() - t0

        results[name] = t / nIter
        print(f"{name:30s}: {results[name]*1e6:8.2f} µs/it")

    # -------------------------------------------------
    # Gráfico — Tempo
    # -------------------------------------------------
    plt.figure(figsize=(10, 6))

    labels = list(results.keys())
    values_us = [v * 1e6 for v in results.values()]
    bars = plt.bar(labels, values_us)

    plt.ylabel("Tempo médio por iteração (µs)")
    plt.title(f"Benchmark RLS — {nIter:,} iterações")
    plt.grid(axis="y", alpha=0.3)

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
# Benchmark — Convergência
# -----------------------------------------------------
def plot_rls_convergence(nIter, nModes, nTaps, lam, seed):
    rng = np.random.default_rng(seed)

    methods = [
        ("Original (loop)", rls_original),
        ("Vectorized (broadcast)", rls_vectorized_broadcast),
    ]

    WINDOW = 200

    def moving_stats(x, w):
        mean = np.convolve(x, np.ones(w) / w, mode="valid")
        sq_mean = np.convolve(x**2, np.ones(w) / w, mode="valid")
        std = np.sqrt(np.maximum(sq_mean - mean**2, 0))
        return mean, std

    plt.figure(figsize=(10, 6))

    for label, func in methods:
        H = np.zeros((nModes * nModes, nTaps), dtype=np.complex128)
        Sd = np.tile(np.eye(nTaps, dtype=np.complex128), (nModes, 1))
        err_trace = np.zeros(nIter)

        for k in range(nIter):
            x = rng.standard_normal((nTaps, nModes)) + 1j * rng.standard_normal((nTaps, nModes))
            dx = rng.standard_normal((1, nModes)) + 1j * rng.standard_normal((1, nModes))
            outEq = rng.standard_normal((1, nModes)) + 1j * rng.standard_normal((1, nModes))

            H, Sd, err = func(x, dx, outEq, lam, H, Sd, nModes)
            err_trace[k] = np.sum(err)

        mean, std = moving_stats(err_trace, WINDOW)
        plt.plot(10 * np.log10(mean + 1e-12), label=label)
        plt.fill_between(
            np.arange(len(mean)),
            10 * np.log10(np.maximum(mean - std, 1e-12)),
            10 * np.log10(mean + std + 1e-12),
            alpha=0.2,
        )

    # Estilo IEEE
    plt.xlabel("Iteração")
    plt.ylabel("Erro médio |e|² (dB)")
    plt.title("Convergência do RLS — Validação Numérica")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()


# -----------------------------------------------------
# MAIN
# -----------------------------------------------------
if __name__ == "__main__":
    print("RLS — Benchmark + Convergência\n")

    benchmark_time_only(
        nIter=N_ITER,
        nModes=N_MODES,
        nTaps=N_TAPS,
        lam=LAMBDA,
        seed=SEED,
    )

    plot_rls_convergence(
        nIter=N_ITER,
        nModes=N_MODES,
        nTaps=N_TAPS,
        lam=LAMBDA,
        seed=SEED,
    )
