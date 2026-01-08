"""
dd-LMS — Benchmark e Convergência
- Preserva TODAS as funções dd-LMS
- Gráfico 1: Análise Temporal
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


# -----------------------------------------------------
# dd-LMS original (loop) — PRESERVADO
# -----------------------------------------------------
def ddLMS_original(x, constSymb, outEq, mu, H, H_, nModes, runWL):
    """
    Coefficient update with the DD-LMS algorithm.

    Parameters
    ----------
    x : np.array
        Input array.
    constSymb : np.array
        Array of constellation symbols.
    outEq : np.array
        Equalized output array.
    mu : float
        Step size for the update.
    H : np.array
        Coefficient matrix.
    H_ : np.array
        Augmented coefficient matrix.
    nModes : int
        Number of modes.
    runWL: bool
        Run widely-linear mode.

    Returns
    -------
    H : np.array
        Updated coefficient matrix.
    H_ : np.array
        Updated augmented coefficient matrix.
    err_sq : np.array
        Squared absolute error.

    """
    indMode = np.arange(0, nModes)
    outEq = outEq.T
    decided = np.zeros(outEq.shape, dtype=np.complex128)

    for k in range(nModes):
        indSymb = np.argmin(np.abs(outEq[0, k] - constSymb))
        decided[0, k] = constSymb[indSymb]
    err = decided - outEq  # calculate output error for the DDLMS algorithm

    errDiag = np.diag(err[0])  # define diagonal matrix from error array

    # update equalizer taps
    for N in range(nModes):
        indUpdTaps = indMode + N * nModes  # simplify indexing
        inAdapt = x[:, N].T
        inAdaptPar = (
            inAdapt.repeat(nModes).reshape(len(x), -1).T
        )  # expand input to parallelize tap adaptation
        H[indUpdTaps, :] += (
            mu * errDiag @ np.conj(inAdaptPar)
        )  # gradient descent update        
        if runWL:
            H_[indUpdTaps, :] += (
                mu * errDiag @ inAdaptPar
            )  # gradient descent update   
    return H, H_, np.abs(err) ** 2


# -----------------------------------------------------
# Vetorizado — tensordot
# -----------------------------------------------------
def ddLMS_vectorized_tensordot(x, constSymb, outEq, mu, H, H_, nModes, runWL):
    outEq = outEq.T

    decided = np.zeros_like(outEq)
    for k in range(nModes):
        decided[0, k] = constSymb[np.argmin(np.abs(outEq[0, k] - constSymb))]

    err = decided - outEq
    errDiag = np.diag(err[0])

    inA = x.T
    inAdaptPar = np.repeat(inA[:, None, :], nModes, axis=1)

    delta = mu * np.transpose(
        np.tensordot(errDiag, np.conj(inAdaptPar), axes=([1], [1])),
        (1, 0, 2),
    )
    H += delta.reshape(nModes * nModes, x.shape[0])

    if runWL:
        delta_wl = mu * np.transpose(
            np.tensordot(errDiag, inAdaptPar, axes=([1], [1])),
            (1, 0, 2),
        )
        H_ += delta_wl.reshape(nModes * nModes, x.shape[0])

    return H, H_, np.abs(err) ** 2


# -----------------------------------------------------
# Vetorizado — einsum
# -----------------------------------------------------
def ddLMS_vectorized_einsum(x, constSymb, outEq, mu, H, H_, nModes, runWL):
    outEq = outEq.T

    decided = np.zeros_like(outEq)
    for k in range(nModes):
        decided[0, k] = constSymb[np.argmin(np.abs(outEq[0, k] - constSymb))]

    err = decided - outEq

    inA = x.T
    inAdaptPar = np.repeat(inA[:, None, :], nModes, axis=1)

    delta = mu * np.einsum("i,nip->nip", err[0], np.conj(inAdaptPar))
    H += delta.reshape(nModes * nModes, x.shape[0])

    if runWL:
        delta_wl = mu * np.einsum("i,nip->nip", err[0], inAdaptPar)
        H_ += delta_wl.reshape(nModes * nModes, x.shape[0])

    return H, H_, np.abs(err) ** 2


# -----------------------------------------------------
# Vetorizado rápido — broadcasting
# -----------------------------------------------------
def ddLMS_vectorized_broadcast(x, constSymb, outEq, mu, H, H_, nModes, runWL):
    outEq = outEq.T

    decided = np.zeros_like(outEq)
    for k in range(nModes):
        decided[0, k] = constSymb[np.argmin(np.abs(outEq[0, k] - constSymb))]

    err = decided - outEq

    delta = mu * err[0][None, :, None] * np.conj(x.T[:, None, :])
    H += delta.reshape(nModes * nModes, x.shape[0])

    if runWL:
        delta_wl = mu * err[0][None, :, None] * x.T[:, None, :]
        H_ += delta_wl.reshape(nModes * nModes, x.shape[0])

    return H, H_, np.abs(err) ** 2


# -----------------------------------------------------
# Benchmark — Tempo (Gráfico)
# -----------------------------------------------------
def benchmark_time_only(nIter, nModes, nTaps, mu, runWL, seed):
    rng = np.random.default_rng(seed)

    methods = [
        ("ddLMS Original", ddLMS_original),
        ("ddLMS tensordot", ddLMS_vectorized_tensordot),
        ("ddLMS einsum", ddLMS_vectorized_einsum),
        ("ddLMS broadcast", ddLMS_vectorized_broadcast),
    ]

    xs = rng.standard_normal((nIter, nTaps, nModes)) + 1j * rng.standard_normal((nIter, nTaps, nModes))

    outs = rng.standard_normal((nIter, nModes, 1)) + 1j * rng.standard_normal((nIter, nModes, 1))

    R = np.array([1, 3, 5])

    results = {}

    for name, func in methods:
        H = np.zeros((nModes * nModes, nTaps), dtype=np.complex128)
        H_ = np.zeros_like(H)

        t0 = time.perf_counter()
        for k in range(nIter):
            H, H_, _ = func(xs[k], R, outs[k], mu, H, H_, nModes, runWL)
        t = time.perf_counter() - t0

        results[name] = t / nIter
        print(f"{name:20s}: {results[name]*1e6:8.2f} µs/it")

    plt.figure(figsize=(10, 6))
    labels = list(results.keys())
    values = [v * 1e6 for v in results.values()]
    bars = plt.bar(labels, values)

    plt.ylabel("Tempo médio por iteração (µs)")
    plt.title(f"Benchmark ddLMS — {nIter:,} iterações")
    plt.grid(axis="y", alpha=0.3)

    for bar, val in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                 f"{val:.2f}", ha="center", va="bottom", fontweight="bold")

    plt.tight_layout()
    plt.show(block=False)



# -----------------------------------------------------
# Benchmark — Análie Temporal (Gráfico 1)
# -----------------------------------------------------
def benchmark_time_only(nIter, nModes, nTaps, mu, runWL, seed):
    rng = np.random.default_rng(seed)

    methods = [
        ("ddLMS Original", ddLMS_original),
        ("ddLMS tensordot", ddLMS_vectorized_tensordot),
        ("ddLMS einsum", ddLMS_vectorized_einsum),
        ("ddLMS broadcast", ddLMS_vectorized_broadcast),
    ]

    xs = (
        rng.standard_normal((nIter, nTaps, nModes))
        + 1j * rng.standard_normal((nIter, nTaps, nModes))
    )

    outs = (
        rng.standard_normal((nIter, nModes, 1))
        + 1j * rng.standard_normal((nIter, nModes, 1))
    )

    # -------------------------------------------------
    # ddLMS: raio desejado (REAL, dimensão correta)
    # -------------------------------------------------
    R = np.full(nModes, 3.0)

    results = {}

    for name, func in methods:
        H = np.zeros((nModes * nModes, nTaps), dtype=np.complex128)
        H_ = np.zeros_like(H)

        t0 = time.perf_counter()
        for k in range(nIter):
            H, H_, _ = func(
                xs[k],
                R,
                outs[k],
                mu,
                H,
                H_,
                nModes,
                runWL,
            )
        t = time.perf_counter() - t0

        results[name] = t / nIter
        print(f"{name:20s}: {results[name]*1e6:8.2f} µs/it")

    # -----------------------------
    # Gráfico de tempo
    # -----------------------------
    plt.figure(figsize=(10, 6))
    labels = list(results.keys())
    values = [v * 1e6 for v in results.values()]

    bars = plt.bar(labels, values)
    plt.ylabel("Tempo médio por iteração (µs)")
    plt.title(f"Benchmark Temporal ddLMS — {nIter:,} iterações")
    plt.grid(axis="y", alpha=0.3)

    for bar, val in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{val:.2f}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    plt.tight_layout()
    plt.show(block=False)


# -----------------------------------------------------
# Convergência — Erro × Iteração (Gráfico 2 com desvio médio padrão)
# -----------------------------------------------------

def plot_ddLMS_convergence(nIter, nModes, nTaps, mu, runWL, seed):
    rng = np.random.default_rng(seed)

    methods = [
        ("ddLMS Original", ddLMS_original),
        ("ddLMS Broadcast", ddLMS_vectorized_broadcast),
    ]

    WINDOW = 200
    R = np.full(nModes, 3.0)

    def moving_stats(x, w):
        mean = np.convolve(x, np.ones(w) / w, mode="valid")
        sq = np.convolve(x**2, np.ones(w) / w, mode="valid")
        std = np.sqrt(np.maximum(sq - mean**2, 0))
        return mean, std

    plt.figure(figsize=(10, 6))

    for label, func in methods:
        H = np.zeros((nModes * nModes, nTaps), dtype=np.complex128)
        H_ = np.zeros_like(H)
        err_trace = np.zeros(nIter)

        for k in range(nIter):
            x = (
                rng.standard_normal((nTaps, nModes))
                + 1j * rng.standard_normal((nTaps, nModes))
            )

            outEq = (
                rng.standard_normal((nModes, 1))
                + 1j * rng.standard_normal((nModes, 1))
            )

            H, H_, err = func(
                x,
                R,
                outEq,
                mu,
                H,
                H_,
                nModes,
                runWL,
            )

            err_trace[k] = np.sum(err)

        mean, std = moving_stats(err_trace, WINDOW)

        # --------------------------------------------
        # Conversão para dB
        # --------------------------------------------
        mean_db = 10 * np.log10(mean + 1e-12)
        up_db = 10 * np.log10(mean + std + 1e-12)
        dn_db = 10 * np.log10(np.maximum(mean - std, 1e-12))

        # --------------------------------------------
        # Desvio-padrão médio em dB (numérico)
        # --------------------------------------------
        sigma_db = np.mean(up_db - mean_db)

        it = np.arange(len(mean_db))
        plt.plot(
            it,
            mean_db,
            label=f"{label} (σ̄ = {sigma_db:.2f} dB)",
            linewidth=2,
        )
        plt.fill_between(it, dn_db, up_db, alpha=0.2)

    plt.xlabel("Iteração")
    plt.ylabel("Erro médio |e|² (dB)")
    plt.title("Convergência do ddLMS — Validação Numérica")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()



# -----------------------------------------------------
# MAIN 
# -----------------------------------------------------
if __name__ == "__main__":
    print("ddLMS — Benchmark + Convergência\n")

    benchmark_time_only(
        nIter=N_ITER,
        nModes=N_MODES,
        nTaps=N_TAPS,
        mu=MU,
        runWL=RUN_WL,
        seed=SEED,
    )

    plot_ddLMS_convergence(
        nIter=N_ITER,
        nModes=N_MODES,
        nTaps=N_TAPS,
        mu=MU,
        runWL=RUN_WL,
        seed=SEED,
    )


