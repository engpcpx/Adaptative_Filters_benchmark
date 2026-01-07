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
# RDE original (loop) — PRESERVADO
# -----------------------------------------------------
def RDE_original(x, R, outEq, mu, H, H_, nModes, runWL):
    indMode = np.arange(0, nModes)
    outEq = outEq.T
    decidedR = np.zeros(outEq.shape, dtype=np.complex128)

    for k in range(nModes):
        indR = np.argmin(np.abs(R - np.abs(outEq[0, k])))
        decidedR[0, k] = R[indR]

    err = decidedR**2 - np.abs(outEq) ** 2
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
# Vetorizado — tensordot
# -----------------------------------------------------
def RDE_vectorized_tensordot(x, R, outEq, mu, H, H_, nModes, runWL):
    outEq = outEq.T

    abs_out = np.abs(outEq[0])
    idx = np.argmin(np.abs(R[:, None] - abs_out[None, :]), axis=0)
    decidedR = R[idx][None, :]

    err = decidedR**2 - abs_out[None, :] ** 2
    prodErrOut = np.diag(err[0] * outEq[0])

    inA = x.T
    inAdaptPar = np.repeat(inA[:, None, :], nModes, axis=1)

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
# Vetorizado — einsum
# -----------------------------------------------------
def RDE_vectorized_einsum(x, R, outEq, mu, H, H_, nModes, runWL):
    outEq = outEq.T

    abs_out = np.abs(outEq[0])
    idx = np.argmin(np.abs(R[:, None] - abs_out[None, :]), axis=0)
    decidedR = R[idx][None, :]

    err = decidedR**2 - abs_out[None, :] ** 2
    prod = err[0] * outEq[0]

    inA = x.T
    inAdaptPar = np.repeat(inA[:, None, :], nModes, axis=1)

    delta = mu * np.einsum("i,nip->nip", prod, np.conj(inAdaptPar))
    H += delta.reshape(nModes * nModes, x.shape[0])

    if runWL:
        delta_wl = mu * np.einsum("i,nip->nip", prod, inAdaptPar)
        H_ += delta_wl.reshape(nModes * nModes, x.shape[0])

    return H, H_, np.abs(err) ** 2


# -----------------------------------------------------
# Vetorizado rápido — broadcasting
# -----------------------------------------------------
def RDE_vectorized_broadcast(x, R, outEq, mu, H, H_, nModes, runWL):
    outEq = outEq.T

    abs_out = np.abs(outEq[0])
    idx = np.argmin(np.abs(R[:, None] - abs_out[None, :]), axis=0)
    decidedR = R[idx]

    err = decidedR**2 - abs_out**2
    prod = err * outEq[0]

    delta = mu * prod[None, :, None] * np.conj(x.T[:, None, :])
    H += delta.reshape(nModes * nModes, x.shape[0])

    if runWL:
        delta_wl = mu * prod[None, :, None] * x.T[:, None, :]
        H_ += delta_wl.reshape(nModes * nModes, x.shape[0])

    return H, H_, np.abs(err) ** 2


# -----------------------------------------------------
# Benchmark — Tempo (Gráfico 1)
# -----------------------------------------------------
def benchmark_time_only(nIter, nModes, nTaps, mu, runWL, seed):
    rng = np.random.default_rng(seed)

    methods = [
        ("RDE Original", RDE_original),
        ("RDE tensordot", RDE_vectorized_tensordot),
        ("RDE einsum", RDE_vectorized_einsum),
        ("RDE broadcast", RDE_vectorized_broadcast),
    ]

    xs = rng.standard_normal((nIter, nTaps, nModes)) + 1j * rng.standard_normal((nIter, nTaps, nModes))

    # ✅ CORREÇÃO 1: outEq com shape (nModes, 1)
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
    plt.title(f"Benchmark RDE — {nIter:,} iterações")
    plt.grid(axis="y", alpha=0.3)

    for bar, val in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                 f"{val:.2f}", ha="center", va="bottom", fontweight="bold")

    plt.tight_layout()
    plt.show(block=False)


# -----------------------------------------------------
# Convergência — Erro × Iteração (Gráfico 2)
# -----------------------------------------------------
def plot_rde_convergence(nIter, nModes, nTaps, mu, runWL, seed):
    rng = np.random.default_rng(seed)

    methods = [
        ("RDE Original", RDE_original),
        ("RDE Broadcast", RDE_vectorized_broadcast),
    ]

    WINDOW = 200
    R = np.array([1, 3, 5])

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
            x = rng.standard_normal((nTaps, nModes)) + 1j * rng.standard_normal((nTaps, nModes))

            out = rng.standard_normal((nModes, 1)) + 1j * rng.standard_normal((nModes, 1))

            H, H_, err = func(x, R, out, mu, H, H_, nModes, runWL)
            err_trace[k] = np.sum(err)

        mean, std = moving_stats(err_trace, WINDOW)
        mean_db = 10 * np.log10(mean + 1e-12)
        up = 10 * np.log10(mean + std + 1e-12)
        dn = 10 * np.log10(np.maximum(mean - std, 1e-12))

        it = np.arange(len(mean_db))
        plt.plot(it, mean_db, label=label, linewidth=2)
        plt.fill_between(it, dn, up, alpha=0.2)

    plt.xlabel("Iteração")
    plt.ylabel("Erro médio |e|² (dB)")
    plt.title("Convergência do RDE — Validação Numérica")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()


# -----------------------------------------------------
# MAIN
# -----------------------------------------------------
if __name__ == "__main__":
    print("RDE — Benchmark + Convergência\n")

    benchmark_time_only(
        nIter=N_ITER,
        nModes=N_MODES,
        nTaps=N_TAPS,
        mu=MU,
        runWL=RUN_WL,
        seed=SEED,
    )

    plot_rde_convergence(
        nIter=N_ITER,
        nModes=N_MODES,
        nTaps=N_TAPS,
        mu=MU,
        runWL=RUN_WL,
        seed=SEED,
    )
