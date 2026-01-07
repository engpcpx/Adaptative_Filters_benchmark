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
def daRDE_original(x, dx, outEq, mu, H, H_, nModes, runWL):
    indMode = np.arange(0, nModes)
    outEq = outEq.T

    decidedR = np.abs(dx)[None, :]
    err = decidedR**2 - np.abs(outEq)**2

    prodErrOut = np.diag(err[0]) @ np.diag(outEq[0])

    for N in range(nModes):
        indUpdTaps = indMode + N * nModes
        inAdapt = x[:, N].T
        inAdaptPar = inAdapt.repeat(nModes).reshape(len(x), -1).T

        H[indUpdTaps, :] += mu * prodErrOut @ np.conj(inAdaptPar)
        if runWL:
            H_[indUpdTaps, :] += mu * prodErrOut @ inAdaptPar

    return H, H_, np.abs(err)**2


# -----------------------------------------------------
# Vetorizado — tensordot
# -----------------------------------------------------
def daRDE_vectorized_tensordot(x, dx, outEq, mu, H, H_, nModes, runWL):
    outEq = outEq.T

    decidedR = np.abs(dx)[None, :]
    err = decidedR**2 - np.abs(outEq)**2
    prodErrOut = np.diag(err[0] * outEq[0])

    inA = x.T
    inAdaptPar = np.repeat(inA[:, None, :], nModes, axis=1)

    delta = mu * np.transpose(
        np.tensordot(prodErrOut, np.conj(inAdaptPar), axes=([1], [1])),
        (1, 0, 2)
    )
    H += delta.reshape(nModes * nModes, x.shape[0])

    if runWL:
        delta_wl = mu * np.transpose(
            np.tensordot(prodErrOut, inAdaptPar, axes=([1], [1])),
            (1, 0, 2)
        )
        H_ += delta_wl.reshape(nModes * nModes, x.shape[0])

    return H, H_, np.abs(err)**2


# -----------------------------------------------------
# Vetorizado — einsum
# -----------------------------------------------------
def daRDE_vectorized_einsum(x, dx, outEq, mu, H, H_, nModes, runWL):
    outEq = outEq.T

    decidedR = np.abs(dx)[None, :]
    err = decidedR**2 - np.abs(outEq)**2
    prod = err[0] * outEq[0]

    inA = x.T
    inAdaptPar = np.repeat(inA[:, None, :], nModes, axis=1)

    delta = mu * np.einsum("i,nip->nip", prod, np.conj(inAdaptPar))
    H += delta.reshape(nModes * nModes, x.shape[0])

    if runWL:
        delta_wl = mu * np.einsum("i,nip->nip", prod, inAdaptPar)
        H_ += delta_wl.reshape(nModes * nModes, x.shape[0])

    return H, H_, np.abs(err)**2


# -----------------------------------------------------
# Vetorizado rápido — broadcasting
# -----------------------------------------------------
def daRDE_vectorized_broadcast(x, dx, outEq, mu, H, H_, nModes, runWL):
    outEq = outEq.T  # (1, nModes)

    abs_out = np.abs(outEq[0])     # |y_k|
    decidedR = np.abs(dx)          # raio desejado por modo

    err = decidedR**2 - abs_out**2
    prod = err * outEq[0]

    delta = mu * prod[None, :, None] * np.conj(x.T[:, None, :])
    H += delta.reshape(nModes * nModes, x.shape[0])

    if runWL:
        delta_wl = mu * prod[None, :, None] * x.T[:, None, :]
        H_ += delta_wl.reshape(nModes * nModes, x.shape[0])

    return H, H_, np.abs(err)**2



# -----------------------------------------------------
# Benchmark — Tempo (Gráfico)
# -----------------------------------------------------
def benchmark_time_only(nIter, nModes, nTaps, mu, runWL, seed):
    rng = np.random.default_rng(seed)

    methods = [
        ("RDE Original", daRDE_original),
        ("RDE tensordot", daRDE_vectorized_tensordot),
        ("RDE einsum", daRDE_vectorized_einsum),
        ("RDE broadcast", daRDE_vectorized_broadcast),
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
    plt.title(f"Benchmark RDE — {nIter:,} iterações")
    plt.grid(axis="y", alpha=0.3)

    for bar, val in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                 f"{val:.2f}", ha="center", va="bottom", fontweight="bold")

    plt.tight_layout()
    plt.show(block=False)



# -----------------------------------------------------
# Benchmark — Tempo (Gráfico 1)
# -----------------------------------------------------
def benchmark_time_only(nIter, nModes, nTaps, mu, runWL, seed):
    rng = np.random.default_rng(seed)

    methods = [
        ("daRDE Original", daRDE_original),
        ("daRDE tensordot", daRDE_vectorized_tensordot),
        ("daRDE einsum", daRDE_vectorized_einsum),
        ("daRDE broadcast", daRDE_vectorized_broadcast),
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
    # daRDE: raio desejado (REAL, dimensão correta)
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
    plt.title(f"Benchmark daRDE — {nIter:,} iterações")
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


# # -----------------------------------------------------
# # Convergência — Erro × Iteração (Gráfico 2)
# # -----------------------------------------------------
# #     rng = np.random.default_rng(seed)

# #     methods = [
# #         ("daRDE Original", daRDE_original),
# #         ("daRDE Broadcast", daRDE_vectorized_broadcast),
# #     ]

# #     WINDOW = 200
# #     R = np.array([1, 3, 5])

# #     def moving_stats(x, w):
# #         mean = np.convolve(x, np.ones(w) / w, mode="valid")
# #         sq = np.convolve(x**2, np.ones(w) / w, mode="valid")
# #         std = np.sqrt(np.maximum(sq - mean**2, 0))
# #         return mean, std

# #     plt.figure(figsize=(10, 6))

# #     for label, func in methods:
# #         H = np.zeros((nModes * nModes, nTaps), dtype=np.complex128)
# #         H_ = np.zeros_like(H)
# #         err_trace = np.zeros(nIter)

# #         for k in range(nIter):
# #             x = rng.standard_normal((nTaps, nModes)) + 1j * rng.standard_normal((nTaps, nModes))

# #             # ✅ CORREÇÃO 2: outEq com shape (nModes, 1)
# #             out = rng.standard_normal((nModes, 1)) + 1j * rng.standard_normal((nModes, 1))

# #             H, H_, err = func(x, R, out, mu, H, H_, nModes, runWL)
# #             err_trace[k] = np.sum(err)

# #         mean, std = moving_stats(err_trace, WINDOW)
# #         mean_db = 10 * np.log10(mean + 1e-12)
# #         up = 10 * np.log10(mean + std + 1e-12)
# #         dn = 10 * np.log10(np.maximum(mean - std, 1e-12))

# #         it = np.arange(len(mean_db))
# #         plt.plot(it, mean_db, label=label, linewidth=2)
# #         plt.fill_between(it, dn, up, alpha=0.2)

# #     plt.xlabel("Iteração")
# #     plt.ylabel("Erro médio |e|² (dB)")
# #     plt.title("Convergência do daRDE — Validação Numérica")
# #     plt.legend()
# #     plt.grid(True, linestyle="--", alpha=0.6)
# #     plt.tight_layout()
# #     plt.show()
# def plot_daRDE_convergence(nIter, nModes, nTaps, mu, runWL, seed):
#     rng = np.random.default_rng(seed)

#     methods = [
#         ("daRDE Original", daRDE_original),
#         ("daRDE Broadcast", daRDE_vectorized_broadcast),
#     ]

#     WINDOW = 200

#     # -------------------------------------------------
#     # daRDE: raio desejado por modo
#     # -------------------------------------------------
#     R = np.full(nModes, 3.0)

#     def moving_stats(x, w):
#         mean = np.convolve(x, np.ones(w) / w, mode="valid")
#         sq = np.convolve(x**2, np.ones(w) / w, mode="valid")
#         std = np.sqrt(np.maximum(sq - mean**2, 0))
#         return mean, std

#     plt.figure(figsize=(10, 6))

#     for label, func in methods:
#         H = np.zeros((nModes * nModes, nTaps), dtype=np.complex128)
#         H_ = np.zeros_like(H)
#         err_trace = np.zeros(nIter)

#         for k in range(nIter):
#             x = (
#                 rng.standard_normal((nTaps, nModes))
#                 + 1j * rng.standard_normal((nTaps, nModes))
#             )

#             outEq = (
#                 rng.standard_normal((nModes, 1))
#                 + 1j * rng.standard_normal((nModes, 1))
#             )

#             H, H_, err = func(
#                 x,
#                 R,
#                 outEq,
#                 mu,
#                 H,
#                 H_,
#                 nModes,
#                 runWL,
#             )

#             err_trace[k] = np.sum(err)

#         mean, std = moving_stats(err_trace, WINDOW)

#         mean_db = 10 * np.log10(mean + 1e-12)
#         up_db = 10 * np.log10(mean + std + 1e-12)
#         dn_db = 10 * np.log10(np.maximum(mean - std, 1e-12))

#         it = np.arange(len(mean_db))
#         plt.plot(it, mean_db, label=label, linewidth=2)
#         plt.fill_between(it, dn_db, up_db, alpha=0.2)

#     plt.xlabel("Iteração")
#     plt.ylabel("Erro médio |e|² (dB)")
#     plt.title("Convergência do daRDE — Validação Numérica")
#     plt.legend()
#     plt.grid(True, linestyle="--", alpha=0.6)
#     plt.tight_layout()
#     plt.show()

# -----------------------------------------------------
# Convergência — Erro × Iteração (Gráfico 2 com desvio médio padrão)
# -----------------------------------------------------

def plot_daRDE_convergence(nIter, nModes, nTaps, mu, runWL, seed):
    rng = np.random.default_rng(seed)

    methods = [
        ("daRDE Original", daRDE_original),
        ("daRDE Broadcast", daRDE_vectorized_broadcast),
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
    plt.title("Convergência do daRDE — Validação Numérica")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()



# -----------------------------------------------------
# MAIN 
# -----------------------------------------------------
if __name__ == "__main__":
    print("daRDE — Benchmark + Convergência\n")

    benchmark_time_only(
        nIter=N_ITER,
        nModes=N_MODES,
        nTaps=N_TAPS,
        mu=MU,
        runWL=RUN_WL,
        seed=SEED,
    )

    plot_daRDE_convergence(
        nIter=N_ITER,
        nModes=N_MODES,
        nTaps=N_TAPS,
        mu=MU,
        runWL=RUN_WL,
        seed=SEED,
    )
