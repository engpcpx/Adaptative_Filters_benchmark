"""
RLS — Benchmark e Convergência
- Gráfico 1: Benchmark Temporal (µs/it)
- Gráfico 2: Convergência (Erro × Iteração)
"""

import numpy as np
import time
import matplotlib.pyplot as plt

#-----------------------------------------------------
# Configurações (Ajustadas para Estabilidade Real)
#-----------------------------------------------------
N_ITER = 10_000
N_MODES = 8
N_TAPS = 64
LAMBDA = 0.999  # Aumentado levemente para maior estabilidade
DELTA = 1e-2    # Fator de regularização inicial (P = 1/delta * I)
EPS = 1e-10     # Estabilização numérica para divisão
SEED = 42

#-----------------------------------------------------
# RLS original (loop) — PRESERVADO & ESTABILIZADO
#-----------------------------------------------------
def rls_original(x, dx, outEq, lam, H, Sd, nModes):
    nTaps = H.shape[1]
    indMode = np.arange(nModes)
    indTaps = np.arange(nTaps)

    err = dx - outEq.T
    errDiag = np.diag(err[0])

    for m in range(nModes):
        rows = indMode + m * nModes
        taps = indTaps + m * nTaps

        P = Sd[taps, :]
        u = np.conj(x[:, m]).reshape(-1, 1)

        # Adicionado EPS para evitar RuntimeWarning: invalid value in divide
        den = lam + (u.conj().T @ P @ u) + EPS
        P = (P - (P @ u @ u.conj().T @ P) / den) / lam

        u_par = (u.T).repeat(nModes).reshape(nTaps, -1).T
        H[rows, :] += errDiag @ (P @ u_par.T).T
        Sd[taps, :] = P

    return H, Sd, np.abs(err) ** 2


#-----------------------------------------------------
# Vetorizado — tensordot
#-----------------------------------------------------
def rls_vectorized_tensordot(x, dx, outEq, lam, H, Sd, nModes):
    err = (dx - outEq.T)[0]

    for m in range(nModes):
        rows = slice(m * nModes, (m + 1) * nModes)
        taps = slice(m * N_TAPS, (m + 1) * N_TAPS)

        P = Sd[taps, :]
        u = np.conj(x[:, m])

        Pu = P @ u
        den = lam + u.conj() @ Pu + EPS

        P = (P - np.tensordot(Pu, Pu.conj(), axes=0) / den) / lam
        g = P @ u

        H[rows, :] += np.tensordot(err, g, axes=0)
        Sd[taps, :] = P

    return H, Sd, np.abs(dx - outEq.T) ** 2


#-----------------------------------------------------
# Vetorizado — einsum
#-----------------------------------------------------
def rls_vectorized_einsum(x, dx, outEq, lam, H, Sd, nModes):
    err = (dx - outEq.T)[0]

    for m in range(nModes):
        rows = slice(m * nModes, (m + 1) * nModes)
        taps = slice(m * N_TAPS, (m + 1) * N_TAPS)

        P = Sd[taps, :]
        u = np.conj(x[:, m])

        Pu = P @ u
        den = lam + u.conj() @ Pu + EPS

        P = (P - np.einsum("i,j->ij", Pu, Pu.conj()) / den) / lam
        g = P @ u

        H[rows, :] += np.einsum("i,j->ij", err, g)
        Sd[taps, :] = P

    return H, Sd, np.abs(dx - outEq.T) ** 2


#-----------------------------------------------------
# Vetorizado com broadcast
#-----------------------------------------------------
def rls_vectorized_broadcast(x, dx, outEq, lam, H, Sd, nModes):
    err = (dx - outEq.T)[0]

    for m in range(nModes):
        rows = slice(m * nModes, (m + 1) * nModes)
        taps = slice(m * N_TAPS, (m + 1) * N_TAPS)

        P = Sd[taps, :]
        u = np.conj(x[:, m])

        Pu = P @ u
        den = lam + u.conj() @ Pu + EPS

        # Broadcasting com segurança numérica
        P = (P - (Pu[:, None] * Pu.conj()[None, :]) / den) / lam
        g = P @ u

        H[rows, :] += err[:, None] * g[None, :]
        Sd[taps, :] = P

    return H, Sd, np.abs(dx - outEq.T) ** 2


#-----------------------------------------------------
# Benchmark — Tempo (Gráfico 1)
#-----------------------------------------------------
def benchmark_time_only(nIter, nModes, nTaps, lam, seed):
    rng = np.random.default_rng(seed)

    methods = [
        ("RLS Original", rls_original),
        ("RLS tensordot", rls_vectorized_tensordot),
        ("RLS einsum", rls_vectorized_einsum),
        ("RLS broadcast", rls_vectorized_broadcast),
    ]

    xs = rng.standard_normal((nIter, nTaps, nModes)) + 1j * rng.standard_normal((nIter, nTaps, nModes))
    dxs = rng.standard_normal((nIter, 1, nModes)) + 1j * rng.standard_normal((nIter, 1, nModes))
    outs = rng.standard_normal((nIter, 1, nModes)) + 1j * rng.standard_normal((nIter, 1, nModes))

    results = {}

    for name, func in methods:
        H = np.zeros((nModes * nModes, nTaps), dtype=np.complex128)
        # Inicialização estável com regularização DELTA
        Sd = np.tile(np.eye(nTaps, dtype=np.complex128) / DELTA, (nModes, 1))

        t0 = time.perf_counter()
        for k in range(nIter):
            H, Sd, _ = func(xs[k], dxs[k], outs[k], lam, H, Sd, nModes)
        t = time.perf_counter() - t0

        results[name] = t / nIter
        print(f"{name:20s}: {results[name]*1e6:8.2f} µs/it")

    plt.figure(figsize=(10, 6))
    vals = [v * 1e6 for v in results.values()]
    bars = plt.bar(results.keys(), vals)

    plt.ylabel("Tempo médio por iteração (µs)")
    plt.title(f"Benchmark Temporal RLS (Estabilizado) — {nIter:,} iterações")
    plt.grid(axis="y", alpha=0.3)

    for bar, val in zip(bars, vals):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                 f"{val:.2f}", ha="center", va="bottom", fontweight="bold")

    plt.tight_layout()
    plt.show(block=False)


#-----------------------------------------------------
# Convergência — Erro × Iteração (Gráfico 2)
#-----------------------------------------------------
def plot_rls_convergence(nIter, nModes, nTaps, lam, seed):
    rng = np.random.default_rng(seed)

    methods = [
        ("RLS Original", rls_original),
        ("RLS Broadcast", rls_vectorized_broadcast),
    ]

    WINDOW = 200

    def moving_stats(x, w):
        mean = np.convolve(x, np.ones(w) / w, mode="valid")
        sq = np.convolve(x**2, np.ones(w) / w, mode="valid")
        std = np.sqrt(np.maximum(sq - mean**2, 0))
        return mean, std

    plt.figure(figsize=(10, 6))

    for label, func in methods:
        H = np.zeros((nModes * nModes, nTaps), dtype=np.complex128)
        # Inicialização estável com regularização DELTA
        Sd = np.tile(np.eye(nTaps, dtype=np.complex128) / DELTA, (nModes, 1))
        err_trace = np.zeros(nIter)

        # Simular um canal CV-QKD real com ruído térmico/shot noise
        # Isso garante que o MSE caia (convergência real)
        H_true = rng.standard_normal((nModes * nModes, nTaps)) + 1j * rng.standard_normal((nModes * nModes, nTaps))

        for k in range(nIter):
            x = rng.standard_normal((nTaps, nModes)) + 1j * rng.standard_normal((nTaps, nModes))
            # d = Hx + noise
            noise = (rng.standard_normal((1, nModes)) + 1j * rng.standard_normal((1, nModes))) * 0.1
            dx = (H_true[0:nModes, :] @ x[:, 0:1]).T + noise
            
            # Saída do filtro atual
            outEq = (H[0:nModes, :] @ x[:, 0:1]).T

            H, Sd, err = func(x, dx, outEq, lam, H, Sd, nModes)
            err_trace[k] = np.mean(err)

        mean, _ = moving_stats(err_trace, WINDOW)
        mean_db = 10 * np.log10(mean + EPS)
        it = np.arange(len(mean_db))

        plt.plot(it, mean_db, label=f"{label}", linewidth=2)

    plt.xlabel("Iteração")
    plt.ylabel("Erro médio |e|² (dB)")
    plt.title("Convergência do RLS — Ambiente CV-QKD Simulado")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()


#-----------------------------------------------------
# MAIN
#-----------------------------------------------------
if __name__ == "__main__":
    print("RLS — Benchmark + Convergência (Versão Robusta CV-QKD)\n")

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

