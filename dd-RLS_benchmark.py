"""
dd-RLS — Benchmark e Convergência
- Gráfico 1: Análise Temporal
- Gráfico 2: Convergência (Erro × Iteração)
"""


import numpy as np
import time
import matplotlib.pyplot as plt

# -----------------------------------------------------
# CONFIGURAÇÕES GLOBAIS (ESTABILIZADAS E OTIMIZADAS)
#-----------------------------------------------------
N_ITER = 10_000
N_MODES = 8
N_TAPS = 64
LAMBDA = 0.9999  # Lambda profundo para minimizar o ruído residual
DELTA = 1e-4     # Regularização refinada
EPS = 1e-12      # Estabilização numérica
SEED = 42

#-----------------------------------------------------
# dd-RLS original (loop) — PRESERVADO
#-----------------------------------------------------
def ddRLS_original(x, constSymb, outEq, lam, H, Sd, nModes):
    nTaps = H.shape[1]
    outEq_T = outEq.T
    decided = np.zeros_like(outEq_T)

    for k in range(nModes):
        decided[0, k] = constSymb[np.argmin(np.abs(outEq_T[0, k] - constSymb))]

    err = decided - outEq_T
    errDiag = np.diag(err[0])

    for N in range(nModes):
        indUpdModes = np.arange(nModes) + N * nModes
        u = np.conj(x[:, N]).reshape(-1, 1)
        uH = u.conj().T

        SdN = Sd[N]
        gain_den = lam + np.real(uH @ SdN @ u) + EPS
        SdN = (1 / lam) * (SdN - (SdN @ u @ uH @ SdN) / gain_den)

        inPar = u.T.repeat(nModes, axis=0)
        H[indUpdModes, :] += errDiag @ (SdN @ inPar.T).T
        Sd[N] = SdN

    return H, Sd, np.abs(err) ** 2

#-----------------------------------------------------
# Vetorizado com tensordot
#-----------------------------------------------------
def ddRLS_vectorized_tensordot(x, constSymb, outEq, lam, H, Sd, nModes):
    outEq_T = outEq.T
    dists = np.abs(outEq_T[:, :, None] - constSymb[None, None, :])
    decided = constSymb[np.argmin(dists, axis=2)]
    err = decided - outEq_T

    for N in range(nModes):
        u = np.conj(x[:, N])
        SdN = Sd[N]
        Pu = SdN @ u
        gain_den = lam + np.real(np.vdot(u, Pu)) + EPS
        SdN = (SdN - np.tensordot(Pu, Pu.conj(), axes=0) / gain_den) / lam

        indUpdModes = slice(N * nModes, (N + 1) * nModes)
        H[indUpdModes, :] += np.outer(err[0], SdN @ u)
        Sd[N] = SdN
    return H, Sd, np.abs(err) ** 2

#-----------------------------------------------------
# Vetorizado com eisum
#-----------------------------------------------------
def ddRLS_vectorized_einsum(x, constSymb, outEq, lam, H, Sd, nModes):
    outEq_T = outEq.T
    dists = np.abs(outEq_T[:, :, None] - constSymb[None, None, :])
    decided = constSymb[np.argmin(dists, axis=2)]
    err = decided - outEq_T

    for N in range(nModes):
        u = np.conj(x[:, N])
        SdN = Sd[N]
        Pu = SdN @ u
        gain_den = lam + np.real(np.vdot(u, Pu)) + EPS
        SdN = (SdN - np.einsum('i,j->ij', Pu, Pu.conj()) / gain_den) / lam

        indUpdModes = slice(N * nModes, (N + 1) * nModes)
        H[indUpdModes, :] += np.einsum('i,j->ij', err[0], SdN @ u)
        Sd[N] = SdN
    return H, Sd, np.abs(err) ** 2

#-----------------------------------------------------
# Vetorizado com broadcast
#-----------------------------------------------------
def ddRLS_vectorized_broadcast(x, constSymb, outEq, lam, H, Sd, nModes):
    outEq_T = outEq.T
    dists = np.abs(outEq_T[:, :, None] - constSymb[None, None, :])
    decided = constSymb[np.argmin(dists, axis=2)]
    err = decided - outEq_T

    for N in range(nModes):
        u = np.conj(x[:, N])
        SdN = Sd[N]
        Pu = SdN @ u
        gain_den = lam + np.real(np.vdot(u, Pu)) + EPS
        SdN = (SdN - np.outer(Pu, Pu.conj()) / gain_den) / lam

        indUpdModes = slice(N * nModes, (N + 1) * nModes)
        H[indUpdModes, :] += err[0][:, None] * (SdN @ u)[None, :]
        Sd[N] = SdN
    return H, Sd, np.abs(err) ** 2

#-----------------------------------------------------
# GRÁFICO 1: BENCHMARK TEMPORAL
#-----------------------------------------------------
def benchmark_time_only(nIter, nModes, nTaps, lam, seed):
    rng = np.random.default_rng(seed)
    methods = [
        ("ddRLS Original", ddRLS_original),
        ("ddRLS tensordot", ddRLS_vectorized_tensordot),
        ("ddRLS einsum", ddRLS_vectorized_einsum),
        ("ddRLS broadcast", ddRLS_vectorized_broadcast),
    ]

    xs = (rng.standard_normal((nIter, nTaps, nModes)) + 1j * rng.standard_normal((nIter, nTaps, nModes))) * 0.707
    outs = (rng.standard_normal((nIter, nModes, 1)) + 1j * rng.standard_normal((nIter, nModes, 1))) * 0.1
    constSymb = np.array([1+1j, 1-1j, -1+1j, -1-1j]) / np.sqrt(2)

    results = {}
    print(f"Executando Benchmark ({nIter} iterações)...")
    
    for name, func in methods:
        H = np.zeros((nModes * nModes, nTaps), dtype=np.complex128)
        Sd = np.zeros((nModes, nTaps, nTaps), dtype=np.complex128)
        for m in range(nModes): Sd[m] = np.eye(nTaps) / DELTA

        t0 = time.perf_counter()
        for k in range(nIter):
            H, Sd, _ = func(xs[k], constSymb, outs[k], lam, H, Sd, nModes)
        results[name] = (time.perf_counter() - t0) / nIter
        print(f"{name:22s}: {results[name]*1e6:8.2f} µs/it")

    plt.figure(figsize=(10, 6))
    labels = list(results.keys())
    values = [v * 1e6 for v in results.values()]
    bars = plt.bar(labels, values, color='skyblue')
    plt.ylabel("Tempo médio por iteração (µs)")
    plt.title(f"Benchmark Temporal ddRLS — {nIter:,} iterações")
    plt.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f"{val:.2f}", ha="center", va="bottom", fontweight="bold")
    plt.tight_layout()
    plt.show(block=False)

#-----------------------------------------------------
# GRÁFICO 2: CONVERGÊNCIA (MSE)
#-----------------------------------------------------
def plot_ddRLS_convergence(nIter, nModes, nTaps, lam, seed):
    rng = np.random.default_rng(seed)
    methods = [("ddRLS Original", ddRLS_original), ("ddRLS Broadcast", ddRLS_vectorized_broadcast)]
    WINDOW = 200
    constSymb = np.array([1+1j, 1-1j, -1+1j, -1-1j]) / np.sqrt(2)

    plt.figure(figsize=(10, 6))
    for label, func in methods:
        H = np.zeros((nModes * nModes, nTaps), dtype=np.complex128)
        Sd = np.zeros((nModes, nTaps, nTaps), dtype=np.complex128)
        for m in range(nModes): Sd[m] = np.eye(nTaps) / DELTA
        
        err_trace = np.zeros(nIter)
        H_true = (rng.standard_normal((nModes * nModes, nTaps)) + 1j * rng.standard_normal((nModes * nModes, nTaps))) * 0.1
        
        for k in range(nIter):
            x = (rng.standard_normal((nTaps, nModes)) + 1j * rng.standard_normal((nTaps, nModes))) * 0.707
            noise = (rng.standard_normal((nModes, 1)) + 1j * rng.standard_normal((nModes, 1))) * 0.01
            out_sim = (H_true[0:nModes, :] @ x[:, 0:1]) + noise
            H, Sd, err = func(x, constSymb, out_sim, lam, H, Sd, nModes)
            err_trace[k] = np.mean(err)

        mean_err = np.convolve(err_trace, np.ones(WINDOW)/WINDOW, mode="valid")
        plt.plot(10 * np.log10(mean_err + EPS), label=f"{label} (λ={lam})", linewidth=2)

    plt.xlabel("Iteração")
    plt.ylabel("Erro médio |e|² (dB)")
    plt.title("Estabilidade de Convergência — CV-QKD Otimizado")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()

#-----------------------------------------------------
# MAIN EXECUTION
#-----------------------------------------------------
if __name__ == "__main__":
    print("Iniciando Verificação Completa de dd-RLS Robusto\n")
    benchmark_time_only(N_ITER, N_MODES, N_TAPS, LAMBDA, SEED)
    plot_ddRLS_convergence(N_ITER, N_MODES, N_TAPS, LAMBDA, SEED)