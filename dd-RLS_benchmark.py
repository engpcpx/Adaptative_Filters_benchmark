


# import numpy as np
# import time
# import matplotlib.pyplot as plt

# # ----------------------------
# # Configurações
# # ----------------------------
# N_ITER = 10_000
# N_MODES = 8
# N_TAPS = 64
# MU = 1e-4
# RUN_WL = True
# SEED = 42


# # -----------------------------------------------------
# # dd-RLS original (loop) — PRESERVADO
# # -----------------------------------------------------
# def ddrls_original(x, constSymb, outEq, λ, H, Sd, nModes):
#     nTaps = H.shape[1]
#     indMode = np.arange(0, nModes)
#     indTaps = np.arange(0, nTaps)

#     outEq = outEq.T
#     decided = np.zeros(outEq.shape, dtype=np.complex128)

#     for k in range(nModes):
#         indSymb = np.argmin(np.abs(outEq[0, k] - constSymb))
#         decided[0, k] = constSymb[indSymb]

#     err = decided - outEq
#     errDiag = np.diag(err[0])

#     for N in range(nModes):
#         indUpdModes = indMode + N * nModes
#         indUpdTaps = indTaps + N * nTaps

#         Sd_ = Sd[indUpdTaps, :]

#         inAdapt = np.conj(x[:, N]).reshape(-1, 1)
#         inAdaptPar = (inAdapt.T).repeat(nModes).reshape(len(x), -1).T

#         Sd_ = (1 / λ) * (
#             Sd_
#             - (Sd_ @ (inAdapt @ (np.conj(inAdapt).T)) @ Sd_)
#             / (λ + (np.conj(inAdapt).T) @ Sd_ @ inAdapt)
#         )

#         H[indUpdModes, :] += errDiag @ (Sd_ @ inAdaptPar.T).T
#         Sd[indUpdTaps, :] = Sd_

#     return H, Sd, np.abs(err) ** 2


# # -----------------------------------------------------
# # Vetorizado — tensordot
# # -----------------------------------------------------
# def ddrls_vectorized_tensordot(x, constSymb, outEq, λ, H, Sd, nModes):
#     nTaps = H.shape[1]
#     indMode = np.arange(nModes)
#     indTaps = np.arange(nTaps)

#     outEq = outEq.T
#     decided = np.zeros_like(outEq)

#     for k in range(nModes):
#         decided[0, k] = constSymb[np.argmin(np.abs(outEq[0, k] - constSymb))]

#     err = decided - outEq
#     errDiag = np.diag(err[0])

#     for N in range(nModes):
#         indUpdModes = indMode + N * nModes
#         indUpdTaps = indTaps + N * nTaps

#         Sd_ = Sd[indUpdTaps, :]

#         u = np.conj(x[:, N]).reshape(-1, 1)
#         uH = np.conj(u).T

#         gain_den = λ + uH @ Sd_ @ u
#         Sd_ = (1 / λ) * (Sd_ - (Sd_ @ (u @ uH) @ Sd_) / gain_den)

#         inAdaptPar = np.repeat(u.T, nModes, axis=0)
#         deltaH = errDiag @ (Sd_ @ inAdaptPar.T).T

#         H[indUpdModes, :] += deltaH
#         Sd[indUpdTaps, :] = Sd_

#     return H, Sd, np.abs(err) ** 2


# # -----------------------------------------------------
# # Vetorizado — einsum
# # -----------------------------------------------------
# def ddrls_vectorized_einsum(x, constSymb, outEq, λ, H, Sd, nModes):
#     nTaps = H.shape[1]
#     indMode = np.arange(nModes)
#     indTaps = np.arange(nTaps)

#     # -----------------------------
#     # Decisão por constelação
#     # -----------------------------
#     outEq = outEq.T
#     decided = np.zeros_like(outEq)

#     for k in range(nModes):
#         decided[0, k] = constSymb[np.argmin(np.abs(outEq[0, k] - constSymb))]

#     err = decided - outEq  # erro complexo

#     # -----------------------------
#     # Loop RLS por modo (INEVITÁVEL)
#     # -----------------------------
#     for N in range(nModes):
#         indUpdModes = indMode + N * nModes
#         indUpdTaps = indTaps + N * nTaps

#         Sd_ = Sd[indUpdTaps, :]              # bloco Sd
#         u = np.conj(x[:, N]).reshape(-1, 1)  # vetor de entrada
#         uH = np.conj(u).T

#         # -----------------------------
#         # Atualização RLS (Joseph form)
#         # -----------------------------
#         gain_den = λ + uH @ Sd_ @ u

#         Sd_ = (1 / λ) * (
#             Sd_
#             - np.einsum("ij,jk,kl->il", Sd_, u @ uH, Sd_) / gain_den
#         )

#         # -----------------------------
#         # Atualização dos coeficientes
#         # -----------------------------
#         Su = Sd_ @ u                        # (nTaps, 1)
#         deltaH = np.einsum("k,i->ki", err[0], Su[:, 0])

#         H[indUpdModes, :] += deltaH
#         Sd[indUpdTaps, :] = Sd_

#     return H, Sd, np.abs(err) ** 2


# # -----------------------------------------------------
# # Vetorizado rápido — broadcasting
# # -----------------------------------------------------
# def ddrls_vectorized_broadcast(x, constSymb, outEq, λ, H, Sd, nModes):
#     nTaps = H.shape[1]
#     indMode = np.arange(nModes)
#     indTaps = np.arange(nTaps)

#     outEq = outEq.T
#     decided = np.zeros_like(outEq)

#     for k in range(nModes):
#         decided[0, k] = constSymb[np.argmin(np.abs(outEq[0, k] - constSymb))]

#     err = decided - outEq

#     for N in range(nModes):
#         indUpdModes = indMode + N * nModes
#         indUpdTaps = indTaps + N * nTaps

#         Sd_ = Sd[indUpdTaps, :]

#         u = np.conj(x[:, N]).reshape(-1, 1)
#         uH = np.conj(u).T

#         gain_den = λ + uH @ Sd_ @ u
#         Sd_ = (1 / λ) * (Sd_ - (Sd_ @ (u @ uH) @ Sd_) / gain_den)

#         deltaH = err[0][:, None] * (Sd_ @ u).T
#         H[indUpdModes, :] += deltaH

#         Sd[indUpdTaps, :] = Sd_

#     return H, Sd, np.abs(err) ** 2



# # -----------------------------------------------------
# # Benchmark — Tempo (Gráfico)
# # -----------------------------------------------------
# def benchmark_time_only(nIter, nModes, nTaps, mu, runWL, seed):
#     rng = np.random.default_rng(seed)

#     methods = [
#         ("ddRLS Original", ddRLS_original),
#         ("ddRLS tensordot", ddRLS_vectorized_tensordot),
#         ("ddRLS einsum", ddRLS_vectorized_einsum),
#         ("ddRLS broadcast", ddRLS_vectorized_broadcast),
#     ]

#     xs = rng.standard_normal((nIter, nTaps, nModes)) + 1j * rng.standard_normal((nIter, nTaps, nModes))

#     outs = rng.standard_normal((nIter, nModes, 1)) + 1j * rng.standard_normal((nIter, nModes, 1))

#     R = np.array([1, 3, 5])

#     results = {}

#     for name, func in methods:
#         H = np.zeros((nModes * nModes, nTaps), dtype=np.complex128)
#         H_ = np.zeros_like(H)

#         t0 = time.perf_counter()
#         for k in range(nIter):
#             H, H_, _ = func(xs[k], R, outs[k], mu, H, H_, nModes, runWL)
#         t = time.perf_counter() - t0

#         results[name] = t / nIter
#         print(f"{name:20s}: {results[name]*1e6:8.2f} µs/it")

#     plt.figure(figsize=(10, 6))
#     labels = list(results.keys())
#     values = [v * 1e6 for v in results.values()]
#     bars = plt.bar(labels, values)

#     plt.ylabel("Tempo médio por iteração (µs)")
#     plt.title(f"Benchmark ddLMS — {nIter:,} iterações")
#     plt.grid(axis="y", alpha=0.3)

#     for bar, val in zip(bars, values):
#         plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
#                  f"{val:.2f}", ha="center", va="bottom", fontweight="bold")

#     plt.tight_layout()
#     plt.show(block=False)



# # -----------------------------------------------------
# # Benchmark — Tempo (Gráfico 1)
# # -----------------------------------------------------
# def benchmark_time_only(nIter, nModes, nTaps, mu, runWL, seed):
#     rng = np.random.default_rng(seed)

#     methods = [
#         ("ddRLS Original", ddRLS_original),
#         ("ddRLS tensordot", ddRLS_vectorized_tensordot),
#         ("ddRLS einsum", ddRLS_vectorized_einsum),
#         ("ddRLS broadcast", ddRLS_vectorized_broadcast),
#     ]

#     xs = (
#         rng.standard_normal((nIter, nTaps, nModes))
#         + 1j * rng.standard_normal((nIter, nTaps, nModes))
#     )

#     outs = (
#         rng.standard_normal((nIter, nModes, 1))
#         + 1j * rng.standard_normal((nIter, nModes, 1))
#     )

#     # -------------------------------------------------
#     # daRDE: raio desejado (REAL, dimensão correta)
#     # -------------------------------------------------
#     R = np.full(nModes, 3.0)

#     results = {}

#     for name, func in methods:
#         H = np.zeros((nModes * nModes, nTaps), dtype=np.complex128)
#         H_ = np.zeros_like(H)

#         t0 = time.perf_counter()
#         for k in range(nIter):
#             H, H_, _ = func(
#                 xs[k],
#                 R,
#                 outs[k],
#                 mu,
#                 H,
#                 H_,
#                 nModes,
#                 runWL,
#             )
#         t = time.perf_counter() - t0

#         results[name] = t / nIter
#         print(f"{name:20s}: {results[name]*1e6:8.2f} µs/it")

#     # -----------------------------
#     # Gráfico de tempo
#     # -----------------------------
#     plt.figure(figsize=(10, 6))
#     labels = list(results.keys())
#     values = [v * 1e6 for v in results.values()]

#     bars = plt.bar(labels, values)
#     plt.ylabel("Tempo médio por iteração (µs)")
#     plt.title(f"Benchmark ddRLS — {nIter:,} iterações")
#     plt.grid(axis="y", alpha=0.3)

#     for bar, val in zip(bars, values):
#         plt.text(
#             bar.get_x() + bar.get_width() / 2,
#             bar.get_height(),
#             f"{val:.2f}",
#             ha="center",
#             va="bottom",
#             fontweight="bold",
#         )

#     plt.tight_layout()
#     plt.show(block=False)


# # -----------------------------------------------------
# # Convergência — Erro × Iteração (Gráfico 2 com desvio médio padrão)
# # -----------------------------------------------------

# def plot_daRDE_convergence(nIter, nModes, nTaps, mu, runWL, seed):
#     rng = np.random.default_rng(seed)

#     methods = [
#         ("ddRLS Original", ddRLS_original),
#         ("ddRLS Broadcast", ddRLS_vectorized_broadcast),
#     ]

#     WINDOW = 200
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

#         # --------------------------------------------
#         # Conversão para dB
#         # --------------------------------------------
#         mean_db = 10 * np.log10(mean + 1e-12)
#         up_db = 10 * np.log10(mean + std + 1e-12)
#         dn_db = 10 * np.log10(np.maximum(mean - std, 1e-12))

#         # --------------------------------------------
#         # Desvio-padrão médio em dB (numérico)
#         # --------------------------------------------
#         sigma_db = np.mean(up_db - mean_db)

#         it = np.arange(len(mean_db))
#         plt.plot(
#             it,
#             mean_db,
#             label=f"{label} (σ̄ = {sigma_db:.2f} dB)",
#             linewidth=2,
#         )
#         plt.fill_between(it, dn_db, up_db, alpha=0.2)

#     plt.xlabel("Iteração")
#     plt.ylabel("Erro médio |e|² (dB)")
#     plt.title("Convergência do daRDE — Validação Numérica")
#     plt.legend()
#     plt.grid(True, linestyle="--", alpha=0.6)
#     plt.tight_layout()
#     plt.show()



# # -----------------------------------------------------
# # MAIN 
# # -----------------------------------------------------
# if __name__ == "__main__":
#     print("ddRLS — Benchmark + Convergência\n")

#     benchmark_time_only(
#         nIter=N_ITER,
#         nModes=N_MODES,
#         nTaps=N_TAPS,
#         mu=MU,
#         runWL=RUN_WL,
#         seed=SEED,
#     )

#     plot_ddRLS_convergence(
#         nIter=N_ITER,
#         nModes=N_MODES,
#         nTaps=N_TAPS,
#         mu=MU,
#         runWL=RUN_WL,
#         seed=SEED,
#     )


# import numpy as np
# import time
# import matplotlib.pyplot as plt

# # =====================================================
# # CONFIGURAÇÕES GLOBAIS (PADRÃO DO FRAMEWORK)
# # =====================================================
# N_ITER = 10_000
# N_MODES = 8
# N_TAPS = 64
# LAMBDA = 0.995
# SEED = 42


# # =====================================================
# # dd-RLS ORIGINAL (LOOP) — PRESERVADO
# # =====================================================
# def ddRLS_original(x, constSymb, outEq, lam, H, Sd, nModes):

#     nTaps = H.shape[1]

#     outEq = outEq.T
#     decided = np.zeros_like(outEq)

#     for k in range(nModes):
#         decided[0, k] = constSymb[
#             np.argmin(np.abs(outEq[0, k] - constSymb))
#         ]

#     err = decided - outEq
#     errDiag = np.diag(err[0])

#     for N in range(nModes):

#         indUpdModes = np.arange(nModes) + N * nModes

#         u = np.conj(x[:, N]).reshape(-1, 1)
#         uH = np.conj(u).T

#         SdN = Sd[N]                              # (nTaps × nTaps)

#         gain_den = lam + uH @ SdN @ u

#         SdN = (1 / lam) * (
#             SdN - (SdN @ (u @ uH) @ SdN) / gain_den
#         )

#         inPar = u.T.repeat(nModes, axis=0)       # (nModes × nTaps)
#         H[indUpdModes, :] += errDiag @ (SdN @ inPar.T).T

#         Sd[N] = SdN

#     return H, Sd, np.abs(err) ** 2


# # =====================================================
# # dd-RLS VETORIZADO — tensordot
# # =====================================================
# def ddRLS_vectorized_tensordot(x, constSymb, outEq, lam, H, Sd, nModes):

#     outEq = outEq.T
#     decided = np.zeros_like(outEq)

#     for k in range(nModes):
#         decided[0, k] = constSymb[np.argmin(np.abs(outEq[0, k] - constSymb))]

#     err = decided - outEq
#     errDiag = np.diag(err[0])

#     for N in range(nModes):

#         u = np.conj(x[:, N]).reshape(-1, 1)
#         SdN = Sd[N]

#         gain_den = lam + u.conj().T @ SdN @ u

#         SdN = (1 / lam) * (
#             SdN - np.tensordot(
#                 np.tensordot(SdN, u @ u.conj().T, axes=1),
#                 SdN, axes=1
#             ) / gain_den
#         )

#         indUpdModes = np.arange(nModes) + N * nModes
#         H[indUpdModes, :] += err[0][:, None] * (SdN @ u).T

#         Sd[N] = SdN

#     return H, Sd, np.abs(err) ** 2


# # =====================================================
# # dd-RLS VETORIZADO — einsum
# # =====================================================
# def ddRLS_vectorized_einsum(x, constSymb, outEq, lam, H, Sd, nModes):

#     outEq = outEq.T
#     decided = np.zeros_like(outEq)

#     for k in range(nModes):
#         decided[0, k] = constSymb[np.argmin(np.abs(outEq[0, k] - constSymb))]

#     err = decided - outEq

#     for N in range(nModes):

#         u = np.conj(x[:, N]).reshape(-1, 1)
#         SdN = Sd[N]

#         gain_den = lam + u.conj().T @ SdN @ u

#         SdN = (1 / lam) * (
#             SdN - np.einsum("ij,jk,kl->il", SdN, u @ u.conj().T, SdN) / gain_den
#         )

#         indUpdModes = np.arange(nModes) + N * nModes
#         H[indUpdModes, :] += err[0][:, None] * (SdN @ u).T

#         Sd[N] = SdN

#     return H, Sd, np.abs(err) ** 2


# # =====================================================
# # dd-RLS VETORIZADO RÁPIDO — broadcasting
# # =====================================================
# def ddRLS_vectorized_broadcast(x, constSymb, outEq, lam, H, Sd, nModes):

#     outEq = outEq.T
#     decided = np.zeros_like(outEq)

#     for k in range(nModes):
#         decided[0, k] = constSymb[np.argmin(np.abs(outEq[0, k] - constSymb))]

#     err = decided - outEq

#     for N in range(nModes):

#         u = np.conj(x[:, N]).reshape(-1, 1)
#         SdN = Sd[N]

#         gain_den = lam + u.conj().T @ SdN @ u
#         SdN = (1 / lam) * (SdN - (SdN @ (u @ u.conj().T) @ SdN) / gain_den)

#         indUpdModes = np.arange(nModes) + N * nModes
#         H[indUpdModes, :] += err[0][:, None] * (SdN @ u).T

#         Sd[N] = SdN

#     return H, Sd, np.abs(err) ** 2


# # =====================================================
# # BENCHMARK TEMPORAL — GRÁFICO 1
# # =====================================================
# # =====================================================
# # BENCHMARK TEMPORAL — GRÁFICO 1 (COMPLETO E CORRETO)
# # =====================================================
# def benchmark_time_only(nIter, nModes, nTaps, lam, seed):

#     rng = np.random.default_rng(seed)

#     methods = [
#         ("ddRLS Original", ddRLS_original),
#         ("ddRLS tensordot", ddRLS_vectorized_tensordot),
#         ("ddRLS einsum", ddRLS_vectorized_einsum),
#         ("ddRLS broadcast", ddRLS_vectorized_broadcast),
#     ]

#     # -------------------------------------------------
#     # Dados sintéticos (benchmark puro)
#     # -------------------------------------------------
#     xs = (
#         rng.standard_normal((nIter, nTaps, nModes))
#         + 1j * rng.standard_normal((nIter, nTaps, nModes))
#     )

#     outs = (
#         rng.standard_normal((nIter, nModes, 1))
#         + 1j * rng.standard_normal((nIter, nModes, 1))
#     )

#     # Constelação (DD)
#     R = np.full(nModes, 3.0)

#     results = {}

#     # -------------------------------------------------
#     # Loop de benchmark
#     # -------------------------------------------------
#     for name, func in methods:

#         # Coeficientes do equalizador
#         H = np.zeros((nModes * nModes, nTaps), dtype=np.complex128)

#         # -------------------------------------------------
#         # MATRIZ Sd CORRETA (MODAL)
#         # Sd.shape = (nModes, nTaps, nTaps)
#         # -------------------------------------------------
#         Sd = np.zeros((nModes, nTaps, nTaps), dtype=np.complex128)
#         for m in range(nModes):
#             Sd[m] = np.eye(nTaps) * 1e2

#         t0 = time.perf_counter()

#         for k in range(nIter):
#             H, Sd, _ = func(
#                 xs[k],
#                 R,
#                 outs[k],
#                 lam,
#                 H,
#                 Sd,
#                 nModes,
#             )

#         t = time.perf_counter() - t0

#         results[name] = t / nIter
#         print(f"{name:22s}: {results[name]*1e6:8.2f} µs/it")

#     # -------------------------------------------------
#     # Gráfico
#     # -------------------------------------------------
#     plt.figure(figsize=(10, 6))

#     labels = list(results.keys())
#     values = [v * 1e6 for v in results.values()]

#     bars = plt.bar(labels, values)

#     plt.ylabel("Tempo médio por iteração (µs)")
#     plt.title(f"Benchmark Temporal ddRLS — {nIter:,} iterações")
#     plt.grid(axis="y", alpha=0.3)

#     for bar, val in zip(bars, values):
#         plt.text(
#             bar.get_x() + bar.get_width() / 2,
#             bar.get_height(),
#             f"{val:.2f}",
#             ha="center",
#             va="bottom",
#             fontweight="bold",
#         )

#     plt.tight_layout()
#     plt.show(block=False)



# # =====================================================
# # CONVERGÊNCIA — ERRO × ITERAÇÃO (GRÁFICO 2)
# # =====================================================
# def plot_ddRLS_convergence(nIter, nModes, nTaps, lam, seed):

#     rng = np.random.default_rng(seed)

#     methods = [
#         ("ddRLS Original", ddRLS_original),
#         ("ddRLS Broadcast", ddRLS_vectorized_broadcast),
#     ]

#     WINDOW = 200
#     R = np.full(nModes, 3.0)

#     def moving_stats(x, w):
#         mean = np.convolve(x, np.ones(w)/w, mode="valid")
#         sq = np.convolve(x**2, np.ones(w)/w, mode="valid")
#         std = np.sqrt(np.maximum(sq - mean**2, 0))
#         return mean, std

#     plt.figure(figsize=(10, 6))

#     for label, func in methods:

#         H = np.zeros((nModes * nModes, nTaps), dtype=np.complex128)
#         Sd = np.eye(nTaps * nModes, dtype=np.complex128) * 1e2
#         err_trace = np.zeros(nIter)

#         for k in range(nIter):

#             x = rng.standard_normal((nTaps, nModes)) + 1j * rng.standard_normal((nTaps, nModes))
#             outEq = rng.standard_normal((nModes, 1)) + 1j * rng.standard_normal((nModes, 1))

#             H, Sd, err = func(x, R, outEq, lam, H, Sd, nModes)
#             err_trace[k] = np.sum(err)

#         mean, std = moving_stats(err_trace, WINDOW)

#         mean_db = 10 * np.log10(mean + 1e-12)
#         up_db = 10 * np.log10(mean + std + 1e-12)
#         dn_db = 10 * np.log10(np.maximum(mean - std, 1e-12))

#         sigma_db = np.mean(up_db - mean_db)

#         it = np.arange(len(mean_db))
#         plt.plot(it, mean_db, label=f"{label} (σ̄ = {sigma_db:.2f} dB)", linewidth=2)
#         plt.fill_between(it, dn_db, up_db, alpha=0.2)

#     plt.xlabel("Iteração")
#     plt.ylabel("Erro médio |e|² (dB)")
#     plt.title("Convergência do ddRLS — Validação Numérica")
#     plt.legend()
#     plt.grid(True, linestyle="--", alpha=0.6)
#     plt.tight_layout()
#     plt.show()


# # =====================================================
# # MAIN
# # =====================================================
# if __name__ == "__main__":

#     print("ddRLS — Benchmark + Convergência\n")

#     benchmark_time_only(
#         nIter=N_ITER,
#         nModes=N_MODES,
#         nTaps=N_TAPS,
#         lam=LAMBDA,
#         seed=SEED,
#     )

#     plot_ddRLS_convergence(
#         nIter=N_ITER,
#         nModes=N_MODES,
#         nTaps=N_TAPS,
#         lam=LAMBDA,
#         seed=SEED,
#     )

import numpy as np
import time
import matplotlib.pyplot as plt

# =====================================================
# CONFIGURAÇÕES GLOBAIS (ESTABILIZADAS E OTIMIZADAS)
# =====================================================
N_ITER = 10_000
N_MODES = 8
N_TAPS = 64
LAMBDA = 0.9999  # Lambda profundo para minimizar o ruído residual
DELTA = 1e-4     # Regularização refinada
EPS = 1e-12      # Estabilização numérica
SEED = 42

# =====================================================
# dd-RLS ORIGINAL (REFERÊNCIA MATEMÁTICA)
# =====================================================
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

# =====================================================
# dd-RLS VETORIZADO — TENSORDOT
# =====================================================
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

# =====================================================
# dd-RLS VETORIZADO — EINSUM
# =====================================================
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

# =====================================================
# dd-RLS VETORIZADO — BROADCAST (MAIS RÁPIDO)
# =====================================================
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

# =====================================================
# GRÁFICO 1: BENCHMARK TEMPORAL
# =====================================================
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

# =====================================================
# GRÁFICO 2: CONVERGÊNCIA (MSE)
# =====================================================
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

# =====================================================
# MAIN EXECUTION
# =====================================================
if __name__ == "__main__":
    print("Iniciando Verificação Completa de dd-RLS Robusto\n")
    benchmark_time_only(N_ITER, N_MODES, N_TAPS, LAMBDA, SEED)
    plot_ddRLS_convergence(N_ITER, N_MODES, N_TAPS, LAMBDA, SEED)