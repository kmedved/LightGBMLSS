import time
import numpy as np
import torch
from lightgbmlss.distributions.Gaussian import Gaussian
from lightgbmlss.distributions.Gamma import Gamma
from lightgbmlss.distributions.StudentT import StudentT
import lightgbm as lgb

np.random.seed(123)
N = 10000
X = np.random.randn(N, 5).astype(np.float32)
y = np.abs(np.random.randn(N).astype(np.float32)) + 0.1

results = {}

for dist_cls, dist_name in [(Gaussian, "Gaussian"), (Gamma, "Gamma"), (StudentT, "StudentT")]:
    dist = dist_cls()

    dtrain = lgb.Dataset(X, label=y, free_raw_data=False)
    init_score = np.ones((N, 1)) * 0.5
    init_score = np.tile(init_score, (1, dist.n_dist_param))
    dtrain.set_init_score(init_score.ravel(order="F"))

    predt = np.random.randn(N * dist.n_dist_param).astype(np.float64)
    start_values = [0.5] * dist.n_dist_param

    # Warm up
    for _ in range(2):
        target = torch.tensor(dtrain.get_label().reshape(-1, 1))
        p, loss = dist.get_params_loss(predt.copy(), target, start_values, requires_grad=True)
        weights = np.ones((N, 1))
        g, h = dist.compute_gradients_and_hessians(loss, p, weights)

    # Benchmark objective_fn (grad+hess) - the hot path called every boosting round
    times = []
    for _ in range(20):
        t0 = time.perf_counter()
        target = torch.tensor(dtrain.get_label().reshape(-1, 1))
        p, loss = dist.get_params_loss(predt.copy(), target, start_values, requires_grad=True)
        weights = np.ones((N, 1))
        g, h = dist.compute_gradients_and_hessians(loss, p, weights)
        times.append(time.perf_counter() - t0)

    results[dist_name] = times
    p50 = np.median(times) * 1000
    p95 = np.percentile(times, 95) * 1000
    print(f"{dist_name:12s}  p50={p50:7.1f}ms  p95={p95:7.1f}ms  (N={N}, {dist.n_dist_param} params)")

# Also benchmark CRPS path
dist_crps = Gaussian(loss_fn="crps")
predt_crps = np.random.randn(N * dist_crps.n_dist_param).astype(np.float64)
times_crps = []
for _ in range(10):
    t0 = time.perf_counter()
    target = torch.tensor(dtrain.get_label().reshape(-1, 1))
    p, loss = dist_crps.get_params_loss(predt_crps.copy(), target, [0.5, 0.5], requires_grad=True)
    weights = np.ones((N, 1))
    g, h = dist_crps.compute_gradients_and_hessians(loss, p, weights)
    times_crps.append(time.perf_counter() - t0)
p50 = np.median(times_crps) * 1000
p95 = np.percentile(times_crps, 95) * 1000
print(f"{'CRPS(Gauss)':12s}  p50={p50:7.1f}ms  p95={p95:7.1f}ms  (N={N})")

# Benchmark start value calculation
times_sv = []
for _ in range(5):
    t0 = time.perf_counter()
    loss, sv = Gaussian().calculate_start_values(y.reshape(-1, 1), max_iter=50)
    times_sv.append(time.perf_counter() - t0)
p50 = np.median(times_sv) * 1000
print(f"{'StartValues':12s}  p50={p50:7.1f}ms  (max_iter=50)")
