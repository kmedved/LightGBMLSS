import time, numpy as np, torch
from lightgbmlss.distributions.Gaussian import Gaussian
from lightgbmlss.model import LightGBMLSS
import lightgbm as lgb

np.random.seed(123)
N = 10000
X = np.random.randn(N, 5).astype(np.float32)
y = np.random.randn(N).astype(np.float32)

dist = Gaussian()
model = LightGBMLSS(dist)
dtrain = lgb.Dataset(X, label=y, free_raw_data=False)
params = {"learning_rate": 0.1, "max_depth": 3}
model.train(params, dtrain, num_boost_round=10)

# Benchmark predict with pred_type="parameters" (should skip sampling now)
times = []
for _ in range(20):
    t0 = time.perf_counter()
    model.predict(X, pred_type="parameters")
    times.append(time.perf_counter() - t0)
print(f"predict(parameters)  p50={np.median(times)*1000:.1f}ms")

# Compare with pred_type="samples" (must sample)
times = []
for _ in range(20):
    t0 = time.perf_counter()
    model.predict(X, pred_type="samples", n_samples=1000)
    times.append(time.perf_counter() - t0)
print(f"predict(samples)     p50={np.median(times)*1000:.1f}ms")
