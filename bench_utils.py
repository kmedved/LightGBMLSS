import time
import torch
import numpy as np

# Simulate hot-path response function calls
N = 10000
predt = torch.randn(N, 1)

# Current: creates torch.tensor(1e-06) every call
def exp_fn_old(predt):
    from lightgbmlss.utils import nan_to_num
    return torch.exp(nan_to_num(predt)) + torch.tensor(1e-06, dtype=predt.dtype)

# Optimized: use scalar addition (no tensor allocation)
def exp_fn_new(predt):
    from lightgbmlss.utils import nan_to_num
    return torch.exp(nan_to_num(predt)) + 1e-06

# Also profile nan_to_num: calls nanmean 3 times (same value each time)
def nan_to_num_old(predt):
    return torch.nan_to_num(predt,
                            nan=float(torch.nanmean(predt)),
                            posinf=float(torch.nanmean(predt)),
                            neginf=float(torch.nanmean(predt)))

def nan_to_num_new(predt):
    fill = float(torch.nanmean(predt))
    return torch.nan_to_num(predt, nan=fill, posinf=fill, neginf=fill)

# Benchmark
for name, fn in [("exp_fn_old", exp_fn_old), ("exp_fn_new", exp_fn_new)]:
    times = []
    for _ in range(200):
        t0 = time.perf_counter()
        fn(predt)
        times.append(time.perf_counter() - t0)
    print(f"{name:20s}  p50={np.median(times)*1e6:.0f}us")

for name, fn in [("nan_to_num_old", nan_to_num_old), ("nan_to_num_new", nan_to_num_new)]:
    times = []
    for _ in range(200):
        t0 = time.perf_counter()
        fn(predt)
        times.append(time.perf_counter() - t0)
    print(f"{name:20s}  p50={np.median(times)*1e6:.0f}us")

# Verify identity
print(f"\nexp match: {torch.allclose(exp_fn_old(predt), exp_fn_new(predt))}")
print(f"nan match: {torch.allclose(nan_to_num_old(predt), nan_to_num_new(predt))}")

# Count how many times response fns are called in a single objective_fn
# For a 2-param dist: 2 calls in get_params_loss (line 282) + 2 in objective_fn = 4 per round
# For 100 boosting rounds = 400 calls to response fns
# For hyper_opt with 100 trials × 10 folds × 500 rounds = 500,000 calls
print("\nResponse fn calls per hyper_opt: ~500,000 (100 trials × 10 folds × 500 rounds × 1 per param)")
