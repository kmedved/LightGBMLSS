import torch
import numpy as np

def crps_score_original(y, yhat_dist):
    n_samples = yhat_dist.shape[0]
    yhat_dist_sorted, _ = torch.sort(yhat_dist, 0)
    y_cdf = torch.zeros_like(y)
    yhat_cdf = torch.zeros_like(y)
    yhat_prev = torch.zeros_like(y)
    crps = torch.zeros_like(y)
    for yhat in yhat_dist_sorted:
        yhat = yhat.reshape(-1, 1)
        flag = (y_cdf == 0) * (y < yhat)
        crps += flag * ((y - yhat_prev) * yhat_cdf ** 2)
        crps += flag * ((yhat - y) * (yhat_cdf - 1) ** 2)
        crps += (~flag) * ((yhat - yhat_prev) * (yhat_cdf - y_cdf) ** 2)
        y_cdf += flag
        yhat_cdf += 1 / n_samples
        yhat_prev = yhat
    flag = (y_cdf == 0)
    crps += flag * (y - yhat)
    return crps

def crps_score_vectorized(y, yhat_dist):
    n_samples = yhat_dist.shape[0]
    # Sort forecasts: (n_samples, n_obs)
    yhat_sorted, _ = torch.sort(yhat_dist, 0)
    # Reshape y for broadcasting: (1, n_obs, 1) vs (n_samples, n_obs)
    y_flat = y.squeeze(-1)  # (n_obs,)

    # Empirical CDF weights: w_i = (2*i - 1) / (2*n) for i=1..n
    # CRPS = (1/n) * sum_i |x_i - y| - (1/(2*n^2)) * sum_i sum_j |x_i - x_j|
    # But that's O(n^2). Use the equivalent weighted form:
    # CRPS = (2/n^2) * sum_i (yhat_sorted_i - y) * (n * I(yhat_sorted_i >= y) - i + 0.5)
    # Or the compact PWM form:
    # CRPS = (1/n)*sum|x_i - y| - (1/n^2)*sum_i<j (x_j - x_i)

    # Energy form (simplest vectorized):
    # CRPS = E|X-y| - 0.5*E|X-X'|
    # = (1/n)*sum|x_i - y| - (1/(2n^2))*sum_ij|x_i - x_j|

    # Term 1: (1/n) * sum|x_i - y|
    term1 = torch.mean(torch.abs(yhat_sorted - y_flat.unsqueeze(0)), dim=0)

    # Term 2: (1/(2n^2)) * sum_ij |x_i - x_j|
    # For sorted values: sum_ij |x_i - x_j| = (2/n)*sum_i (2i - n - 1)*x_i
    idx = torch.arange(1, n_samples + 1, dtype=yhat_sorted.dtype, device=yhat_sorted.device)
    weights = (2 * idx - n_samples - 1).unsqueeze(1)  # (n_samples, 1)
    term2 = torch.sum(weights * yhat_sorted, dim=0) / (n_samples * n_samples)

    crps = (term1 - term2).unsqueeze(-1)  # (n_obs, 1)
    return crps

# Test correctness
torch.manual_seed(42)
N = 5000
n_samples = 30
y = torch.randn(N, 1)
yhat = torch.randn(n_samples, N)

crps_orig = crps_score_original(y, yhat)
crps_vec = crps_score_vectorized(y, yhat)

print(f"Original  sum: {crps_orig.sum().item():.6f}")
print(f"Vectorized sum: {crps_vec.sum().item():.6f}")
print(f"Max abs diff:  {(crps_orig - crps_vec).abs().max().item():.8f}")
print(f"Mean abs diff: {(crps_orig - crps_vec).abs().mean().item():.8f}")
print(f"Relative diff: {((crps_orig - crps_vec).abs() / (crps_orig.abs() + 1e-10)).mean().item():.8f}")

# Check that gradients flow
yhat_grad = torch.randn(n_samples, N, requires_grad=True)
crps_v = crps_score_vectorized(y, yhat_grad)
crps_v.sum().backward()
print(f"Gradient flows: {yhat_grad.grad is not None}")
print(f"Grad norm:      {yhat_grad.grad.norm().item():.4f}")

# Speed comparison
import time

times_orig = []
for _ in range(20):
    t0 = time.perf_counter()
    crps_score_original(y, yhat)
    times_orig.append(time.perf_counter() - t0)

times_vec = []
for _ in range(20):
    t0 = time.perf_counter()
    crps_score_vectorized(y, yhat)
    times_vec.append(time.perf_counter() - t0)

print(f"\nOriginal   p50: {np.median(times_orig)*1000:.2f}ms")
print(f"Vectorized p50: {np.median(times_vec)*1000:.2f}ms")
print(f"Speedup:        {np.median(times_orig)/np.median(times_vec):.1f}x")
