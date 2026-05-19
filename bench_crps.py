import cProfile
import pstats
import numpy as np
import torch
from lightgbmlss.distributions.Gaussian import Gaussian
import lightgbm as lgb

np.random.seed(123)
N = 10000
X = np.random.randn(N, 5).astype(np.float32)
y = np.abs(np.random.randn(N).astype(np.float32)) + 0.1

dist = Gaussian(loss_fn="crps")
dtrain = lgb.Dataset(X, label=y, free_raw_data=False)
init_score = np.ones((N, 1)) * 0.5
init_score = np.tile(init_score, (1, dist.n_dist_param))
dtrain.set_init_score(init_score.ravel(order="F"))
predt = np.random.randn(N * dist.n_dist_param).astype(np.float64)

def run():
    for _ in range(50):
        target = torch.tensor(dtrain.get_label().reshape(-1, 1))
        p, loss = dist.get_params_loss(predt.copy(), target, [0.5, 0.5], requires_grad=True)
        weights = np.ones((N, 1))
        g, h = dist.compute_gradients_and_hessians(loss, p, weights)

cProfile.run('run()', '/tmp/crps_profile.prof')
stats = pstats.Stats('/tmp/crps_profile.prof')
stats.sort_stats('cumulative')
stats.print_stats(30)
