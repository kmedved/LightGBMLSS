from ..utils import BaseTestClass
import numpy as np
import torch

from lightgbmlss.distributions.Gaussian import Gaussian


class TestClass(BaseTestClass):
    def test_crps_score(self, dist_class_crps):
        # Create data for testing
        torch.manual_seed(123)
        n_obs = 10
        n_samples = 20
        y = torch.rand(n_obs, 1)
        yhat_dist = torch.rand(n_samples, n_obs)

        # Call the function
        loss = dist_class_crps.dist.crps_score(y, yhat_dist)

        # Assertions
        assert isinstance(loss, torch.Tensor)
        assert not torch.isnan(loss).any()
        assert not torch.isinf(loss).any()
        assert loss.shape == y.shape

    def test_crps_score_single_observation_accepts_flat_samples(self, dist_class_crps):
        y = torch.tensor([[0.5]])
        yhat_dist = torch.tensor([0.1, 0.4, 0.8])

        loss = dist_class_crps.dist.crps_score(y, yhat_dist)
        expected = dist_class_crps.dist.crps_score(y, yhat_dist.reshape(-1, 1))

        assert loss.shape == y.shape
        torch.testing.assert_close(loss, expected.to(loss.dtype), rtol=1e-6, atol=1e-6)

    def test_gaussian_crps_closed_form_preserves_torch_rng_state(self):
        dist = Gaussian(loss_fn="crps")
        predt = np.array([0.0, 1.0, 0.0, -0.2])
        target = torch.tensor([[0.25], [1.5]])
        start_values = [0.0, 0.0]

        torch.manual_seed(2024)
        rng_state = torch.random.get_rng_state()
        predt_params, loss = dist.get_params_loss(predt, target, start_values, requires_grad=True)

        loc = torch.tensor([[0.0], [1.0]])
        scale = torch.exp(torch.tensor([[0.0], [-0.2]]))
        expected = torch.nansum(dist.crps_loss([loc, scale], target))

        assert torch.equal(torch.random.get_rng_state(), rng_state)
        torch.testing.assert_close(loss, expected.to(loss.dtype), rtol=1e-6, atol=1e-6)

        grad, hess = dist.compute_gradients_and_hessians(loss, predt_params, np.ones((2, 1)))
        assert grad.shape == (dist.n_dist_param * 2,)
        assert hess.shape == (dist.n_dist_param * 2,)
        assert np.isfinite(grad).all()
        assert np.isfinite(hess).all()
