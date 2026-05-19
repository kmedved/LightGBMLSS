from ..utils import BaseTestClass
import numpy as np
import torch


class TestClass(BaseTestClass):
    def test_crps_score(self, flow_class):
        # Create data for testing
        torch.manual_seed(123)
        n_obs = 10
        n_samples = 20
        y = torch.rand(n_obs, 1)
        yhat_dist = torch.rand(n_samples, n_obs)

        # Call the function
        loss = flow_class.dist.crps_score(y, yhat_dist)

        # Assertions
        assert isinstance(loss, torch.Tensor)
        assert not torch.isnan(loss).any()
        assert not torch.isinf(loss).any()
        assert loss.shape == y.shape

    def test_crps_score_single_observation_accepts_flat_samples(self, flow_class):
        y = torch.tensor([[0.5]])
        yhat_dist = torch.tensor([0.1, 0.4, 0.8])

        loss = flow_class.dist.crps_score(y, yhat_dist)
        expected = flow_class.dist.crps_score(y, yhat_dist.reshape(-1, 1))

        assert loss.shape == y.shape
        torch.testing.assert_close(loss, expected)

    def test_get_params_loss_crps_single_observation(self, flow_class):
        predt = np.full((1, flow_class.dist.n_dist_param), 0.5)
        target = torch.tensor([[0.5]])
        start_values = np.full(flow_class.dist.n_dist_param, 0.5)
        weights = np.ones((1, 1))

        flow_class.dist.loss_fn = "crps"

        params, loss = flow_class.dist.get_params_loss(predt, target, start_values, requires_grad=True)
        grad, hess = flow_class.dist.compute_gradients_and_hessians(loss, params, weights)

        assert loss.ndim == 0
        assert not torch.isnan(loss).any()
        assert grad.shape == (flow_class.dist.n_dist_param,)
        assert hess.shape == (flow_class.dist.n_dist_param,)
        assert np.isfinite(grad).all()
        assert np.isfinite(hess).all()
