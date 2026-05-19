from ..utils import BaseTestClass
import numpy as np
import pandas as pd
import lightgbm as lgb
import torch

from lightgbmlss.distributions.Gaussian import Gaussian
from lightgbmlss.distributions.Expectile import Expectile
from lightgbmlss.distributions.Mixture import Mixture
from lightgbmlss.distributions.SplineFlow import SplineFlow


class DummyBooster:
    def __init__(self, raw_score):
        self.raw_score = raw_score

    def predict(self, data, raw_score=True):
        return self.raw_score


class TestClass(BaseTestClass):
    def _assert_quantiles_match_samples(self, dist):
        n_obs = 5
        n_samples = 101
        quantiles = [0.1, 0.5, 0.9]
        X_dta = pd.DataFrame(np.linspace(0.0, 1.0, n_obs).reshape(-1, 1))
        raw_score = np.linspace(-0.2, 0.2, n_obs * dist.n_dist_param).reshape(n_obs, dist.n_dist_param)
        start_values = np.full(dist.n_dist_param, 0.5)
        booster = DummyBooster(raw_score)

        pred_samples = dist.predict_dist(
            booster, X_dta, start_values, "samples", n_samples=n_samples, quantiles=quantiles, seed=321
        )
        pred_quantiles = dist.predict_dist(
            booster, X_dta, start_values, "quantiles", n_samples=n_samples, quantiles=quantiles, seed=321
        )

        expected = np.quantile(pred_samples.values, quantiles, axis=1).T
        np.testing.assert_allclose(pred_quantiles.values, expected)

    def test_quantiles_match_seeded_samples_without_intermediate_dataframe(self):
        self._assert_quantiles_match_samples(Gaussian())
        self._assert_quantiles_match_samples(Mixture(Gaussian()))
        self._assert_quantiles_match_samples(
            SplineFlow(target_support="real", count_bins=2, bound=2.0, order="linear")
        )

    def test_seeded_sampling_does_not_mutate_global_torch_rng_state(self):
        predt_params = pd.DataFrame({"loc": [0.0, 1.0], "scale": [1.0, 2.0]})
        dist = Gaussian()

        torch.manual_seed(2024)
        rng_state = torch.random.get_rng_state()
        dist.draw_samples(predt_params=predt_params, n_samples=10, seed=123)

        assert torch.equal(torch.random.get_rng_state(), rng_state)

    def test_expectile_samples_remain_unavailable(self):
        dist = Expectile(expectiles=[0.1, 0.5, 0.9])
        n_obs = 3
        X_dta = pd.DataFrame(np.linspace(0.0, 1.0, n_obs).reshape(-1, 1))
        raw_score = np.zeros((n_obs, dist.n_dist_param))
        start_values = np.zeros(dist.n_dist_param)

        pred_samples = dist.predict_dist(
            DummyBooster(raw_score), X_dta, start_values, pred_type="samples"
        )

        assert pred_samples is None

    ####################################################################################################################
    # Univariate Distribution
    ####################################################################################################################
    def test_predict_dist_univariate(self, dist_class, pred_type):
        if dist_class.dist.univariate and not hasattr(dist_class.dist, "base_dist"):
            # Create data for testing
            np.random.seed(123)
            X_dta = pd.DataFrame(np.random.rand(100).reshape(-1, 1))
            y_dta = np.random.rand(100)
            dtrain = lgb.Dataset(X_dta, label=y_dta)

            # Train the model
            params = {"eta": 0.01}
            dist_class.train(params, dtrain, num_boost_round=2)

            # Call the function
            if dist_class.dist.tau is not None and pred_type in ["quantiles", "samples"]:
                pred_type = "parameters"
            predt_df = dist_class.dist.predict_dist(dist_class.booster,
                                                    X_dta,
                                                    dist_class.start_values,
                                                    pred_type,
                                                    n_samples=100,
                                                    quantiles=[0.1, 0.5, 0.9]
                                                    )

            # Assertions
            assert isinstance(predt_df, pd.DataFrame)
            assert not predt_df.isna().any().any()
            assert not np.isinf(predt_df).any().any()
            if pred_type == "parameters" or pred_type == "expectiles":
                assert predt_df.shape[1] == dist_class.dist.n_dist_param
            if dist_class.dist.tau is None:
                if pred_type == "samples":
                    assert predt_df.shape[1] == 100
                elif pred_type == "quantiles":
                    assert predt_df.shape[1] == 3

    ####################################################################################################################
    # Normalizing Flow
    ####################################################################################################################
    def test_predict_dist_flow(self, flow_class, pred_type):
        # Create data for testing
        np.random.seed(123)
        X_dta = pd.DataFrame(np.random.rand(100).reshape(-1, 1))
        y_dta = np.random.rand(100)
        dtrain = lgb.Dataset(X_dta, label=y_dta)

        # Train the model
        params = {"eta": 0.01}
        flow_class.train(params, dtrain, num_boost_round=2)

        # Call the function
        if pred_type in ["expectiles"]:
            pred_type = "parameters"
        predt_df = flow_class.dist.predict_dist(flow_class.booster,
                                                X_dta,
                                                flow_class.start_values,
                                                pred_type,
                                                n_samples=100,
                                                quantiles=[0.1, 0.5, 0.9]
                                                )

        # Assertions
        assert isinstance(predt_df, pd.DataFrame)
        assert not predt_df.isna().any().any()
        assert not np.isinf(predt_df).any().any()
        if pred_type == "parameters" or pred_type == "expectiles":
            assert predt_df.shape[1] == flow_class.dist.n_dist_param
        if pred_type == "samples":
            assert predt_df.shape[1] == 100
        elif pred_type == "quantiles":
            assert predt_df.shape[1] == 3

    ####################################################################################################################
    # Mixture Distributions
    ####################################################################################################################
    def test_predict_dist_mixture(self, mixture_class, pred_type):
        # Create data for testing
        np.random.seed(123)
        X_dta = np.random.rand(100).reshape(-1, 1)
        y_dta = np.random.rand(100)
        dtrain = lgb.Dataset(X_dta, label=y_dta)

        # Train the model
        params = {"eta": 0.01}
        mixture_class.train(params, dtrain, num_boost_round=2)

        # Call the function
        if pred_type in ["expectiles"]:
            pred_type = "parameters"
        predt_df = mixture_class.dist.predict_dist(mixture_class.booster,
                                                   X_dta,
                                                   mixture_class.start_values,
                                                   pred_type,
                                                   n_samples=100,
                                                   quantiles=[0.1, 0.5, 0.9]
                                                   )

        # Assertions
        assert isinstance(predt_df, pd.DataFrame)
        assert not predt_df.isna().any().any()
        assert not np.isinf(predt_df).any().any()
        if pred_type == "parameters" or pred_type == "expectiles":
            assert predt_df.shape[1] == mixture_class.dist.n_dist_param
        if pred_type == "samples":
            assert predt_df.shape[1] == 100
        elif pred_type == "quantiles":
            assert predt_df.shape[1] == 3
