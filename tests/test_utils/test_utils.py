import pytest
import torch
from joblib import Parallel, delayed
from lightgbmlss import utils


def get_response_fn():
    functions_list = [fn for fn in dir(utils) if "_fn" in fn]

    func_list = []
    for func_name in functions_list:
        func_list.append(getattr(utils, func_name))

    return func_list


class TestClass:
    @pytest.fixture(params=get_response_fn())
    def response_fn(self, request):
        return request.param

    def test_response_fn(self, response_fn):
        # Create Data for testing
        predt = torch.tensor([1.0, 2.0, 3.0, 4.0]).reshape(-1,1)

        # Call the function
        predt_transformed = response_fn(predt)

        # Assertions
        assert isinstance(predt_transformed, torch.Tensor)
        assert not torch.isnan(predt_transformed).any()
        assert not torch.isinf(predt_transformed).any()

    def test_local_torch_seed_is_thread_safe(self):
        def seeded_draw():
            with utils.local_torch_seed(123):
                return torch.rand(5)

        with utils.local_torch_seed(123):
            expected = torch.rand(5)

        torch.manual_seed(2024)
        rng_state = torch.random.get_rng_state()
        draws = Parallel(n_jobs=4, prefer="threads")(delayed(seeded_draw)() for _ in range(16))

        for draw in draws:
            torch.testing.assert_close(draw, expected)
        assert torch.equal(torch.random.get_rng_state(), rng_state)
