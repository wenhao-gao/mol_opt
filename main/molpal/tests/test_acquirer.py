import string
import uuid

import numpy as np
import pytest

from molpal.acquirer import Acquirer

@pytest.fixture(
    params=[list(string.printable), list(range(50)), list(str(uuid.uuid4())+str(uuid.uuid4()))]
)
def xs(request):
    return request.param

@pytest.fixture
def Y_mean(xs):
    return np.random.normal(len(xs), size=len(xs))

@pytest.fixture
def Y_var(xs):
    return np.random.normal(size=len(xs)) / 10

@pytest.fixture
def init_size():
    return 10

@pytest.fixture
def batch_sizes():
    return [10]

@pytest.fixture(params=[None, 42])
def seed(request):
    return request.param

@pytest.fixture
def acq(xs, init_size, batch_sizes, seed):
    return Acquirer(len(xs), init_size, batch_sizes, 'greedy', seed=seed)

@pytest.fixture(params=[0., 0.1, 0.5, 1.])
def epsilon(request):
    return request.param

def test_acquire_initial(acq, xs):
    xs_0 = acq.acquire_initial(xs)

    assert len(xs_0) == acq.init_size

def test_acquire_initial_reacquire(acq, xs):
    xs_0 = acq.acquire_initial(xs)
    xs_1 = acq.acquire_initial(xs)

    assert xs_0 != xs_1

def test_acquire_initial_reacquire_seeded(acq, xs, seed):
    if seed is None:
        return

    xs_0 = acq.acquire_initial(xs)
    acq.reset()
    xs_1 = acq.acquire_initial(xs)

    assert xs_0 == xs_1

def test_acquire_batch_unexplored(acq, xs, Y_mean, Y_var):
    init_xs = acq.acquire_initial(xs)
    explored = {x: 0. for x in init_xs}

    batch_xs = acq.acquire_batch(xs, Y_mean, Y_var, explored)

    assert len(batch_xs) == acq.batch_sizes[0]
    assert set(init_xs) != set(batch_xs)

def test_acquire_batch_top_m(acq, xs, Y_mean, Y_var):
    batch_xs = acq.acquire_batch(xs, Y_mean, Y_var, {})

    top_m_idxs = np.argsort(Y_mean)[:-1 - acq.batch_sizes[0]:-1]
    top_m_xs = np.array(xs)[top_m_idxs]

    assert len(batch_xs) == len(top_m_xs)
    assert set(batch_xs) == set(top_m_xs)

def test_aquire_batch_epsilon(acq, xs, Y_mean, Y_var, epsilon):
    """this test may randomly fail but the chances of that are low.
    repeated failures indicate problems"""
    acq.epsilon = epsilon
    batch_xs_0 = acq.acquire_batch(xs, Y_mean, Y_var, {})
    batch_xs_1 = acq.acquire_batch(xs, Y_mean, Y_var, {})

    top_m_idxs = np.argsort(Y_mean)[:-1 - acq.batch_sizes[0]:-1]
    top_m_xs = np.array(xs)[top_m_idxs]

    assert len(batch_xs_0) == len(top_m_xs)
    if epsilon == 0.:
        assert batch_xs_0 == batch_xs_1
    else:
        assert batch_xs_0 != batch_xs_1
#     def test_acquire_batch_epsilon(self):
#         """There is roughly a  1-in-5*10^6 (= nCr(26, 10)) chance that a random
#         batch is the same as the calculated top-m batch, causing this test to
#         report a failure. There is a roughly 1-in-2.5E13 chance this test
#         fails twice in arow. If that happens, it is safe to assume that the
#         test is genuinely failing."""
#         acq = Acquirer(
#             size=len(self.xs), init_size=self.init_size,
#             batch_size=self.batch_size, metric='greedy', epsilon=1., seed=0
#         )
#         explored = {}

#         batch_xs = acq.acquire_batch(
#             self.xs, self.y_means, self.y_vars, explored=explored
#         )
#         self.assertNotEqual(set(batch_xs),
#                          set(self.xs[:self.batch_size]))

# if __name__ == "__main__":
#     unittest.main()