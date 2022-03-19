import torch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition import UpperConfidenceBound
from botorch.optim import optimize_acqf



# 0. data 
train_X = torch.rand(10, 2)
Y = 1 - (train_X - 0.5).norm(dim=-1, keepdim=True)  # explicit output dimension
Y += 0.1 * torch.rand_like(Y)
train_Y = (Y - Y.mean()) / Y.std()
# print(train_X.shape, train_Y.shape, train_X, train_Y)
print(train_X.shape, train_Y.shape)


# 1. Fit a Gaussian Process model to data
gp = SingleTaskGP(train_X, train_Y)
mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
fit_gpytorch_model(mll)


# 2. Construct an acquisition function
UCB = UpperConfidenceBound(gp, beta=0.1) 


# 3. Optimize the acquisition function 
bounds = torch.stack([torch.zeros(2), torch.ones(2)])
candidate, acq_value = optimize_acqf(
    UCB, bounds=bounds, q=1, num_restarts=5, raw_samples=20,
)

print(candidate.shape, acq_value.shape)

