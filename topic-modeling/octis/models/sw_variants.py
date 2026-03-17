
import torch
import numpy as np
from power_spherical import PowerSpherical

def one_dimensional_Wasserstein_prod(X,Y,theta,p):
    X_prod = torch.matmul(X, theta.transpose(0, 1))
    Y_prod = torch.matmul(Y, theta.transpose(0, 1))
    X_prod = X_prod.view(X_prod.shape[0], -1)
    Y_prod = Y_prod.view(Y_prod.shape[0], -1)
    wasserstein_distance = torch.abs(
        (
                torch.sort(X_prod, dim=0)[0]
                - torch.sort(Y_prod, dim=0)[0]
        )
    )
    wasserstein_distance = torch.sum(torch.pow(wasserstein_distance, p), dim=0,keepdim=True)
    return wasserstein_distance

def RPSW(X, Y, L=10, p=2, device='cuda', kappa=50):
    dim = X.size(1)
    theta = (X.detach()[np.random.choice(X.shape[0], L, replace=True)] - Y.detach()[np.random.choice(Y.shape[0], L, replace=True)])
    theta = theta / torch.sqrt(torch.sum(theta ** 2, dim=1, keepdim=True))
    ps = PowerSpherical(
        loc=theta,
        scale=torch.full((theta.shape[0],), kappa, device=device),
    )
    theta = ps.rsample()
    sw = one_dimensional_Wasserstein_prod(X, Y, theta, p=p).mean()
    return torch.pow(sw, 1. / p)
def EBRPSW(X, Y, L=10, p=2, device='cuda', kappa=1):
    dim = X.size(1)
    theta = (X.detach()[np.random.choice(X.shape[0], L, replace=True)] - Y.detach()[np.random.choice(Y.shape[0], L, replace=True)])
    theta = theta / torch.sqrt(torch.sum(theta ** 2, dim=1, keepdim=True))
    ps = PowerSpherical(
        loc=theta,
        scale=torch.full((theta.shape[0],), kappa, device=device),
    )
    theta = ps.rsample()
    wasserstein_distances = one_dimensional_Wasserstein_prod(X, Y, theta, p=p)
    wasserstein_distances = wasserstein_distances.view(1, L)
    weights = torch.softmax(wasserstein_distances, dim=1)
    sw = torch.sum(weights * wasserstein_distances, dim=1).mean()
    return torch.pow(sw, 1. / p)
def DSW(X, Y, L=5, kappa=10, p=2, s_lr=0.1, n_lr=2, device='cuda'):
    dim = X.size(1)
    epsilon = torch.randn((1, dim), device=device, requires_grad=True)
    epsilon.data = epsilon.data / torch.sqrt(torch.sum(epsilon.data ** 2, dim=1, keepdim=True))
    optimizer = torch.optim.SGD([epsilon], lr=s_lr)
    X_detach = X.detach()
    Y_detach = Y.detach()
    for _ in range(n_lr - 1):
        vmf = PowerSpherical(epsilon, torch.full((1,), kappa, device=device))
        theta = vmf.rsample((L,)).view(L, -1)
        negative_sw = -torch.pow(one_dimensional_Wasserstein_prod(X_detach, Y_detach, theta, p=p).mean(), 1. / p)
        optimizer.zero_grad()
        negative_sw.backward()
        optimizer.step()
        epsilon.data = epsilon.data / torch.sqrt(torch.sum(epsilon.data ** 2, dim=1, keepdim=True))
    vmf = PowerSpherical(epsilon, torch.full((1,), kappa, device=device))
    theta = vmf.rsample((L,)).view(L, -1)
    sw = one_dimensional_Wasserstein_prod(X, Y, theta, p=p).mean()
    return torch.pow(sw, 1. / p)
def rand_projections(dim, num_projections=1000,device='cuda'):
    projections = torch.randn((num_projections, dim),device=device)
    projections = projections / torch.sqrt(torch.sum(projections ** 2, dim=1, keepdim=True))
    return projections
def SW(X, Y, L=10, p=2, device='cuda'):
    dim = X.size(1)
    theta = rand_projections(dim, L,device)
    sw=one_dimensional_Wasserstein_prod(X,Y,theta,p=p).mean()
    return  torch.pow(sw,1./p)