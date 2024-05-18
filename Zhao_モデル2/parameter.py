import torch
import os
Folder = os.getcwd()
Dt = "実験結果\\"

T = 500
N = 2
epoch = 4
beta_val = 0.1
h = 20/T
lr = 1e-4
lam = 3.0
datasize = 16
Batch = 4
x0 = torch.tensor([[12], [10]], dtype=torch.double)
sita = 3
phi = 0.5
rho = 0.05
wmax = 0.0001
wmin = -0.0001
K_s = torch.tensor([[0, -1]], dtype=torch.double)
evolution = 4


def x1_dot(x1, x2, u):
    return -x1+x2**2


def x2_dot(x1, x2, u):
    return u
