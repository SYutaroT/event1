import torch
import os

T = 100
N = 2
Folder = os.getcwd()
Dt = "\\実験結果\\"


epoch = 1
beta_val = 0.1
h = 10/T
lr = 0.01
lam = 3
datasize = 1600
Batch = 4
x0 = torch.tensor([[7], [9]], dtype=torch.double)
sita = 5
phi = -0.5
rho = 0.2
wmax = 0.5
wmin = -0.5
K_s = torch.tensor([[0, -1]], dtype=torch.double)
evolution = 4


def x1_dot(x1, x2, u):
    return -x1+x2*torch.sin(x2)


def x2_dot(x1, x2, u):
    return u-x1*torch.sin(x2)

# def x1_dot(x1, x2, u):
#     return 4.0*x2


# def x2_dot(x1, x2, u):
#     return x1+u
