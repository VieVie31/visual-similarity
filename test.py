import torch
import torch.nn.functional as F

eps = 1e-5

a, b = torch.relu(torch.randn(4, 10)), torch.relu(torch.randn(4, 10))
a, b = F.normalize(a), F.normalize(b)

T = lambda a,b : a*b
A = lambda a,b : T(a,b) + eps

def s(a, b):
    return (a * b * A(a, b)).sum(1)


# this doesn't work
x = s(a.repeat(b.shape[0], 1), b.repeat(1, a.shape[0]).reshape(-1, 10))

print(x.shape)

def test(a, b):
    n, m = a.shape[0], b.shape[0]
    i, j = torch.meshgrid(torch.arange(n), torch.arange(m))

    res = ((a[i] * b[j]) * (a[i] * b[j] + eps))
    #res = (a[i] * b[j])
    # sum over 1
    res = res.reshape(n*m, -1).sum(1)

    # reshape to get the matrix form

    return res.reshape(n, m)

"""
if we were using just res = (a[i] * b[j]), the whole test function is equivalent to a @ b.T
"""
