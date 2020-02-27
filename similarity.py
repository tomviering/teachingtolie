import torch
from numpy import intersect1d
from numpy import linspace, meshgrid

# let A and B be explenation of size [batch size * W * H]
# this function computes the spearman correlation of the batch
def spearman_rank(A, B):
    Af = torch.reshape(A, (256, -1))
    Ai = torch.argsort(Af, dim=1)

    Bf = torch.reshape(B, (256, -1))
    Bi = torch.argsort(Bf, dim=1)

    diff = Ai - Bi
    diff = diff ** 2
    diff = diff.float()

    num = float(6) / float((Ai.shape[1]) * (Ai.shape[1] ** 2 - 1))

    spearman = 1 - torch.sum(diff, 1) * num

    return spearman


# let A and B be explenation of size [batch size * W * H]
# this function computes the size of the intersection of the
# k most important features in the batch (summed over the batch)
def top_k(A, B, k=5):
    Af = torch.reshape(A, (256, -1))
    Atop = torch.topk(Af, k, 1)  # sorts along pixels
    Ai = Atop.indices

    Bf = torch.reshape(B, (256, -1))
    Btop = torch.topk(Bf, k, 1)
    Bi = Btop.indices

    num = 0

    for i in range(0, Ai.shape[0]):  # loop over batch
        Av = Ai[i, :]
        Bv = Bi[i, :]
        An = Av.cpu().numpy()
        Bn = Bv.cpu().numpy()
        intersection = intersect1d(An, Bn, assume_unique=False, return_indices=False)

        num = num + len(intersection)

    return num

def get_center(A):
    nx = A.shape[1]
    ny = A.shape[2]

    x = linspace(0, 1, nx)
    y = linspace(0, 1, ny)
    xv, yv = meshgrid(x, y)
    
    if A.is_cuda:
        xv, yv = torch.Tensor(xv).cuda(), torch.Tensor(yv).cuda()
    else:
        xv, yv = torch.Tensor(xv), torch.Tensor(yv) 
        
    x_c = (A*xv).sum((1,2))/A.sum((1,2))
    y_c = (A*yv).sum((1,2))/A.sum((1,2))
    
    return x_c, y_c

def center_loss_tom(A, B):

    x_A, y_A = get_center(A)
    x_B, y_B = get_center(B)

    return torch.mean(torch.sqrt((x_A - x_B)**2 + (y_A - y_B)**2), 0)


if __name__ == '__main__':
    E1 = torch.rand(256, 14, 14)
    E2 = torch.rand(256, 14, 14)

    spearman = spearman_rank(E1, E2)
    print(spearman)

    top = top_k(E1, E2, 5)
    print(top)

    print('distance')
    dist = center_loss_tom(E1, E2)
    print(dist)

