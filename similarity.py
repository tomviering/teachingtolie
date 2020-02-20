import torch
from numpy import intersect1d


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


if __name__ == '__main__':
    E1 = torch.rand(256, 14, 14)
    E2 = torch.rand(256, 14, 14)

    spearman = spearman_rank(E1, E2)
    print(spearman)

    top = top_k(E1, E2, 5)
    print(top)
