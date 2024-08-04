import sys
import os
import numpy as np
import torch
import torch.nn.functional as F
import time
import argparse
import tqdm
import random
import matplotlib.pyplot as plt
import matplotlib
import ssim3d

seed = 1
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
os.environ['TOOLBOX_PATH'] = '/data/ryy/homebackup/dl_project/bart'
sys.path.append('/data/ryy/homebackup/dl_project/bart/python')
from bart import bart


def k2i_torch(K, ax=[-3, -2, -1]):
    X = torch.fft.fftshift(torch.fft.ifftn(torch.fft.ifftshift(K, dim=ax), dim=ax, norm="ortho"),
                           dim=ax)
    return X


def i2k_torch(K, ax=[-3, -2, -1]):
    X = torch.fft.fftshift(torch.fft.fftn(torch.fft.ifftshift(K, dim=ax), dim=ax, norm="ortho"),
                           dim=ax)
    return X


def nRMSE(pred, target, use_torch=False):
    if use_torch:
        return (torch.sqrt(torch.mean((pred - target) ** 2)) / (torch.max(target) - torch.min(target)))
    else:
        return np.sqrt(np.mean((pred - target) ** 2)) / (np.max(target) - np.min(target))


def SSIM(pred, target, device='cuda', use_torch=False):
    if not use_torch:
        pred = torch.as_tensor(np.ascontiguousarray(pred)).to(torch.float32).to(device)
        target = torch.as_tensor(np.ascontiguousarray(target)).to(torch.float32).to(device)
    ssim = 0
    for i in range(pred.shape[0]):
        ssim += ssim3d.ssim3D(pred[i], target[i])
    return (ssim / pred.shape[0] * 100).item()


def PSNR(pred, target, peakval=1., use_torch=False):
    if use_torch:
        mse = torch.mean((pred - target) ** 2)
        return 10 * torch.log10(peakval / mse)
    else:
        mse = np.mean((pred - target) ** 2)
        return 10 * np.log10(peakval / mse)

def minmax(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))

def SoftThres(X, reg):
    X = torch.sgn(X) * (torch.abs(X) - reg) * ((torch.abs(X) - reg) > 0)
    return X

def Sparse(S, reg, ax=(0, 1)):
    temp = SoftThres(i2k_torch(S, ax=ax), reg)
    return k2i_torch(temp, ax=ax), torch.sum(torch.abs(temp)).item()

def GETWIDTH(M, N, B):
    temp = (np.sqrt(M) + np.sqrt(N))
    if M > N:
        return temp + np.sqrt(np.log(B * N))
    else:
        return temp + np.sqrt(np.log(B * M))


def SVT(X, reg):
    Np, Nt, FE, PE, SPE = X.shape
    U, S, Vh = torch.linalg.svd(X.view(Np * Nt, -1), full_matrices=False)
    S_new = SoftThres(S, reg)
    S_new = torch.diag_embed(S_new).to(torch.complex64)
    X = torch.linalg.matmul(torch.linalg.matmul(U, S_new), Vh).view(Np, Nt, FE, PE, SPE)
    return X, torch.sum(torch.abs(S_new)).item()


def SVT_LLR(X, reg, blk):
    Np, Nt, FE, PE, SPE = X.shape
    stepx = np.ceil(FE / blk)
    stepy = np.ceil(PE / blk)
    stepz = np.ceil(SPE / blk)
    padx = (stepx * blk).astype('int64')
    pady = (stepy * blk).astype('int64')
    padz = (stepz * blk).astype('int64')
    rrx = torch.randperm(blk)[0]
    rry = torch.randperm(blk)[0]
    rrz = torch.randperm(blk)[0]
    X = F.pad(X, (0, padz - SPE, 0, pady - PE, 0, padx - FE))
    X = torch.roll(X, (rrz, rry, rrx), (-1, -2, -3))
    FEp, PEp, SPEp = X.shape[-3:]
    patches = X.unfold(2, blk, blk).unfold(3, blk, blk).unfold(4, blk, blk)
    unfold_shape = patches.size()
    patches = patches.contiguous().view(Np, Nt, -1, blk, blk, blk).permute((2, 0, 1, 3, 4, 5))
    Nb = patches.shape[0]
    U, S, Vh = torch.linalg.svd(patches.view(Nb, Np * Nt, -1), full_matrices=False)
    S_new = SoftThres(S, reg)
    S_new = torch.diag_embed(S_new).to(torch.complex64)
    patches = torch.linalg.matmul(torch.linalg.matmul(U, S_new), Vh).view(Nb, Np, Nt, blk, blk, blk)
    patches = patches.permute((1, 2, 0, 3, 4, 5))
    patches_orig = patches.view(unfold_shape)
    patches_orig = patches_orig.permute(0, 1, 2, 5, 3, 6, 4, 7).contiguous()
    patches_orig = patches_orig.view(Np, Nt, FEp, PEp, SPEp)
    patches_orig = torch.roll(patches_orig, (-rrz, -rry, -rrx), (-1, -2, -3))
    X = patches_orig[..., :FE, :PE, :SPE]
    return X, torch.sum(torch.abs(S_new)).item()

def make_mask(usv, t, PE, SPE):
    ng1, ng2 = np.meshgrid(np.linspace(-1, 1, PE), np.linspace(-1, 1, SPE), indexing='ij')
    v = np.sqrt(ng1 ** 2 + ng2 ** 2)
    v = np.reshape(v, [1, PE, SPE])
    v = v / np.max(v)
    masks = np.random.uniform(size=[t, PE, SPE]) > v ** usv
    masks[:, PE // 2, SPE // 2] = 1.
    # Nt FE PE SPE
    return np.expand_dims(masks, axis=((0, 1, 3, 4)))


class Eop():
    def __init__(self, csm, us_mask):
        super(Eop, self).__init__()
        self.csm = csm
        self.us_mask = us_mask

    def mtimes(self, b, inv):
        if inv:
            # b: nv,nt,nc,x,y,z
            x = torch.sum(k2i_torch(b * self.us_mask, ax=[-3, -2, -1]) * torch.conj(self.csm), dim=2)
        else:
            b = b.unsqueeze(2) * self.csm
            x = i2k_torch(b, ax=[-3, -2, -1]) * self.us_mask
        return x
