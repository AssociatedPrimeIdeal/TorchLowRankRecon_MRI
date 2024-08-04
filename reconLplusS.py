from utils import *
def recon_LplusS(A, Kv, method, llr, it, regL, regS, blk, Lc=1., device='cuda', gt=None, save_loss=False):
    '''
    :param A: Operator consists of under-sampling mask, fourier transform and coil sensitivity maps
    :param Kv: Acquired under-sampled k-space
    :param method: Iteration method (ISTA, FISTA, POGM)
    :param llr: Use locally low-rank or globally low-rank
    :param it: Iteration number
    :param regL: regularization of the low-rank term
    :param regS: regularization of the sparse term
    :param blk: block size for locally low-rank
    :param Lc: Lipschitz constant
    :param device: cuda or cpu
    :param gt: ground truth image, if it is not None, PSNR, SSIM, and nRMSE will be calculated and stored in metrics
    :param save_loss: if save_loss is True, loss will be calculated and stored in metrics
    :return: X:reconstructed image, metrics:dictionary
    '''
    Np, Nt, Nc, FE, PE, SPE = Kv.shape
    loop = tqdm.tqdm(range(1, it + 1), total=it)
    metrics = {}
    if save_loss:
        metrics['loss'] = []
        metrics['loss1'] = []
        metrics['loss2'] = []
        metrics['loss3'] = []
    if gt is not None:
        metrics['psnr'] = []
        metrics['ssim'] = []
        metrics['nrmse'] = []

    if llr:
        stepx = np.ceil(FE / blk)
        stepy = np.ceil(PE / blk)
        stepz = np.ceil(SPE / blk)
        padx = (stepx * blk).astype('int64')
        pady = (stepy * blk).astype('int64')
        padz = (stepz * blk).astype('int64')
        M = blk ** 3
        N = Nt * Np
        B = padx * pady * padz / M
        RF = GETWIDTH(M, N, B)
        regL *= RF
    else:
        regL *= (np.sqrt(np.prod(Kv.shape[-3:])) + 1)

    if method == 'ISTA':
        X = A.mtimes(Kv, 1)
        L, Lp = X.clone(), X.clone()
        S = torch.zeros_like(X).to(device)
        for i in loop:
            if llr:
                L, loss2 = SVT_LLR(X - S, regL / Lc, blk)
            else:
                L, loss2 = SVT(X - S, regL / Lc)
            S, loss3 = Sparse(X - Lp, regS / Lc)
            axb = A.mtimes(L + S, 0) - Kv
            X = L + S - 1 / Lc * A.mtimes(axb, 1)
            Lp = L
            if save_loss:
                loss1 = torch.sum(torch.abs(A.mtimes(X, 0) - Kv) ** 2).item()
                metrics['loss1'].append(loss1)
                metrics['loss2'].append(loss2)
                metrics['loss3'].append(loss3)
                metrics['loss'].append(loss1 * 0.5 + loss2 * regL + loss3 * regS)
            if gt is not None:
                metrics['psnr'].append(PSNR(torch.abs(X), gt[0], use_torch=True).item())
                metrics['ssim'].append(SSIM(torch.abs(X.unsqueeze(0)), gt, use_torch=True))
                metrics['nrmse'].append(nRMSE(torch.abs(X), gt[0], use_torch=True).item())
    elif method == 'FISTA':
        tp = 1
        X = A.mtimes(Kv, 1)
        L, Lp, Lh = X.clone(), X.clone(), X.clone()
        S, Sp, Sh = torch.zeros_like(X).to(device), torch.zeros_like(X).to(device), torch.zeros_like(X).to(device)
        for i in loop:
            t = (1 + np.sqrt(1 + 4 * tp ** 2)) / 2
            if llr:
                L, loss2 = SVT_LLR(X - Sh, regL / Lc, blk)
            else:
                L, loss2 = SVT(X - Sh, regL / Lc)
            S, loss3 = Sparse(X - Lh, regS / Lc)
            Lh = L + (tp - 1) / t * (L - Lp)
            Sh = S + (tp - 1) / t * (S - Sp)
            axb = A.mtimes(Lh + Sh, 0) - Kv
            X = Lh + Sh - 1 / Lc * A.mtimes(axb, 1)
            tp = t
            Lp = L
            Sp = S
            if save_loss:
                loss1 = torch.sum(torch.abs(A.mtimes(X, 0) - Kv) ** 2).item()
                metrics['loss1'].append(loss1)
                metrics['loss2'].append(loss2)
                metrics['loss3'].append(loss3)
                metrics['loss'].append(loss1 * 0.5 + loss2 * regL + loss3 * regS)
            if gt is not None:
                metrics['psnr'].append(PSNR(torch.abs(X), gt[0], use_torch=True).item())
                metrics['ssim'].append(SSIM(torch.abs(X.unsqueeze(0)), gt, use_torch=True))
                metrics['nrmse'].append(nRMSE(torch.abs(X), gt[0], use_torch=True).item())
    elif method == 'POGM':
        tp = 1
        gp = 1
        X = A.mtimes(Kv, 1)
        L, L_, L_p, Lh, Lhp = X.clone(), X.clone(), X.clone(), X.clone(), X.clone()
        S, S_, S_p, Sh, Shp = torch.zeros_like(X).to(device), torch.zeros_like(X).to(device), torch.zeros_like(X).to(
            device), torch.zeros_like(X).to(device), torch.zeros_like(X).to(device)
        for i in loop:
            Lh = X - S
            Sh = X - L
            t = (1 + np.sqrt(1 + 4 * tp ** 2)) / 2
            L_ = Lh + (tp - 1) / t * (Lh - Lhp) + tp / t * (Lh - L) + (tp - 1) / (gp * t) * 1 / Lc * (L_p - L)
            S_ = Sh + (tp - 1) / t * (Sh - Shp) + tp / t * (Sh - S) + (tp - 1) / (gp * t) * 1 / Lc * (S_p - S)
            g = 1 / Lc * (1 + (tp - 1) / t + tp / t)
            if llr:
                L, loss2 = SVT_LLR(L_, regL * g, blk)
            else:
                L, loss2 = SVT(L_, regL * g)
            S, loss3 = Sparse(S_, regS * g)
            axb = A.mtimes(L + S, 0) - Kv
            X = L + S - 1 / Lc * A.mtimes(axb, 1)
            tp = t
            gp = g
            L_p = L_
            S_p = S_
            Lhp = Lh
            Shp = Sh
            if save_loss:
                loss1 = torch.sum(torch.abs(A.mtimes(X, 0) - Kv) ** 2).item()
                metrics['loss1'].append(loss1)
                metrics['loss2'].append(loss2)
                metrics['loss3'].append(loss3)
                metrics['loss'].append(loss1 * 0.5 + loss2 * regL + loss3 * regS)
            if gt is not None:
                metrics['psnr'].append(PSNR(torch.abs(X), gt[0], use_torch=True).item())
                metrics['ssim'].append(SSIM(torch.abs(X.unsqueeze(0)), gt, use_torch=True))
                metrics['nrmse'].append(nRMSE(torch.abs(X), gt[0], use_torch=True).item())
    return X, L, S, metrics