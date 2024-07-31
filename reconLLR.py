from utils import *
def recon_LLR(A, Kv, method, llr, it, reg, blk, L=1., device='cuda', gt=None, save_loss=False):
    '''
    :param A: Operator consists of under-sampling mask, fourier transform and coil sensitivity maps
    :param Kv: Acquired under-sampled k-space
    :param method: Iteration method (ISTA, FISTA, POGM)
    :param llr: Use locally low-rank or globally low-rank
    :param it: Iteration number
    :param reg: regularization of the nuclear norm term
    :param blk: block size for locally low-rank
    :param L: Lipschitz constant
    :param device: cuda or cpu
    :param gt: ground truth image, if it is not None, PSNR, SSIM, and nRMSE will be calculated and stored in metrics
    :param save_loss: if save_loss is True, loss will be calculated and stored in metrics
    :return: X:reconstructed image, metrics:dictionary
    '''
    Np, Nt, Nc, FE, PE, SPE = Kv.shape
    if method != 'BART':
        loop = tqdm.tqdm(range(1, it + 1), total=it)
    metrics = {}
    if save_loss:
        metrics['loss'] = []
        metrics['loss1'] = []
        metrics['loss2'] = []
    if gt is not None:
        metrics['psnr'] = []
        metrics['ssim'] = []
        metrics['nrmse'] = []
    if method != 'BART':
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
            reg *= RF
        else:
            reg *= (np.sqrt(np.prod(Kv.shape[-3:])) + 1)
    if method == 'ISTA':
        X = A.mtimes(Kv, 1)
        for i in loop:
            axb = A.mtimes(X, 0) - Kv
            X = X - 1 / L * A.mtimes(axb, 1)
            if llr:
                X, loss2 = SVT_LLR(X, reg, blk)
            else:
                X, loss2 = SVT(X, reg)
            if save_loss:
                loss1 = torch.sum(torch.abs(A.mtimes(X, 0) - Kv) ** 2).item()
                metrics['loss1'].append(loss1)
                metrics['loss2'].append(loss2)
                metrics['loss'].append(loss1 * 0.5 + loss2 * reg)
            if gt is not None:
                metrics['psnr'].append(PSNR(torch.abs(X), gt[0], use_torch=True).item())
                metrics['ssim'].append(SSIM(torch.abs(X.unsqueeze(0)), gt, use_torch=True))
                metrics['nrmse'].append(nRMSE(torch.abs(X), gt[0], use_torch=True).item())
    elif method == 'FISTA':
        tp = 1
        Xp = A.mtimes(Kv, 1)
        Y = Xp.clone()
        for i in loop:
            t = (1 + np.sqrt(1 + 4 * tp ** 2)) / 2
            axb = A.mtimes(Y, 0) - Kv
            Y = Y - 1 / L * A.mtimes(axb, 1)
            if llr:
                X, loss2 = SVT_LLR(Y, reg, blk)
            else:
                X, loss2 = SVT(Y, reg)
            Y = X + (tp - 1) / t * (X - Xp)
            Xp = X
            tp = t
            if save_loss:
                loss1 = torch.sum(torch.abs(A.mtimes(X, 0) - Kv) ** 2).item()
                metrics['loss1'].append(loss1)
                metrics['loss2'].append(loss2)
                metrics['loss'].append(loss1 * 0.5 + loss2 * reg)
            if gt is not None:
                metrics['psnr'].append(PSNR(torch.abs(X), gt[0], use_torch=True).item())
                metrics['ssim'].append(SSIM(torch.abs(X.unsqueeze(0)), gt, use_torch=True))
                metrics['nrmse'].append(nRMSE(torch.abs(X), gt[0], use_torch=True).item())
    elif method == 'POGM':
        tp = 1
        gp = 1
        Xp = A.mtimes(Kv, 1)
        X, Y, Z, Yp, Zp = Xp.clone(), Xp.clone(), Xp.clone(), Xp.clone(), Xp.clone()
        for i in loop:
            t = (1 + np.sqrt(1 + 4 * tp ** 2)) / 2
            g = 1 / L * (2 * tp + t - 1) / t
            axb = A.mtimes(X, 0) - Kv
            Y = X - 1 / L * A.mtimes(axb, 1)
            Z = Y + (tp - 1) / t * (Y - Yp) + tp / t * (Y - Xp) + (tp - 1) / (L * gp * t) * (Zp - Xp)
            if llr:
                X, loss2 = SVT_LLR(Z, reg * g, blk)
            else:
                X, loss2 = SVT(Z, reg * g)
            Xp = X
            Yp = Y
            Zp = Z
            tp = t
            gp = g
            if save_loss:
                loss1 = torch.sum(torch.abs(A.mtimes(X, 0) - Kv) ** 2).item()
                metrics['loss1'].append(loss1)
                metrics['loss2'].append(loss2)
                metrics['loss'].append(loss1 * 0.5 + loss2 * reg)
            if gt is not None:
                metrics['psnr'].append(PSNR(torch.abs(X), gt[0], use_torch=True).item())
                metrics['ssim'].append(SSIM(torch.abs(X.unsqueeze(0)), gt, use_torch=True))
                metrics['nrmse'].append(nRMSE(torch.abs(X), gt[0], use_torch=True).item())
    # elif method == 'ADMM':
    elif method == 'BART':
        Kv = Kv.cpu().numpy()
        csm = A.csm.cpu().numpy()
        Kv = Kv.transpose((3, 4, 5, 2, 1, 0))  # FE PE SPE Nc Nt Np
        csm = csm.transpose((1, 2, 3, 0))
        if llr:
            rllr = 7
        else:
            rllr = 0
        if device == 'cuda':
            bart_string = 'pics -s1 -u1 -w 1 -i %d -R L:7:%d:%.3e -b %d -g' % (it, rllr, reg, blk)
        elif device == 'cpu':
            bart_string = 'pics -s1 -u1 -w 1 -i %d -R L:7:%d:%.3e -b %d ' % (it, rllr, reg, blk)

        X = bart(1, bart_string, Kv[:, :, :, :, None, None, None, None, None, None, :, :], csm)
        X = np.transpose(X[:, :, :, 0, 0, 0, 0, 0, 0, :, :], [3, 4, 0, 1, 2])
    return X, metrics