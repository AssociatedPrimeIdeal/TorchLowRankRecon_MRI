from reconLLR import *
from utils import *

torch.set_num_threads(os.cpu_count())
seed = 1
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

def test_LLR():
    data = np.load('/nas-data/xcat2/res2/data_xcat_res2.npy', allow_pickle=True).item()
    K = data['kspc']
    csm = data['sens']
    device = 'cuda'

    # recon param
    llr = True
    it = 50
    SNR = 40
    reg = 0.01
    blk = 16
    L = 2

    Nv, Np, Nt, Nc, FE, PE, SPE = K.shape
    sos = np.sqrt(np.sum(np.abs(csm) ** 2, axis=0, keepdims=True)) + 1e-11
    csm /= sos
    csm = torch.as_tensor(np.ascontiguousarray(csm)).to(torch.complex64).to(device)

    img_gt = torch.sum(k2i_torch(torch.as_tensor(np.ascontiguousarray(K)).to(torch.complex64).to(device),
                                 ax=[-3, -2, -1]) * torch.conj(csm), 3).cpu().numpy()
    img_gt = torch.as_tensor(np.ascontiguousarray(abs(img_gt))).to(torch.float32).to(device)

    # generate us mask
    us_mask_gen = make_mask(0.06, Nt, PE, SPE)
    us_rate = 1 / np.mean(us_mask_gen)

    # add noise
    stdev = 1 / SNR
    noise = stdev * (np.random.randn(*K.shape) + 1j * np.random.randn(*K.shape))
    Knoise = K + noise
    img_noise = torch.sum(
        k2i_torch(torch.as_tensor(np.ascontiguousarray(Knoise)).to(torch.complex64).to(device),
                  ax=[-3, -2, -1]) * torch.conj(csm), 3).cpu().numpy()
    Knoise *= us_mask_gen
    img_noise = torch.as_tensor(np.ascontiguousarray(abs(img_noise))).to(torch.float32).to(device)

    param_string = 'US Rate: {us_rate:.2f}_SNR: {SNR:d}_Iter Num: {it:d}_Reg: {reg:0.4f}_Block Size: {blk:d}'.format(
        us_rate=us_rate, SNR=SNR, it=it, reg=reg, blk=blk)
    print(param_string)

    methods = ['BART', 'POGM']
    X = np.zeros((Nv, Np, Nt, FE, PE, SPE)).astype('complex64')
    for v in range(Nv):
        print("#Velocity Encoding " + str(v + 1).zfill(3))
        for i, method in enumerate(methods):
            print('Method:', method)
            st = time.time()
            Kv = torch.as_tensor(np.ascontiguousarray(Knoise[v])).to(torch.complex64).to(device)
            us_mask = (torch.abs(Kv[:, :, 0:1, FE // 2:FE // 2 + 1]) > 0).to(torch.float32).to(device)
            rcomb = torch.sum(k2i_torch(Kv, ax=[-3, -2, -1]) * torch.conj(csm), 2)
            regFactor = torch.max(torch.abs(rcomb))
            Kv /= regFactor
            Xv, metrics = recon_LLR(Eop(csm, us_mask), Kv, method, llr, it, reg, blk, device=device, L=L, gt=img_gt,
                                    save_loss=True)
            if device == 'cuda' and method != 'BART':
                Xv = Xv.cpu().numpy()
            X[v] = Xv
            print("TIME COMSUMING:{:.2f}s".format(time.time() - st))
            nrmse = nRMSE(np.abs(X), np.abs(img_gt.cpu().numpy()))
            psnr = PSNR(np.abs(X), np.abs(img_gt.cpu().numpy()))
            ssim = SSIM(np.abs(X), np.abs(img_gt.cpu().numpy()), device)
            print("nRMSE:{:.4f}".format(nrmse))
            print("PSNR:{:.2f}".format(psnr))
            print("SSIM:{:.2f}".format(ssim))
            plt.figure(figsize=(20, 5))
            plt.subplot(1, 4, 1)
            plt.imshow(abs(img_gt[v, 0, Nt // 2, :, :, SPE // 2]).cpu().numpy(), cmap='gray',
                       origin='lower',
                       norm=matplotlib.colors.Normalize(0, 1))
            plt.title("Ground Truth")
            plt.axis('off')
            plt.subplot(1, 4, 2)
            plt.imshow(abs(img_noise[v, 0, Nt // 2, :, :, SPE // 2]).cpu().numpy(), cmap='gray',
                       origin='lower',
                       norm=matplotlib.colors.Normalize(0, 1))
            plt.title("Noisy img")
            plt.subplot(1, 4, 3)
            plt.imshow(abs(Xv[0, Nt//2, :, :, SPE // 2]), cmap='gray',
                                origin='lower',
                                norm=matplotlib.colors.Normalize(0, 1))
            plt.title(method)
            plt.subplot(1, 4, 4)
            plt.imshow(abs(abs(Xv[0, Nt//2, :, :, SPE // 2]) - abs(img_gt[v, 0, Nt//2, :, :, SPE // 2]).cpu().numpy()), cmap='gray',
                                origin='lower',
                                norm=matplotlib.colors.Normalize(0, 0.3))
            plt.title("Difference")
            plt.show()

def test_LplusS():
    data = np.load('/nas-data/xcat2/res2/data_xcat_res2.npy', allow_pickle=True).item()
    K = data['kspc']
    csm = data['sens']
    device = 'cuda'

    # recon param
    llr = True
    it = 50
    SNR = 40
    reg = 0.01
    blk = 16
    L = 2

    Nv, Np, Nt, Nc, FE, PE, SPE = K.shape
    sos = np.sqrt(np.sum(np.abs(csm) ** 2, axis=0, keepdims=True)) + 1e-11
    csm /= sos
    csm = torch.as_tensor(np.ascontiguousarray(csm)).to(torch.complex64).to(device)

    img_gt = torch.sum(k2i_torch(torch.as_tensor(np.ascontiguousarray(K)).to(torch.complex64).to(device),
                                 ax=[-3, -2, -1]) * torch.conj(csm), 3).cpu().numpy()
    img_gt = torch.as_tensor(np.ascontiguousarray(abs(img_gt))).to(torch.float32).to(device)

    # generate us mask
    us_mask_gen = make_mask(0.06, Nt, PE, SPE)
    us_rate = 1 / np.mean(us_mask_gen)

    # add noise
    stdev = 1 / SNR
    noise = stdev * (np.random.randn(*K.shape) + 1j * np.random.randn(*K.shape))
    Knoise = K + noise
    img_noise = torch.sum(
        k2i_torch(torch.as_tensor(np.ascontiguousarray(Knoise)).to(torch.complex64).to(device),
                  ax=[-3, -2, -1]) * torch.conj(csm), 3).cpu().numpy()
    Knoise *= us_mask_gen
    img_noise = torch.as_tensor(np.ascontiguousarray(abs(img_noise))).to(torch.float32).to(device)

    param_string = 'US Rate: {us_rate:.2f}_SNR: {SNR:d}_Iter Num: {it:d}_Reg: {reg:0.4f}_Block Size: {blk:d}'.format(
        us_rate=us_rate, SNR=SNR, it=it, reg=reg, blk=blk)
    print(param_string)

    methods = ['BART', 'POGM']
    X = np.zeros((Nv, Np, Nt, FE, PE, SPE)).astype('complex64')
    for v in range(Nv):
        print("#Velocity Encoding " + str(v + 1).zfill(3))
        for i, method in enumerate(methods):
            print('Method:', method)
            st = time.time()
            Kv = torch.as_tensor(np.ascontiguousarray(Knoise[v])).to(torch.complex64).to(device)
            us_mask = (torch.abs(Kv[:, :, 0:1, FE // 2:FE // 2 + 1]) > 0).to(torch.float32).to(device)
            rcomb = torch.sum(k2i_torch(Kv, ax=[-3, -2, -1]) * torch.conj(csm), 2)
            regFactor = torch.max(torch.abs(rcomb))
            Kv /= regFactor
            Xv, metrics = recon_LLR(Eop(csm, us_mask), Kv, method, llr, it, reg, blk, device=device, L=L, gt=img_gt,
                                    save_loss=True)
            if device == 'cuda' and method != 'BART':
                Xv = Xv.cpu().numpy()
            X[v] = Xv
            print("TIME COMSUMING:{:.2f}s".format(time.time() - st))
            nrmse = nRMSE(np.abs(X), np.abs(img_gt.cpu().numpy()))
            psnr = PSNR(np.abs(X), np.abs(img_gt.cpu().numpy()))
            ssim = SSIM(np.abs(X), np.abs(img_gt.cpu().numpy()), device)
            print("nRMSE:{:.4f}".format(nrmse))
            print("PSNR:{:.2f}".format(psnr))
            print("SSIM:{:.2f}".format(ssim))
            plt.figure(figsize=(20, 5))
            plt.subplot(1, 4, 1)
            plt.imshow(abs(img_gt[v, 0, Nt // 2, :, :, SPE // 2]).cpu().numpy(), cmap='gray',
                       origin='lower',
                       norm=matplotlib.colors.Normalize(0, 1))
            plt.title("Ground Truth")
            plt.axis('off')
            plt.subplot(1, 4, 2)
            plt.imshow(abs(img_noise[v, 0, Nt // 2, :, :, SPE // 2]).cpu().numpy(), cmap='gray',
                       origin='lower',
                       norm=matplotlib.colors.Normalize(0, 1))
            plt.title("Noisy img")
            plt.subplot(1, 4, 3)
            plt.imshow(abs(Xv[0, Nt//2, :, :, SPE // 2]), cmap='gray',
                                origin='lower',
                                norm=matplotlib.colors.Normalize(0, 1))
            plt.title(method)
            plt.subplot(1, 4, 4)
            plt.imshow(abs(abs(Xv[0, Nt//2, :, :, SPE // 2]) - abs(img_gt[v, 0, Nt//2, :, :, SPE // 2]).cpu().numpy()), cmap='gray',
                                origin='lower',
                                norm=matplotlib.colors.Normalize(0, 0.3))
            plt.title("Difference")
            plt.show()

if __name__ == '__main__':
    test_LLR()
