import glob
import os
import time
from collections import OrderedDict
import torchvision.transforms.functional as TF
import numpy as np
import torch
import cv2
import argparse
from natsort import natsort
from IQA_pytorch import SSIM,LPIPSvgg
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import lpips
from PIL import Image

class Measure():
    def __init__(self, net='alex', use_gpu=False):
        self.device = 'cuda' if use_gpu else 'cpu'
        self.model = lpips.LPIPS(net=net)
        self.model.to(self.device)

    def measure(self, imgA, imgB):
        return [float(f(imgA, imgB)) for f in [self.psnr, self.ssim, self.lpips]]

    def lpips(self, imgA, imgB, model=None):
        tA = t(imgA).to(self.device)
        tB = t(imgB).to(self.device)
        dist01 = self.model.forward(tA, tB).item()
        return dist01

    def ssim(self, imgA, imgB):
        
        score, diff = ssim(imgA, imgB, full=True, channel_axis=2)
        return score

    def psnr(self, imgA, imgB):
        # print(imgA.shape,imgB.shape)
        psnr_val = psnr(imgA, imgB)
        return psnr_val


def t(img):
    def to_4d(img):
        assert len(img.shape) == 3
        assert img.dtype == np.uint8
        img_new = np.expand_dims(img, axis=0)
        assert len(img_new.shape) == 4
        return img_new

    def to_CHW(img):
        return np.transpose(img, [2, 0, 1])

    def to_tensor(img):
        return torch.Tensor(img)

    return to_tensor(to_4d(to_CHW(img))) / 127.5 - 1


def fiFindByWildcard(wildcard):
    return natsort.natsorted(glob.glob(wildcard, recursive=True))


def imread(path):
    # img = Image.open(path).convert('RGB')
    # img= TF.to_tensor(img).unsqueeze(0).cuda()
    # return img
    return cv2.imread(path)[:, :, [2, 1, 0]]


def format_result(psnr, ssim, lpips):
    return f'{psnr:0.2f}, {ssim:0.3f}, {lpips:0.3f}'

def measure_dirs(dirA, dirB, use_gpu, verbose=False):
    if verbose:
        vprint = lambda x: print(x)
    else:
        vprint = lambda x: None


    t_init = time.time()

    paths_A = fiFindByWildcard(os.path.join(dirA, f'*.{type}'))
    paths_B = fiFindByWildcard(os.path.join(dirB, f'*.{type}'))

    vprint("Comparing: ")
    vprint(dirA)
    vprint(dirB)

    measure = Measure(use_gpu=use_gpu)

    results = []
    for pathA, pathB in zip(paths_A, paths_B):
        result = OrderedDict()

        t = time.time()
        result['psnr'], result['ssim'], result['lpips'] = measure.measure(imread(pathA), imread(pathB))
        d = time.time() - t
        vprint(f"{pathA.split('/')[-1]}, {pathB.split('/')[-1]}, {format_result(**result)}, {d:0.1f}")

        results.append(result)

    psnr = np.mean([result['psnr'] for result in results])
    ssim = np.mean([result['ssim'] for result in results])
    lpips = np.mean([result['lpips'] for result in results])

    vprint(f"Final Result: {format_result(psnr, ssim, lpips)}, {time.time() - t_init:0.1f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-dirA', default='/root/autodl-tmp/dataset/lol/lol_dataset/Real_world/eval/high', type=str) #gt
    parser.add_argument('-dirB', default='/root/autodl-tmp/MonoPix/results/MonoPix_LowLight_Default5_without_ISN__exhaustive/LOL', type=str) #lq
    parser.add_argument('-type', default='png')
    parser.add_argument('--use_gpu', action='store_true', default=True)
    args = parser.parse_args()

    dirA = args.dirA
    dirB = args.dirB
    type = args.type
    use_gpu = args.use_gpu

    if len(dirA) > 0 and len(dirB) > 0:
        measure_dirs(dirA, dirB, use_gpu=use_gpu, verbose=True)