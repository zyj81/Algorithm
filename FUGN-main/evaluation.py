import matplotlib.pyplot as plt
import cv2
import os
import skimage
import imageio as iio
from uiqm import *
from skimage.metrics import peak_signal_noise_ratio,mean_squared_error
import torch
import piq
def NormalizeData(data):
    return (data - data.min()) / (data.max() - data.min())

target=[]
generated=[]

generated_addr="./result/net_C/UIEB/"
target_addr="./dataset/UIEB/test/target/"

for item in os.listdir(generated_addr):
    if item.endswith(".png"):
        generated.append((((cv2.cvtColor(cv2.imread(generated_addr+item), cv2.COLOR_BGR2RGB).astype("float32")))))
for item in os.listdir(target_addr):
    if item.endswith(".png"):
        target.append((cv2.cvtColor(cv2.imread(target_addr+item), cv2.COLOR_BGR2RGB).astype("float32")))

SSIM_results=[]
PSNR_results=[]
FSIM_results=[]
UIQM=[]
UCIQE=[]

for i in range(len(target)):

    PSNR = peak_signal_noise_ratio(NormalizeData(generated[i]), NormalizeData(target[i]))
    PSNR_results.append(PSNR)

    SSIM = structural_similarity(
        NormalizeData(generated[i]),
        NormalizeData(target[i]),
        data_range=1.0,
        channel_axis=-1  # 新版本替代 multichannel
    )
    SSIM_results.append(SSIM)

    gen_tensor = torch.from_numpy(NormalizeData(generated[i]).transpose(2, 0, 1)).unsqueeze(0).float().clamp(0, 1)
    gt_tensor = torch.from_numpy(NormalizeData(target[i]).transpose(2, 0, 1)).unsqueeze(0).float().clamp(0, 1)
    fsim_score = piq.fsim(gen_tensor, gt_tensor, data_range=1.0)
    FSIM_results.append(fsim_score.item())


print(np.mean(SSIM_results), np.mean(PSNR_results), np.mean(FSIM_results),np.mean(UIQM),np.mean(UCIQE))