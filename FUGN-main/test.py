import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import torch
import torch.nn as nn
import torchvision
import torch.optim
import numpy as np


import argparse
import utils
from tqdm import tqdm

from networks import graph_network
from graphmethods import build_adjacency_matrices

import time




def test(config):
    enhan_net = graph_network.graph_net(config.block_size).cuda()
    utils.load_checkpoint(enhan_net, os.path.join(config.checkpoint_path, config.net_name, 'model_best.pth'))

    print("gpu_id:", config.cudaid)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = config.cudaid
    device_ids = [i for i in range(torch.cuda.device_count())]

    if torch.cuda.device_count() > 1:
        enhan_net = nn.DataParallel(enhan_net, device_ids=device_ids)

    print(os.path.join(config.ori_images_path, config.dataset_name))
    test_dataset = utils.test_loader(config.ori_images_path)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False,
                                              num_workers=config.num_workers, drop_last=False, pin_memory=True)
    test_adj = build_adjacency_matrices(test_loader, config.block_size)

    result_dir = os.path.join(config.result_path, config.net_name, config.dataset_name)

    enhan_net.eval()

    with torch.no_grad():
        for i, (img_ori, filenames) in enumerate(tqdm(test_loader)):
            test_adj_batch = test_adj[i]
            torch.cuda.ipc_collect()
            torch.cuda.empty_cache()
            img_ori = img_ori.cuda()
            # depth_image = d_net(img_ori, False)
            from thop import profile
            flops, params = profile(enhan_net, inputs=(img_ori, test_adj_batch))

            enhan_image = enhan_net(img_ori, test_adj_batch)

            for j in range(len(enhan_image)):
                torchvision.utils.save_image(enhan_image[j], os.path.join(result_dir, os.path.basename(filenames[j])))
            # torchvision.utils.save_image(enhan_image, os.path.join(result_dir, filenames[0]))
            # print(filenames[0], "is done!")


if __name__ == '__main__':
    """
        param orig_images_path:original underwater images
    """

    torch.cuda.empty_cache()
    parser = argparse.ArgumentParser()
    parser.add_argument('--block_size', type=int, default="16")

    # Input Parameters
    parser.add_argument('--net_name', type=str, default="net_C")
    # parser.add_argument('--d_net_name', type=str, default="d_net")
    parser.add_argument('--dataset_name', type=str, default="UIEB")
    # parser.add_argument('--dataset_name', type=str, default="test_no_ref")
    # parser.add_argument('--ori_images_path', type=str, default="E:/PHD/UIEB/base/UIEB/test/ref/input/")
    parser.add_argument('--ori_images_path', type=str, default="./dataset/UIEB/test/input")
    # parser.add_argument('--ori_images_path', type=str, default="./dataset/UIEB/no-reference")
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=6)
    parser.add_argument('--checkpoint_path', type=str, default="./trained_model/")
    parser.add_argument('--result_path', type=str, default="./result/")
    parser.add_argument('--cudaid', type=str, default="0", help="choose cuda device id 0-7).")

    config = parser.parse_args()

    if not os.path.exists(os.path.join(config.result_path, config.net_name)):
        os.mkdir(os.path.join(config.result_path, config.net_name))

    if not os.path.exists(os.path.join(config.result_path, config.net_name, config.dataset_name)):
        os.mkdir(os.path.join(config.result_path, config.net_name, config.dataset_name))

    start_time = time.time()
    test(config)
    print("final_time:" + str(time.time() - start_time))
# import torch
# import torch.nn as nn
# import torchvision
# import numpy as np
# import os
# import argparse
# from PIL import Image
# from tqdm import tqdm
# from networks import graph_network
# from graphmethods import build_adjacency_matrices
# from loss import torchPSNR, SSIM_Loss
# from piq import fsim
#
# from utils import load_checkpoint, test_loader
#
# def load_gt_image(gt_dir, filename):
#     gt_path = os.path.join(gt_dir, filename)
#     gt_img = Image.open(gt_path).convert('RGB')
#     gt_img = np.asarray(gt_img) / 255.0
#     gt_img = torch.from_numpy(gt_img).permute(2, 0, 1).float()
#     return gt_img
#
# def ensure_dir(path):
#     if not os.path.exists(path):
#         os.makedirs(path)
#
# def test(config):
#     enhan_net = graph_network.graph_net(config.block_size).cuda()
#     load_checkpoint(enhan_net, os.path.join(config.checkpoint_path, config.net_name, 'model_best.pth'))
#
#     print("gpu_id:", config.cudaid)
#     os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#     os.environ["CUDA_VISIBLE_DEVICES"] = config.cudaid
#     device_ids = list(range(torch.cuda.device_count()))
#
#     if torch.cuda.device_count() > 1:
#         enhan_net = nn.DataParallel(enhan_net, device_ids=device_ids)
#
#     input_path = os.path.join(config.ori_images_path, "input")
#     gt_path = os.path.join(config.ori_images_path, "target")
#     result_path = os.path.join(config.result_path, config.net_name, config.dataset_name)
#     ensure_dir(result_path)
#
#     test_dataset = test_loader(input_path)
#     test_loader_ = torch.utils.data.DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False,
#                                                 num_workers=config.num_workers, drop_last=False, pin_memory=True)
#     test_adj = build_adjacency_matrices(test_loader_, config.block_size)
#
#     enhan_net.eval()
#
#     ssim_fn = SSIM_Loss()
#     psnr_list, ssim_list, fsim_list = [], [], []
#
#     with torch.no_grad():
#         for i, (img_input, filenames) in enumerate(tqdm(test_loader_)):
#             img_input = img_input.cuda()
#             adj = test_adj[i]
#
#             output = enhan_net(img_input, adj)
#
#             for j in range(len(output)):
#                 pred = output[j].unsqueeze(0)
#                 input_img = img_input[j].unsqueeze(0)
#
#                 # Load GT image
#                 gt_img = load_gt_image(gt_path, os.path.basename(filenames[j])).unsqueeze(0).cuda()
#                 # 修复FSIM输入异常：确保值在[0,1]
#                 pred = torch.clamp(pred, 0, 1)
#                 gt_img = torch.clamp(gt_img, 0, 1)
#                 # Metrics
#                 psnr_val = torchPSNR(gt_img, pred).item()
#                 ssim_val = ssim_fn(gt_img, pred).item()
#                 fsim_val = fsim(pred, gt_img, data_range=1.).item()
#
#                 psnr_list.append(psnr_val)
#                 ssim_list.append(ssim_val)
#                 fsim_list.append(fsim_val)
#
#                 # Save image
#                 torchvision.utils.save_image(pred, os.path.join(result_path, os.path.basename(filenames[j])))
#
#     print("\n==== Test Metrics ====")
#     print(f"Average PSNR: {np.mean(psnr_list):.4f}")
#     print(f"Average SSIM: {np.mean(ssim_list):.4f}")
#     print(f"Average FSIM: {np.mean(fsim_list):.4f}")
#
# if __name__ == '__main__':
#     torch.cuda.empty_cache()
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--block_size', type=int, default=16)
#     parser.add_argument('--net_name', type=str, default="net_C")
#     parser.add_argument('--dataset_name', type=str, default="UIEB")
#     parser.add_argument('--ori_images_path', type=str, default="D:/PHD/UIEB/base/UIEB/test/")  # 应包含input/ 和 target/
#     parser.add_argument('--batch_size', type=int, default=1)
#     parser.add_argument('--num_workers', type=int, default=6)
#     parser.add_argument('--checkpoint_path', type=str, default="./trained_model/")
#     parser.add_argument('--result_path', type=str, default="./result/")
#     parser.add_argument('--cudaid', type=str, default="0")
#
#     config = parser.parse_args()
#     test(config)


