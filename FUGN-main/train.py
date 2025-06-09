import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

import os
import argparse
import time
import utils
from tqdm import tqdm
from warmup_scheduler import GradualWarmupScheduler
from loss import Charbonnier_Loss, SSIM_Loss, torchPSNR

from networks import network, graph_network

from graphmethods import build_adjacency_matrices

def train(config):
    graph_net = graph_network.graph_net(config.block_size).cuda()

    # Trainable_params = sum(p.numel() for p in graph_net.parameters() if p.requires_grad)
    # print(f'Trainable params: {Trainable_params / 1e6}M')


    print("gpu_id:", config.cudaid)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = config.cudaid
    device_ids = [i for i in range(torch.cuda.device_count())]

    if torch.cuda.device_count() > 1:
        graph_net = nn.DataParallel(graph_net, device_ids=device_ids)

    train_dataset = utils.train_val_loader(config.enhan_images_path,config.ori_images_path)
    val_dataset = utils.train_val_loader(config.enhan_images_path,config.ori_images_path, mode="val")
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=True, num_workers=config.num_workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config.val_batch_size, shuffle=False,num_workers=config.num_workers, pin_memory=True)

    ######### Graph_adj ###########
    train_adj = build_adjacency_matrices(train_loader, config.block_size)
    val_adj = build_adjacency_matrices(val_loader, config.block_size)

    ######### Adam optimizer ###########
    optimizer = optim.Adam(graph_net.parameters(), lr=config.lr)
    ######### Scheduler ###########
    warmup_epochs = 3
    scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, config.num_epochs - warmup_epochs,
                                                            eta_min=config.lr)
    scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epochs,
                                       after_scheduler=scheduler_cosine)
    scheduler.step()

    criterion_char = Charbonnier_Loss()
    criterion_ssim = SSIM_Loss()

    graph_net.train()

    # Record best index and corresponding epoch
    best_psnr = 0
    best_epoch = 0

    for epoch in range(1, config.num_epochs+1):
        epoch_start_time = time.time()
        # Record train loss and validation index
        train_loss = []
        val_psnr = []
        print("*" * 30 + "The %i epoch" % epoch + "*" * 30+'\n')

        for i, (img_clean, img_ori) in enumerate(tqdm(train_loader)):
            img_clean = img_clean.cuda()
            img_ori = img_ori.cuda()

            try:
                train_adj_batch = train_adj[i]
                #此处来获取深度图
                # d = 传统算法(img_ori)
                # enhanced_image = enhan_net(img_ori, d)
                enhanced_image = graph_net(img_ori, train_adj_batch)
                char_loss = criterion_char(img_clean, enhanced_image)
                ssim_loss = criterion_ssim(img_clean, enhanced_image)
                ssim_loss = 1 - ssim_loss

                sum_loss = char_loss + 0.5 * ssim_loss

                train_loss.append(sum_loss.item())
                optimizer.zero_grad()
                sum_loss.backward()
                torch.nn.utils.clip_grad_norm_(graph_net.parameters(), config.grad_clip_norm)
                optimizer.step()

            except RuntimeError as e:
                if 'out of memory' in str(e):
                    print(e)
                    torch.cuda.empty_cache()
                else:
                    raise e

        with open(os.path.join(config.checkpoint_path, config.net_name, "loss.log"), "a+", encoding="utf-8") as f:
            s = "The %i Epoch mean_loss is :%f" % (epoch, np.mean(train_loss)) + "\n"
            f.write(s)

        # Validation Stage
        with torch.no_grad():
            for i, (img_clean, img_ori) in enumerate(val_loader):
                val_adj_batch = val_adj[i]
                img_clean = img_clean.cuda()
                img_ori = img_ori.cuda()
                # 此处来获取深度图
                # d = 传统算法(img_ori)
                # enhanced_image = enhan_net(img_ori, d)
                enhanced_image = graph_net(img_ori, val_adj_batch)

                psnr = torchPSNR(img_clean, enhanced_image)
                val_psnr.append(psnr.item())

        val_psnr = np.mean(np.array(val_psnr))

        if val_psnr > best_psnr:
            best_psnr = val_psnr
            best_epoch = epoch
            torch.save({'epoch': epoch,
                        'state_dict': graph_net.state_dict(),
                        'optimizer': optimizer.state_dict()
                        }, os.path.join(config.checkpoint_path, config.net_name, "model_best.pth"))
        scheduler.step()

        print("------------------------------------------------------------------")
        print("Epoch: {}\tTime: {:.4f}\tLoss: {:.4f}\tLearningRate {:.6f}".format
              (epoch, time.time() - epoch_start_time, np.mean(train_loss), scheduler.get_lr()[0]))

        print("------------------------------------------------------------------")

        print("[epoch %d PSNR: %.4f --- best_epoch %d Best_PSNR %.4f]" %
              (epoch, val_psnr, best_epoch, best_psnr))

        with open(os.path.join(config.checkpoint_path, config.net_name, "val_PSNR.log"), "a+", encoding="utf-8") as f:
            f.write("[epoch %d PSNR: %.4f --- best_epoch %d Best_PSNR %.4f]" %
                    (epoch, val_psnr, best_epoch, best_psnr) + "\n")

        torch.save({'epoch': epoch,
                    # 'state_dict': enhan_net.state_dict(),
                    'state_dict': graph_net.state_dict(),
                    'optimizer': optimizer.state_dict()
                    }, os.path.join(config.checkpoint_path, config.net_name, "model_latest.pth"))

if __name__ == "__main__":
    """
        input:enhanced underwater images and original underwater images
    	param enhan_images_path:enhanced underwater images
    	param orig_images_path:original underwater images
    """

    torch.cuda.empty_cache()
    parser = argparse.ArgumentParser()

    parser.add_argument('--block_size', type=int, default="16")

    # Input Parameters
    parser.add_argument('--net_name', type=str, default="net_C")
    parser.add_argument('--d_net_name', type=str, default="d_net")
    # parser.add_argument('--enhan_images_path', type=str, default="../target/")
    # parser.add_argument('--ori_images_path', type=str, default="../input/")
    parser.add_argument('--enhan_images_path', type=str, default="./dataset/UIEB/train/target/")
    parser.add_argument('--ori_images_path', type=str, default="./dataset/UIEB/train/input/")

    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--grad_clip_norm', type=float, default=0.1)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--train_batch_size', type=int, default=2)
    parser.add_argument('--val_batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=6)
    parser.add_argument('--checkpoint_path', type=str, default="./trained_model/")
    parser.add_argument('--cudaid', type=str, default="0",help="choose cuda device id 0-7).")

    config = parser.parse_args()

    if not os.path.exists(os.path.join(config.checkpoint_path, config.net_name)):
        os.mkdir(os.path.join(config.checkpoint_path, config.net_name))

    torch.cuda.empty_cache()
    s = time.time()
    train(config)
    e = time.time()
    print(str(e-s))
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import numpy as np
# import os
# import argparse
# import time
# import utils
# from tqdm import tqdm
# from warmup_scheduler import GradualWarmupScheduler
# from loss import Charbonnier_Loss, SSIM_Loss, torchPSNR, FSIM
# from networks import graph_network
# from graphmethods import build_adjacency_matrices
#
# def train(config):
#     graph_net = graph_network.graph_net(config.block_size).cuda()
#     print("gpu_id:", config.cudaid)
#     os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#     os.environ["CUDA_VISIBLE_DEVICES"] = config.cudaid
#     device_ids = [i for i in range(torch.cuda.device_count())]
#
#     if torch.cuda.device_count() > 1:
#         graph_net = nn.DataParallel(graph_net, device_ids=device_ids)
#
#     train_dataset = utils.train_val_loader(config.enhan_images_path, config.ori_images_path)
#     val_dataset = utils.train_val_loader(config.enhan_images_path, config.ori_images_path, mode="val")
#
#     train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=True,
#                                                num_workers=config.num_workers, pin_memory=True)
#     val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config.val_batch_size, shuffle=False,
#                                              num_workers=config.num_workers, pin_memory=True)
#
#     train_adj = build_adjacency_matrices(train_loader, config.block_size)
#     val_adj = build_adjacency_matrices(val_loader, config.block_size)
#
#     optimizer = optim.Adam(graph_net.parameters(), lr=config.lr)
#
#     warmup_epochs = 3
#     scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, config.num_epochs - warmup_epochs,
#                                                             eta_min=config.lr)
#     scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epochs,
#                                        after_scheduler=scheduler_cosine)
#     scheduler.step()
#
#     criterion_char = Charbonnier_Loss()
#     criterion_ssim = SSIM_Loss()
#
#     graph_net.train()
#
#     best_psnr = 0
#     best_epoch = 0
#
#     for epoch in range(1, config.num_epochs + 1):
#         epoch_start_time = time.time()
#         train_loss = []
#         val_psnr, val_ssim, val_fsim = [], [], []
#         print("*" * 30 + f"The {epoch} epoch" + "*" * 30 + '\n')
#
#         for i, (img_clean, img_ori) in enumerate(tqdm(train_loader)):
#             img_clean = img_clean.cuda()
#             img_ori = img_ori.cuda()
#
#             try:
#                 train_adj_batch = train_adj[i]
#                 enhanced_image = graph_net(img_ori, train_adj_batch)
#                 char_loss = criterion_char(img_clean, enhanced_image)
#                 ssim_loss = 1 - criterion_ssim(img_clean, enhanced_image)
#
#                 sum_loss = char_loss + 0.5 * ssim_loss
#                 train_loss.append(sum_loss.item())
#
#                 optimizer.zero_grad()
#                 sum_loss.backward()
#                 torch.nn.utils.clip_grad_norm_(graph_net.parameters(), config.grad_clip_norm)
#                 optimizer.step()
#
#             except RuntimeError as e:
#                 if 'out of memory' in str(e):
#                     print(e)
#                     torch.cuda.empty_cache()
#                 else:
#                     raise e
#
#         with open(os.path.join(config.checkpoint_path, config.net_name, "loss.log"), "a+", encoding="utf-8") as f:
#             f.write(f"The {epoch} Epoch mean_loss is :{np.mean(train_loss):.6f}\n")
#
#         # Validation Stage
#         with torch.no_grad():
#             for i, (img_clean, img_ori) in enumerate(val_loader):
#                 val_adj_batch = val_adj[i]
#                 img_clean = img_clean.cuda()
#                 img_ori = img_ori.cuda()
#                 enhanced_image = graph_net(img_ori, val_adj_batch)
#
#                 psnr = torchPSNR(img_clean, enhanced_image)
#                 ssim = criterion_ssim(img_clean, enhanced_image)
#                 fsim = FSIM(img_clean, enhanced_image)
#
#                 val_psnr.append(psnr.item())
#                 val_ssim.append(ssim.item())
#                 val_fsim.append(fsim.item())
#
#         psnr_mean = np.mean(val_psnr)
#         ssim_mean = np.mean(val_ssim)
#         fsim_mean = np.mean(val_fsim)
#
#         if psnr_mean > best_psnr:
#             best_psnr = psnr_mean
#             best_epoch = epoch
#             torch.save({'epoch': epoch,
#                         'state_dict': graph_net.state_dict(),
#                         'optimizer': optimizer.state_dict()},
#                        os.path.join(config.checkpoint_path, config.net_name, "model_best.pth"))
#
#         scheduler.step()
#
#         print("-" * 66)
#         print(f"Epoch: {epoch}\tTime: {time.time() - epoch_start_time:.4f}\tLoss: {np.mean(train_loss):.4f}\tLR: {scheduler.get_lr()[0]:.6f}")
#         print(f"[Validation] PSNR: {psnr_mean:.4f}, SSIM: {ssim_mean:.4f}, FSIM: {fsim_mean:.4f} --- Best PSNR: {best_psnr:.4f} at Epoch {best_epoch}")
#         print("-" * 66)
#
#         with open(os.path.join(config.checkpoint_path, config.net_name, "val_metrics.log"), "a+", encoding="utf-8") as f:
#             f.write(f"[Epoch {epoch}] PSNR: {psnr_mean:.4f}, SSIM: {ssim_mean:.4f}, FSIM: {fsim_mean:.4f}\n")
#
#         torch.save({'epoch': epoch,
#                     'state_dict': graph_net.state_dict(),
#                     'optimizer': optimizer.state_dict()},
#                    os.path.join(config.checkpoint_path, config.net_name, "model_latest.pth"))
#
# if __name__ == "__main__":
#     torch.cuda.empty_cache()
#     parser = argparse.ArgumentParser()
#
#     parser.add_argument('--block_size', type=int, default=16)
#     parser.add_argument('--net_name', type=str, default="net_C")
#     parser.add_argument('--d_net_name', type=str, default="d_net")
#     parser.add_argument('--enhan_images_path', type=str, default="D:/PHD/UIEB/base/UIEB/train/target/")
#     parser.add_argument('--ori_images_path', type=str, default="D:/PHD/UIEB/base/UIEB/train/input/")
#     parser.add_argument('--lr', type=float, default=2e-4)
#     parser.add_argument('--grad_clip_norm', type=float, default=0.1)
#     parser.add_argument('--num_epochs', type=int, default=100)
#     parser.add_argument('--train_batch_size', type=int, default=2)
#     parser.add_argument('--val_batch_size', type=int, default=1)
#     parser.add_argument('--num_workers', type=int, default=6)
#     parser.add_argument('--checkpoint_path', type=str, default="./trained_model/")
#     parser.add_argument('--cudaid', type=str, default="0")
#
#     config = parser.parse_args()
#
#     os.makedirs(os.path.join(config.checkpoint_path, config.net_name), exist_ok=True)
#
#     s = time.time()
#     train(config)
#     e = time.time()
#     print(f"Total Training Time: {e - s:.2f} seconds")
