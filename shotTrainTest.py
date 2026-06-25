#!/usr/bin/python
# -*- encoding: utf-8 -*-
import os
import argparse
import datetime
import time
import logging
from pathlib import Path
import warnings
import random

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, ReduceLROnPlateau

from logger import setup_logger
from Fusionnet import MACTFusion
from Networks.FusionNet import UNetFusionModel as PRFusion
from loss import Fusionloss

import numpy as np
from PIL import Image
import cv2
import glob
import pandas as pd

warnings.filterwarnings('ignore')

MODEL_REGISTRY = {
    'MACTFusion': MACTFusion,
    'PRFusion': PRFusion,
}

# ==============================================================================
# Evaluation Metrics
# ==============================================================================
def mutual_information(img1, img2, bins=256):
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    img1_flat = img1.ravel()
    img2_flat = img2.ravel()
    hist_joint, _, _ = np.histogram2d(img1_flat, img2_flat, bins=bins)
    p_joint = hist_joint / np.sum(hist_joint)
    p1 = np.sum(p_joint, axis=1)
    p2 = np.sum(p_joint, axis=0)
    p1[p1 == 0] = 1e-10
    p2[p2 == 0] = 1e-10
    p_joint[p_joint == 0] = 1e-10
    h1 = -np.sum(p1 * np.log2(p1))
    h2 = -np.sum(p2 * np.log2(p2))
    h12 = -np.sum(p_joint * np.log2(p_joint))
    return h1 + h2 - h12


def mean_squared_error(img1, img2):
    return np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)


def correlation_coefficient(img1, img2):
    img1 = img1.astype(np.float64).ravel()
    img2 = img2.astype(np.float64).ravel()
    cov = np.cov(img1, img2)[0, 1]
    var1 = np.var(img1)
    var2 = np.var(img2)
    if var1 == 0 or var2 == 0:
        return 0.0
    return cov / np.sqrt(var1 * var2)


def peak_signal_noise_ratio(img1, img2, data_range=1.0):
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100.0
    return 10 * np.log10((data_range ** 2) / mse)


def perceptual_quality(img1, img2, win_size=11, sigma=1.5):
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    window = cv2.getGaussianKernel(win_size, sigma)
    window = np.outer(window, window.transpose())
    C1 = (0.01 * 1.0) ** 2
    C2 = (0.03 * 1.0) ** 2
    mu1 = cv2.filter2D(img1, -1, window)[win_size//2:-win_size//2+1, win_size//2:-win_size//2+1]
    mu2 = cv2.filter2D(img2, -1, window)[win_size//2:-win_size//2+1, win_size//2:-win_size//2+1]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[win_size//2:-win_size//2+1, win_size//2:-win_size//2+1] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[win_size//2:-win_size//2+1, win_size//2:-win_size//2+1] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[win_size//2:-win_size//2+1, win_size//2:-win_size//2+1] - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return np.mean(ssim_map)


def visual_information_fidelity(img1, img2, sigma_n=0.01):
    try:
        img1 = img1.astype(np.float64)
        img2 = img2.astype(np.float64)
        if img1.ndim == 3 and img1.shape[-1] == 3:
            img1 = np.dot(img1[..., :3], [0.2989, 0.5870, 0.1140])
        if img2.ndim == 3 and img2.shape[-1] == 3:
            img2 = np.dot(img2[..., :3], [0.2989, 0.5870, 0.1140])

        def gaussian_pyramid(img, levels=3):
            pyramid = [img]
            for _ in range(levels - 1):
                img = cv2.pyrDown(img)
                pyramid.append(img)
            return pyramid

        pyramid1 = gaussian_pyramid(img1)
        pyramid2 = gaussian_pyramid(img2)
        viff = 0
        total_weight = 0
        for l in range(len(pyramid1)):
            weight = 2 ** (-2 * l)
            total_weight += weight
            im1 = pyramid1[l]
            im2 = pyramid2[l]
            im1_flat = im1.ravel()
            im2_flat = im2.ravel()
            cov1 = np.cov(im1_flat)
            cov12 = np.cov(im1_flat, im2_flat)[0, 1]
            if cov1 + sigma_n ** 2 == 0:
                continue
            viff += weight * (cov12 ** 2) / (cov1 + sigma_n ** 2)
        return viff / total_weight if total_weight > 0 else 0.0
    except Exception as e:
        logging.warning(f"VIFF calculation failed: {e}")
        return np.nan


def q_abf(fused_img, img1, img2):
    try:
        def to_grayscale(img):
            if img.ndim == 3 and img.shape[-1] == 3:
                return np.dot(img[..., :3], [0.2989, 0.5870, 0.1140])
            return img.squeeze()

        fused = to_grayscale(fused_img)
        im1 = to_grayscale(img1)
        im2 = to_grayscale(img2)

        def edge_strength(img):
            sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
            return np.sqrt(sobel_x ** 2 + sobel_y ** 2)

        edge_fused = edge_strength(fused)
        edge1 = edge_strength(im1)
        edge2 = edge_strength(im2)
        weight1 = edge1 / (edge1 + edge2 + 1e-10)
        weight2 = edge2 / (edge1 + edge2 + 1e-10)
        q1 = perceptual_quality(fused, im1)
        q2 = perceptual_quality(fused, im2)
        return np.mean(weight1 * q1 + weight2 * q2)
    except Exception as e:
        logging.warning(f"QABF calculation failed: {e}")
        return np.nan


def sum_correlation_differences(fused_img, img1, img2):
    try:
        def to_grayscale(img):
            if img.ndim == 3 and img.shape[-1] == 3:
                return np.dot(img[..., :3], [0.2989, 0.5870, 0.1140])
            return img.squeeze()
        fused = to_grayscale(fused_img)
        im1 = to_grayscale(img1)
        im2 = to_grayscale(img2)
        cc1 = correlation_coefficient(fused, im1)
        cc2 = correlation_coefficient(fused, im2)
        cc_source = correlation_coefficient(im1, im2)
        return np.abs(cc1 - cc_source) + np.abs(cc2 - cc_source)
    except Exception as e:
        logging.warning(f"SCD calculation failed: {e}")
        return np.nan


def save_metrics_to_csv(metrics, csv_path, append=True):
    df = pd.DataFrame([metrics])
    if append and os.path.exists(csv_path):
        df.to_csv(csv_path, mode='a', header=False, index=False)
    else:
        df.to_csv(csv_path, mode='w', header=True, index=False)


# ==============================================================================
# Dataset for Few-shot
# ==============================================================================
class Fusion_dataset(torch.utils.data.Dataset):
    def __init__(self, split, data_dir_vis, data_dir_ir, length=0, fold=0, num_folds=2, train_ratio=None):
        super(Fusion_dataset, self).__init__()
        assert split in ['train', 'test'], 'split must be "train"|"test"'
        self.filepath_ir = []
        self.filenames_ir = []
        self.filepath_vis = []
        self.filenames_vis = []
        self.length = length
        self.split = split
        self.train_ratio = train_ratio

        vis_files = glob.glob(os.path.join(data_dir_vis, "*.png"))
        ir_files = glob.glob(os.path.join(data_dir_ir, "*.png"))
        vis_files.sort()
        ir_files.sort()

        all_files = list(zip(vis_files, ir_files))
        test_split_idx = int(len(all_files) * 0.8)
        train_all_files = all_files[:test_split_idx]
        test_files = all_files[test_split_idx:]

        if split == 'test':
            self.filepath_vis = [file[0] for file in test_files]
            self.filepath_ir = [file[1] for file in test_files]
        else:
            selected_files = train_all_files
            if self.train_ratio is not None and 0 < self.train_ratio < 1:
                num_samples = int(len(selected_files) * self.train_ratio)
                num_samples = max(1, num_samples)
                selected_files = selected_files[:num_samples]
            self.filepath_vis = [file[0] for file in selected_files]
            self.filepath_ir = [file[1] for file in selected_files]

        self.filenames_vis = [os.path.basename(f) for f in self.filepath_vis]
        self.filenames_ir = [os.path.basename(f) for f in self.filepath_ir]

    def __getitem__(self, index):
        vis_path = self.filepath_vis[index]
        ir_path = self.filepath_ir[index]
        image_vis = cv2.imread(vis_path)
        image_inf = cv2.imread(ir_path, 0)
        image_vis = (np.asarray(Image.fromarray(image_vis), dtype=np.float32).transpose((2, 0, 1)) / 255.0)
        image_ir = np.asarray(Image.fromarray(image_inf), dtype=np.float32) / 255.0
        image_ir = np.expand_dims(image_ir, axis=0)
        name = self.filenames_vis[index]
        return torch.tensor(image_vis), torch.tensor(image_ir), name

    def __len__(self):
        return len(self.filepath_ir)


# ==============================================================================
# Utility Functions
# ==============================================================================
def prepare_tensor_for_eval(tensor):
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    if tensor.dim() == 3 and tensor.size(0) in [1, 3]:
        tensor = tensor.permute(1, 2, 0)
    return tensor


def save_image_original_color(tensor, save_path):
    try:
        img_tensor = tensor.squeeze()
        if img_tensor.dim() == 3 and img_tensor.size(0) in [1, 3]:
            img_tensor = img_tensor.permute(1, 2, 0)
        img_np = img_tensor.cpu().detach().numpy()
        if img_np.ndim == 3:
            if img_np.max() <= 1.0:
                img_np = (img_np * 255).astype(np.uint8)
            if img_np.shape[-1] == 3:
                img = Image.fromarray(img_np, mode='RGB')
            else:
                img = Image.fromarray(img_np.squeeze(), mode='L')
        else:
            if img_np.max() <= 1.0:
                img_np = (img_np * 255).astype(np.uint8)
            img = Image.fromarray(img_np, mode='L')
        img.save(save_path)
    except Exception as e:
        logging.error(f"Save image error: {e}")


def RGB2YCrCb(input_im):
    im_flat = input_im.transpose(1, 3).transpose(1, 2).reshape(-1, 3)
    R = im_flat[:, 0]
    G = im_flat[:, 1]
    B = im_flat[:, 2]
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cr = (R - Y) * 0.713 + 0.5
    Cb = (B - Y) * 0.564 + 0.5
    Y = torch.unsqueeze(Y, 1)
    Cr = torch.unsqueeze(Cr, 1)
    Cb = torch.unsqueeze(Cb, 1)
    temp = torch.cat((Y, Cr, Cb), dim=1).cuda()
    out = temp.reshape(list(input_im.size())[0], list(input_im.size())[2],
                       list(input_im.size())[3], 3).transpose(1, 3).transpose(2, 3)
    return out


def YCrCb2RGB(input_im):
    im_flat = input_im.transpose(1, 3).transpose(1, 2).reshape(-1, 3)
    mat = torch.tensor([[1.0, 1.0, 1.0], [1.403, -0.714, 0.0], [0.0, -0.344, 1.773]]).cuda()
    bias = torch.tensor([0.0 / 255, -0.5, -0.5]).cuda()
    temp = (im_flat + bias).mm(mat).cuda()
    out = temp.reshape(list(input_im.size())[0], list(input_im.size())[2],
                       list(input_im.size())[3], 3).transpose(1, 3).transpose(2, 3)
    return out


# ==============================================================================
# Argument Parser
# ==============================================================================
def parse_args():
    parser = argparse.ArgumentParser(description='PRFusion: Few-shot Training & Testing')

    parser.add_argument('--train_ratio', '-tr', type=float, required=True,
                        choices=[0.1, 0.3, 0.5],
                        help='Training data ratio (0.1=10%, 0.3=30%, 0.5=50%)')
    parser.add_argument('--model_name', '-M', type=str, default='MACTFusion',
                        choices=list(MODEL_REGISTRY.keys()))
    parser.add_argument('--output_channels', '-oc', type=int, default=1)
    parser.add_argument('--batch_size', '-B', type=int, default=2)
    parser.add_argument('--epochs', '-E', type=int, default=30)
    parser.add_argument('--lr_start', '-lr', type=float, default=0.0005)
    parser.add_argument('--lr_decay_type', type=str, default='cosine',
                        choices=['step', 'cosine', 'plateau', 'poly'])
    parser.add_argument('--lr_decay_rate', '-lrd', type=float, default=0.9)
    parser.add_argument('--lr_decay_epoch', '-lre', type=int, default=5)
    parser.add_argument('--lr_min', type=float, default=1e-6)
    parser.add_argument('--optimizer', type=str, default='adamw',
                        choices=['adam', 'adamw', 'sgd'])
    parser.add_argument('--gpu', '-G', type=int, default=0)
    parser.add_argument('--num_workers', '-j', type=int, default=4)
    parser.add_argument('--seed', '-s', type=int, default=42)
    parser.add_argument('--method', '-method', type=str, default='MACTFusion')
    parser.add_argument('--exp_name', '-exp', type=str, default='SPECT-MRI',
                        choices=['CT-MRI', 'PET-MRI', 'SPECT-MRI'])
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Root path to Havard-Medical-Image-Fusion-Datasets-main')
    parser.add_argument('--log_dir', type=str, default='./logs_fewshot')
    parser.add_argument('--model_dir', type=str, default='./models_fewshot')
    parser.add_argument('--result_dir', type=str, default='./results_fewshot')
    parser.add_argument('--skip_viff', action='store_true')
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_experiment_dirs(args):
    ratio_percent = f"{int(args.train_ratio * 100)}percent"
    main_model_dir = Path(args.model_dir) / args.exp_name / args.method / ratio_percent
    log_dir = Path(args.log_dir) / args.exp_name / args.method / ratio_percent
    result_dir = Path(args.result_dir) / args.exp_name / args.method / ratio_percent
    fused_dir = result_dir / 'fused_images'
    vis_dir = result_dir / 'vis_images'
    ir_dir = result_dir / 'ir_images'
    for dir_path in [main_model_dir, log_dir, result_dir, fused_dir, vis_dir, ir_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    return main_model_dir, log_dir, result_dir, fused_dir, vis_dir, ir_dir


def get_optimizer(model, args):
    if args.optimizer == 'adam':
        return torch.optim.Adam(model.parameters(), lr=args.lr_start, betas=(0.9, 0.999), weight_decay=1e-4)
    elif args.optimizer == 'adamw':
        return torch.optim.AdamW(model.parameters(), lr=args.lr_start, betas=(0.9, 0.999), weight_decay=0.01)
    elif args.optimizer == 'sgd':
        return torch.optim.SGD(model.parameters(), lr=args.lr_start, momentum=0.9, weight_decay=1e-4)


def get_scheduler(optimizer, args):
    if args.lr_decay_type == 'step':
        return StepLR(optimizer, step_size=args.lr_decay_epoch, gamma=args.lr_decay_rate)
    elif args.lr_decay_type == 'cosine':
        return CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr_min)
    elif args.lr_decay_type == 'plateau':
        return ReduceLROnPlateau(optimizer, mode='min', factor=args.lr_decay_rate,
                                 patience=args.lr_decay_epoch // 2, min_lr=args.lr_min)
    else:
        def poly_lr(epoch):
            lr = args.lr_start * (1 - epoch / args.epochs) ** 0.9
            return max(lr, args.lr_min)
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=poly_lr)


# ==============================================================================
# Training & Testing
# ==============================================================================
def train_fewshot(args, logger, model_dir):
    torch.cuda.set_device(args.gpu)
    ratio_percent = f"{int(args.train_ratio * 100)}%"
    logger.info(f"Initializing model {args.model_name} with {ratio_percent} training samples")
    fusionmodel = MODEL_REGISTRY[args.model_name]()
    fusionmodel = fusionmodel.cuda()
    fusionmodel.train()

    optimizer = get_optimizer(fusionmodel, args)
    scheduler = get_scheduler(optimizer, args)

    exp_path_map = {
        "CT-MRI": ("CT-MRI/CT", "CT-MRI/MRI"),
        "PET-MRI": ("PET-MRI/PET", "PET-MRI/MRI"),
        "SPECT-MRI": ("SPECT-MRI/SPECT", "SPECT-MRI/MRI")
    }
    vis_subdir, ir_subdir = exp_path_map[args.exp_name]
    data_dir_vis = os.path.join(args.data_dir, vis_subdir)
    data_dir_ir = os.path.join(args.data_dir, ir_subdir)

    train_dataset = Fusion_dataset(split='train', data_dir_vis=data_dir_vis,
                                   data_dir_ir=data_dir_ir, train_ratio=args.train_ratio)
    test_dataset = Fusion_dataset(split='test', data_dir_vis=data_dir_vis, data_dir_ir=data_dir_ir)
    logger.info(f"Dataset sizes - Train: {len(train_dataset)}, Test: {len(test_dataset)}")

    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers,
                              pin_memory=True, drop_last=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)

    criteria_fusion = Fusionloss()
    best_train_loss = float('inf')
    st = glob_st = time.time()

    for epo in range(args.epochs):
        fusionmodel.train()
        train_losses = []

        for it, (image_vis, image_ir, name) in enumerate(train_loader):
            image_vis = Variable(image_vis).cuda()
            image_ir = Variable(image_ir).cuda()
            image_vis_ycrcb = RGB2YCrCb(image_vis)
            image_vis_y = image_vis_ycrcb[:, 0:1, :, :]
            logits = fusionmodel(image_vis_y, image_ir)
            fusion_ycrcb = torch.cat([logits, image_vis_ycrcb[:, 1:2, :, :],
                                      image_vis_ycrcb[:, 2:3, :, :]], dim=1)
            fusion_image = YCrCb2RGB(fusion_ycrcb)
            fusion_image = torch.clamp(fusion_image, 0, 1)

            optimizer.zero_grad()
            loss_fusion, loss_in, ssim_loss, loss_grad = criteria_fusion(
                image_vis=image_vis_ycrcb, image_ir=image_ir, generate_img=logits, i=0, labels=None)
            loss_total = loss_fusion
            loss_total.backward()
            optimizer.step()
            train_losses.append(loss_total.item())

            if (it + 1) % 5 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                logger.info(f'Epoch {epo+1}/{args.epochs} | Step {it+1}/{len(train_loader)} | '
                            f'Loss: {loss_total.item():.4f} | In: {loss_in.item():.4f} | '
                            f'Grad: {loss_grad.item():.4f} | SSIM: {ssim_loss.item():.4f} | LR: {current_lr:.6f}')

        avg_train_loss = sum(train_losses) / len(train_losses)
        if args.lr_decay_type == 'plateau':
            scheduler.step(avg_train_loss)
        else:
            scheduler.step()
        logger.info(f'Epoch {epo+1} | Train Loss: {avg_train_loss:.4f} | '
                    f'LR: {optimizer.param_groups[0]["lr"]:.6f}')

        if avg_train_loss < best_train_loss:
            best_train_loss = avg_train_loss
            model_file = model_dir / 'best_model.pth'
            torch.save({'epoch': epo + 1, 'model_state_dict': fusionmodel.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'best_train_loss': best_train_loss, 'args': args,
                        'train_ratio': args.train_ratio}, model_file)
            logger.info(f'Saved best model to: {model_file}')

    logger.info(f'Training completed! Best train loss: {best_train_loss:.4f}')
    return test_loader, fusionmodel, len(train_dataset)


def test_fewshot(args, model, test_loader, model_dir, result_dir, fused_dir, vis_dir, ir_dir, logger, train_samples_count):
    ratio_percent = f"{int(args.train_ratio * 100)}%"
    logger.info(f'Start testing (train ratio: {ratio_percent})...')

    best_model_path = model_dir / 'best_model.pth'
    if not best_model_path.exists():
        logger.error(f"Best model not found at {best_model_path}")
        raise FileNotFoundError(f"Best model file {best_model_path} does not exist")

    checkpoint = torch.load(best_model_path)
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        unwanted_keys = ['vis_rmm.body.bias', 'inf_rmm.body.bias']
        filtered_state_dict = {k: v for k, v in state_dict.items() if k not in unwanted_keys}
        model.load_state_dict(filtered_state_dict, strict=False)
    else:
        unwanted_keys = ['vis_rmm.body.bias', 'inf_rmm.body.bias']
        filtered_state_dict = {k: v for k, v in checkpoint.items() if k not in unwanted_keys}
        model.load_state_dict(filtered_state_dict, strict=False)

    model.eval()
    model.cuda()

    all_metrics = []
    metrics_csv = result_dir / 'metrics.csv'

    with torch.no_grad():
        for it, (image_vis, image_ir, name) in enumerate(test_loader):
            img_name = name[0] if isinstance(name, (list, tuple)) else name
            logger.info(f"Processing {img_name} ({it+1}/{len(test_loader)})")

            image_vis = Variable(image_vis).cuda()
            image_ir = Variable(image_ir).cuda()
            save_image_original_color(image_vis, vis_dir / img_name)
            save_image_original_color(image_ir, ir_dir / img_name)

            image_vis_ycrcb = RGB2YCrCb(image_vis)
            image_vis_y = image_vis_ycrcb[:, 0:1, :, :]
            logits = model(image_vis_y, image_ir)
            fusion_ycrcb = torch.cat([logits, image_vis_ycrcb[:, 1:2, :, :],
                                      image_vis_ycrcb[:, 2:3, :, :]], dim=1)
            fusion_image = YCrCb2RGB(fusion_ycrcb)
            fusion_image = torch.clamp(fusion_image, 0, 1)
            save_image_original_color(fusion_image, fused_dir / img_name)

            fused_tensor = prepare_tensor_for_eval(fusion_image).cpu().numpy()
            vis_tensor = prepare_tensor_for_eval(image_vis).cpu().numpy()
            ir_tensor = prepare_tensor_for_eval(image_ir).cpu().numpy()

            metrics_vis = {'MI_VIS': np.nan, 'MSE_VIS': np.nan, 'CC_VIS': np.nan,
                           'PSNR_VIS': np.nan, 'SSIM_VIS': np.nan, 'VIFF_VIS': np.nan}
            metrics_ir = {'MI_IR': np.nan, 'MSE_IR': np.nan, 'CC_IR': np.nan,
                          'PSNR_IR': np.nan, 'SSIM_IR': np.nan, 'VIFF_IR': np.nan}

            try: metrics_vis['MI_VIS'] = mutual_information(fused_tensor, vis_tensor)
            except: pass
            try: metrics_vis['MSE_VIS'] = mean_squared_error(fused_tensor, vis_tensor)
            except: pass
            try: metrics_vis['CC_VIS'] = correlation_coefficient(fused_tensor, vis_tensor)
            except: pass
            try: metrics_vis['PSNR_VIS'] = peak_signal_noise_ratio(fused_tensor, vis_tensor)
            except: pass
            try: metrics_vis['SSIM_VIS'] = perceptual_quality(fused_tensor, vis_tensor)
            except: pass
            try:
                if not args.skip_viff:
                    metrics_vis['VIFF_VIS'] = visual_information_fidelity(fused_tensor, vis_tensor)
            except: pass

            try: metrics_ir['MI_IR'] = mutual_information(fused_tensor, ir_tensor)
            except: pass
            try: metrics_ir['MSE_IR'] = mean_squared_error(fused_tensor, ir_tensor)
            except: pass
            try: metrics_ir['CC_IR'] = correlation_coefficient(fused_tensor, ir_tensor)
            except: pass
            try: metrics_ir['PSNR_IR'] = peak_signal_noise_ratio(fused_tensor, ir_tensor)
            except: pass
            try: metrics_ir['SSIM_IR'] = perceptual_quality(fused_tensor, ir_tensor)
            except: pass
            try:
                if not args.skip_viff:
                    metrics_ir['VIFF_IR'] = visual_information_fidelity(fused_tensor, ir_tensor)
            except: pass

            scd_value = np.nan
            qabf_value = np.nan
            try: scd_value = sum_correlation_differences(fused_tensor, vis_tensor, ir_tensor)
            except: pass
            try: qabf_value = q_abf(fused_tensor, vis_tensor, ir_tensor)
            except: pass

            def safe_average(v1, v2):
                values = [v for v in [v1, v2] if not np.isnan(v) and not np.isinf(v)]
                return np.mean(values) if values else np.nan

            metrics = {
                'ImageName': img_name,
                'MI': safe_average(metrics_vis['MI_VIS'], metrics_ir['MI_IR']),
                'MSE': safe_average(metrics_vis['MSE_VIS'], metrics_ir['MSE_IR']),
                'CC': safe_average(metrics_vis['CC_VIS'], metrics_ir['CC_IR']),
                'PSNR': safe_average(metrics_vis['PSNR_VIS'], metrics_ir['PSNR_IR']),
                'SSIM': safe_average(metrics_vis['SSIM_VIS'], metrics_ir['SSIM_IR']),
                'VIFF': safe_average(metrics_vis['VIFF_VIS'], metrics_ir['VIFF_IR']),
                'SCD': scd_value,
                'Qabf': qabf_value,
            }
            metrics.update(metrics_vis)
            metrics.update(metrics_ir)

            csv_metrics = {}
            for key, value in metrics.items():
                if isinstance(value, float) and np.isnan(value):
                    csv_metrics[key] = 'nan'
                elif isinstance(value, (int, float)):
                    csv_metrics[key] = f"{value:.4f}"
                else:
                    csv_metrics[key] = value
            save_metrics_to_csv(csv_metrics, metrics_csv, append=(it > 0))
            all_metrics.append(metrics)

    if all_metrics:
        avg_metrics = {}
        metric_keys = [key for key in all_metrics[0].keys() if key != 'ImageName']
        for key in metric_keys:
            values = [m.get(key, np.nan) for m in all_metrics
                      if not np.isnan(m.get(key, np.nan)) and not np.isinf(m.get(key, np.nan))]
            avg_metrics[key] = np.mean(values) if values else np.nan

        avg_metrics['ImageName'] = 'Average'
        csv_avg_metrics = {}
        for key, value in avg_metrics.items():
            if isinstance(value, float) and np.isnan(value):
                csv_avg_metrics[key] = 'nan'
            elif isinstance(value, (int, float)):
                csv_avg_metrics[key] = f"{value:.4f}"
            else:
                csv_avg_metrics[key] = value
        save_metrics_to_csv(csv_avg_metrics, metrics_csv, append=True)

        logger.info('=' * 50)
        logger.info(f'Average Test Metrics (train ratio: {ratio_percent}):')
        logger.info(f"MI: {avg_metrics['MI']:.4f} | CC: {avg_metrics['CC']:.4f} | PSNR: {avg_metrics['PSNR']:.4f}")
        logger.info(f"SSIM: {avg_metrics['SSIM']:.4f} | Qabf: {avg_metrics['Qabf']:.4f} | SCD: {avg_metrics['SCD']:.4f}")

        stats_txt = result_dir / 'test_statistics.txt'
        with open(stats_txt, 'w') as f:
            f.write(f"PRFusion Few-shot Test Statistics - Train Ratio: {ratio_percent}\n")
            f.write(f"Model: {args.model_name} | Best Train Loss: {checkpoint['best_train_loss']:.4f}\n")
            f.write(f"Tested Images: {len(all_metrics)} | Train Samples: {train_samples_count}\n\n")
            f.write("Average Metrics:\n" + "-" * 40 + "\n")
            for key in ['MI', 'CC', 'PSNR', 'SSIM', 'VIFF', 'Qabf', 'SCD', 'MSE']:
                f.write(f"{key}: {avg_metrics.get(key, np.nan):.4f}\n")


def main():
    args = parse_args()
    set_seed(args.seed)
    model_dir, log_dir, result_dir, fused_dir, vis_dir, ir_dir = create_experiment_dirs(args)
    setup_logger(log_dir)
    logger = logging.getLogger(__name__)

    ratio_percent = f"{int(args.train_ratio * 100)}%"
    logger.info("=" * 80)
    logger.info(f"PRFusion Few-shot Experiment: {args.exp_name} | {args.model_name} | {ratio_percent}")
    logger.info(f"Epochs: {args.epochs} | Batch: {args.batch_size} | Optimizer: {args.optimizer}")
    logger.info("=" * 80)

    test_loader, model, train_samples_count = train_fewshot(args, logger, model_dir)
    test_fewshot(args=args, model=model, test_loader=test_loader, model_dir=model_dir,
                 result_dir=result_dir, fused_dir=fused_dir, vis_dir=vis_dir, ir_dir=ir_dir,
                 logger=logger, train_samples_count=train_samples_count)

    logger.info(f"\nAll processes completed! Results saved to: {result_dir}")


if __name__ == "__main__":
    main()
