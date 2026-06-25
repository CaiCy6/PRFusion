#!/usr/bin/python
# -*- encoding: utf-8 -*-
import os
import argparse
import logging
from pathlib import Path
from PIL import Image
import numpy as np
import pandas as pd
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
import warnings
from datetime import datetime

from dataset import Fusion_dataset
from Fusionnet import MACTFusion
from Networks.FusionNet import UNetFusionModel as PRFusion
from Networks.OursA1 import UNetFusionModel as PRFusionA1
from Networks.OursA2 import UNetFusionModel as PRFusionA2
from Networks.OursA3 import UNetFusionModel as PRFusionA3
from Networks.OursA4 import UNetFusionModel as PRFusionA4
from eval_metrics import (
    mutual_information, mean_squared_error, correlation_coefficient,
    peak_signal_noise_ratio, perceptual_quality, visual_information_fidelity,
    q_abf, sum_correlation_differences, save_metrics_to_csv
)

warnings.filterwarnings('ignore')

MODEL_REGISTRY = {
    'MACTFusion': MACTFusion,
    'PRFusion': PRFusion,
    'PRFusionA1': PRFusionA1,
    'PRFusionA2': PRFusionA2,
    'PRFusionA3': PRFusionA3,
    'PRFusionA4': PRFusionA4,
}


def parse_args():
    parser = argparse.ArgumentParser(description='PRFusion: Medical Image Fusion Testing')

    parser.add_argument('--model_name', '-M', type=str, default='MACTFusion',
                        choices=list(MODEL_REGISTRY.keys()))
    parser.add_argument('--output_channels', '-oc', type=int, default=1)

    parser.add_argument('--gpu', '-G', type=int, default=0)
    parser.add_argument('--num_workers', '-j', type=int, default=4)
    parser.add_argument('--batch_size', '-B', type=int, default=1)

    parser.add_argument('--data_dir', type=str, required=True,
                        help='Root path to Havard-Medical-Image-Fusion-Datasets-main')
    parser.add_argument('--model_dir', type=str, default='./models')
    parser.add_argument('--result_dir', type=str, default='./results')
    parser.add_argument('--exp_name', '-exp', type=str, default='PET-MRI',
                        choices=['CT-MRI', 'PET-MRI', 'SPECT-MRI'])
    parser.add_argument('--method', '-method', type=str, default='MACTFusion')

    parser.add_argument('--num_folds', '-k', type=int, default=5)
    parser.add_argument('--test_split', type=str, default='val', choices=['val', 'train'])
    parser.add_argument('--skip_viff', action='store_true')
    parser.add_argument('--test_folds', type=int, nargs='+', default=None)

    return parser.parse_args()


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


def setup_logger(log_path):
    logger = logging.getLogger('test_logger')
    logger.setLevel(logging.INFO)
    if logger.handlers:
        logger.handlers.clear()
    file_handler = logging.FileHandler(log_path)
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


def calculate_metrics(fused_image, vis_image, ir_image, img_name, args):
    try:
        fused_tensor = prepare_tensor_for_eval(fused_image)
        vis_tensor = prepare_tensor_for_eval(vis_image)
        ir_tensor = prepare_tensor_for_eval(ir_image)

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
        return metrics
    except Exception as e:
        logging.error(f"Metrics calculation failed for {img_name}: {e}")
        return {'ImageName': img_name, 'MI': np.nan, 'MSE': np.nan, 'CC': np.nan,
                'PSNR': np.nan, 'SSIM': np.nan, 'VIFF': np.nan, 'SCD': np.nan, 'Qabf': np.nan}


def find_available_models(args):
    model_base_dir = Path(args.model_dir) / args.exp_name / args.method
    available_folds = []
    for fold in range(args.num_folds):
        fold_dir = model_base_dir / f'fold_{fold}'
        if fold_dir.exists():
            model_files = list(fold_dir.glob('best_model.pth')) + list(fold_dir.glob('last_model.pth'))
            if model_files:
                available_folds.append(fold)
    return available_folds


def get_dataset_paths(args):
    exp_path_map = {
        "CT-MRI": ("CT-MRI/CT", "CT-MRI/MRI"),
        "PET-MRI": ("PET-MRI/PET", "PET-MRI/MRI"),
        "SPECT-MRI": ("SPECT-MRI/SPECT", "SPECT-MRI/MRI")
    }
    vis_subdir, ir_subdir = exp_path_map[args.exp_name]
    return os.path.join(args.data_dir, vis_subdir), os.path.join(args.data_dir, ir_subdir)


def test_fold(args, fold, logger):
    torch.cuda.set_device(args.gpu)

    fold_model_dir = Path(args.model_dir) / args.exp_name / args.method / f'fold_{fold}'
    fold_result_dir = Path(args.result_dir) / args.exp_name / args.method / f'fold_{fold}'
    fused_dir = fold_result_dir / 'fused_images'
    vis_dir = fold_result_dir / 'vis_images'
    ir_dir = fold_result_dir / 'ir_images'

    for dir_path in [fused_dir, vis_dir, ir_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading model for fold {fold}")
    model = MODEL_REGISTRY[args.model_name]()

    model_files = list(fold_model_dir.glob('best_model.pth'))
    if not model_files:
        model_files = list(fold_model_dir.glob('last_model.pth'))
    if not model_files:
        logger.error(f"No model found for fold {fold}")
        return None

    model_path = model_files[0]
    checkpoint = torch.load(model_path)

    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        unwanted_keys = ['vis_rmm.body.bias', 'inf_rmm.body.bias']
        filtered_state_dict = {k: v for k, v in state_dict.items() if k not in unwanted_keys}
        model.load_state_dict(filtered_state_dict, strict=False)
    else:
        unwanted_keys = ['vis_rmm.body.bias', 'inf_rmm.body.bias']
        filtered_state_dict = {k: v for k, v in checkpoint.items() if k not in unwanted_keys}
        model.load_state_dict(filtered_state_dict, strict=False)

    model = model.cuda()
    model.eval()

    data_dir_vis, data_dir_ir = get_dataset_paths(args)
    test_dataset = Fusion_dataset(args.test_split, fold=fold, num_folds=args.num_folds,
                                  data_dir_vis=data_dir_vis, data_dir_ir=data_dir_ir)
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size,
                             shuffle=False, num_workers=args.num_workers,
                             pin_memory=True, drop_last=False)

    logger.info(f"Testing fold {fold} with {len(test_dataset)} images")
    metrics_csv = fold_result_dir / 'metrics.csv'
    all_metrics = []

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

            fusion_ycrcb = torch.cat(
                (logits, image_vis_ycrcb[:, 1:2, :, :], image_vis_ycrcb[:, 2:3, :, :]), dim=1)
            fusion_image = YCrCb2RGB(fusion_ycrcb)
            fusion_image = torch.clamp(fusion_image, 0, 1)
            save_image_original_color(fusion_image, fused_dir / img_name)

            metrics = calculate_metrics(fusion_image, image_vis, image_ir, img_name, args)
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

    fold_avg_metrics = None
    if all_metrics:
        fold_avg_metrics = calculate_average_metrics(all_metrics)
        fold_avg_metrics['ImageName'] = f'Fold_{fold}_Average'
        csv_avg_metrics = {}
        for key, value in fold_avg_metrics.items():
            if isinstance(value, float) and np.isnan(value):
                csv_avg_metrics[key] = 'nan'
            elif isinstance(value, (int, float)):
                csv_avg_metrics[key] = f"{value:.4f}"
            else:
                csv_avg_metrics[key] = value
        save_metrics_to_csv(csv_avg_metrics, metrics_csv, append=True)

        logger.info(f"Average metrics for fold {fold}:")
        for key, value in fold_avg_metrics.items():
            if key != 'ImageName' and not np.isnan(value):
                logger.info(f"  {key}: {value:.4f}")

    logger.info(f"Completed fold {fold}")
    return fold_avg_metrics


def calculate_average_metrics(all_metrics):
    avg_metrics = {}
    metric_keys = [key for key in all_metrics[0].keys() if key != 'ImageName']
    for key in metric_keys:
        values = [m.get(key, np.nan) for m in all_metrics
                  if not np.isnan(m.get(key, np.nan)) and not np.isinf(m.get(key, np.nan))]
        avg_metrics[key] = np.mean(values) if values else np.nan
    return avg_metrics


def calculate_folds_statistics(folds_metrics, logger):
    if not folds_metrics:
        return None
    metrics_dict = {}
    for fold_idx, fold_metrics in enumerate(folds_metrics):
        if fold_metrics is None:
            continue
        for key, value in fold_metrics.items():
            if key != 'ImageName' and not np.isnan(value) and not np.isinf(value):
                if key not in metrics_dict:
                    metrics_dict[key] = []
                metrics_dict[key].append(value)
    stats = {}
    for metric, values in metrics_dict.items():
        if len(values) > 0:
            stats[f'{metric}_mean'] = np.mean(values)
            stats[f'{metric}_std'] = np.std(values)
            if np.mean(values) > 0:
                stats[f'{metric}_cv'] = stats[f'{metric}_std'] / stats[f'{metric}_mean'] * 100
            else:
                stats[f'{metric}_cv'] = np.nan
    return stats


def save_method_statistics(stats, method_dir, args, logger):
    if not stats:
        return
    stats_txt = method_dir / 'method_statistics.txt'
    with open(stats_txt, 'w') as f:
        f.write(f"PRFusion Method Statistics - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model: {args.model_name} | Experiment: {args.exp_name}\n")
        f.write("=" * 80 + "\n\n")
        main_metrics = ['MI', 'MSE', 'CC', 'PSNR', 'SSIM', 'VIFF', 'SCD', 'Qabf']
        for metric in main_metrics:
            if f'{metric}_mean' in stats:
                mean_val = stats[f'{metric}_mean']
                std_val = stats[f'{metric}_std']
                f.write(f"{metric}: {mean_val:.4f} +/- {std_val:.4f}\n")
    logger.info("\nMethod Statistics (Mean +/- Std):")
    for metric in main_metrics:
        if f'{metric}_mean' in stats:
            logger.info(f"{metric}: {stats[f'{metric}_mean']:.4f} +/- {stats[f'{metric}_std']:.4f}")


def main():
    args = parse_args()
    method_dir = Path(args.result_dir) / args.exp_name / args.method
    method_dir.mkdir(parents=True, exist_ok=True)
    log_path = method_dir / 'test_log.txt'
    logger = setup_logger(log_path)

    available_folds = find_available_models(args)
    test_folds = [f for f in (args.test_folds or available_folds) if f in available_folds]

    logger.info("=" * 80)
    logger.info(f"PRFusion Testing: {args.exp_name} | {args.method} | {args.model_name}")
    logger.info(f"Folds to test: {test_folds} | Split: {args.test_split}")
    logger.info("=" * 80)

    if not test_folds:
        logger.error("No valid model folds found!")
        return

    folds_metrics = []
    for fold in test_folds:
        logger.info(f"\nStarting testing for fold {fold}")
        fold_avg_metrics = test_fold(args, fold, logger)
        if fold_avg_metrics:
            folds_metrics.append(fold_avg_metrics)

    if folds_metrics:
        stats = calculate_folds_statistics(folds_metrics, logger)
        if stats:
            save_method_statistics(stats, method_dir, args, logger)

    logger.info("\nTesting completed!")
    logger.info(f"Results saved to: {method_dir}")


if __name__ == "__main__":
    main()
