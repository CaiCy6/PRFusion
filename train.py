#!/usr/bin/python
# -*- encoding: utf-8 -*-
import os
import argparse
import datetime
import time
import logging
from pathlib import Path
import warnings

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, ReduceLROnPlateau

from dataset import Fusion_dataset
from logger import setup_logger
from Fusionnet import MACTFusion
from Networks.FusionNet import UNetFusionModel as PRFusion
from Networks.OursA1 import UNetFusionModel as PRFusionA1
from Networks.OursA2 import UNetFusionModel as PRFusionA2
from Networks.OursA3 import UNetFusionModel as PRFusionA3
from Networks.OursA4 import UNetFusionModel as PRFusionA4
from loss import Fusionloss

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
    parser = argparse.ArgumentParser(description='PRFusion: Medical Image Fusion Training')

    parser.add_argument('--model_name', '-M', type=str, default='MACTFusion',
                        choices=list(MODEL_REGISTRY.keys()),
                        help='Model name')
    parser.add_argument('--output_channels', '-oc', type=int, default=1,
                        help='Number of output channels (default: 1)')

    parser.add_argument('--batch_size', '-B', type=int, default=4)
    parser.add_argument('--epochs', '-E', type=int, default=50)
    parser.add_argument('--lr_start', '-lr', type=float, default=0.001)
    parser.add_argument('--lr_decay_type', type=str, default='step',
                        choices=['step', 'cosine', 'plateau', 'poly'])
    parser.add_argument('--lr_decay_rate', '-lrd', type=float, default=0.9)
    parser.add_argument('--lr_decay_epoch', '-lre', type=int, default=5)
    parser.add_argument('--lr_min', type=float, default=1e-4)
    parser.add_argument('--optimizer', type=str, default='adam',
                        choices=['adam', 'adamw', 'sgd'])

    parser.add_argument('--gpu', '-G', type=int, default=0)
    parser.add_argument('--num_workers', '-j', type=int, default=12)
    parser.add_argument('--seed', '-s', type=int, default=42)

    parser.add_argument('--method', '-method', type=str, default='MACTFusion')
    parser.add_argument('--num_folds', '-k', type=int, default=5)
    parser.add_argument('--exp_name', '-exp', type=str, default='SPECT-MRI',
                        choices=['CT-MRI', 'PET-MRI', 'SPECT-MRI'])

    parser.add_argument('--data_dir', type=str, required=True,
                        help='Root path to Havard-Medical-Image-Fusion-Datasets-main')
    parser.add_argument('--log_dir', type=str, default='./logs')
    parser.add_argument('--model_dir', type=str, default='./models')

    return parser.parse_args()


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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
    out = temp.reshape(
        list(input_im.size())[0],
        list(input_im.size())[2],
        list(input_im.size())[3],
        3,
    ).transpose(1, 3).transpose(2, 3)
    return out


def YCrCb2RGB(input_im):
    im_flat = input_im.transpose(1, 3).transpose(1, 2).reshape(-1, 3)
    mat = torch.tensor([[1.0, 1.0, 1.0],
                        [1.403, -0.714, 0.0],
                        [0.0, -0.344, 1.773]]).cuda()
    bias = torch.tensor([0.0 / 255, -0.5, -0.5]).cuda()
    temp = (im_flat + bias).mm(mat).cuda()
    out = temp.reshape(
        list(input_im.size())[0],
        list(input_im.size())[2],
        list(input_im.size())[3],
        3,
    ).transpose(1, 3).transpose(2, 3)
    return out


def create_experiment_dirs(args):
    main_model_dir = Path(args.model_dir) / args.exp_name / args.method
    main_model_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(args.log_dir) / args.exp_name / args.method
    log_dir.mkdir(parents=True, exist_ok=True)
    return main_model_dir, log_dir


def get_optimizer(model, args):
    if args.optimizer == 'adam':
        return torch.optim.Adam(model.parameters(), lr=args.lr_start,
                                betas=(0.9, 0.999), weight_decay=1e-4)
    elif args.optimizer == 'adamw':
        return torch.optim.AdamW(model.parameters(), lr=args.lr_start,
                                 betas=(0.9, 0.999), weight_decay=0.01)
    elif args.optimizer == 'sgd':
        return torch.optim.SGD(model.parameters(), lr=args.lr_start,
                               momentum=0.9, weight_decay=1e-4)


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


def train_fusion(args, fold, logger, model_dir):
    torch.cuda.set_device(args.gpu)

    fold_dir = model_dir / f'fold_{fold}'
    fold_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Initializing model {args.model_name} for fold {fold}")
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

    logger.info(f"Loading datasets for fold {fold}")
    train_dataset = Fusion_dataset('train', fold=fold, num_folds=args.num_folds,
                                   data_dir_vis=data_dir_vis, data_dir_ir=data_dir_ir)
    val_dataset = Fusion_dataset('val', fold=fold, num_folds=args.num_folds,
                                 data_dir_vis=data_dir_vis, data_dir_ir=data_dir_ir)

    logger.info(f"Training dataset length: {len(train_dataset)}")
    logger.info(f"Validation dataset length: {len(val_dataset)}")

    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers,
                              pin_memory=True, drop_last=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers,
                            pin_memory=True, drop_last=True)

    criteria_fusion = Fusionloss()
    best_val_loss = float('inf')
    st = glob_st = time.time()

    logger.info(f'Start training fold {fold}...')

    for epo in range(args.epochs):
        fusionmodel.train()
        train_losses = []

        for it, (image_vis, image_ir, name) in enumerate(train_loader):
            image_vis = Variable(image_vis).cuda()
            image_vis_ycrcb = RGB2YCrCb(image_vis)
            image_vis_y = image_vis_ycrcb[:, 0:1, :, :]
            image_ir = Variable(image_ir).cuda()

            logits = fusionmodel(image_vis_y, image_ir)

            fusion_ycrcb = torch.cat(
                [logits, image_vis_ycrcb[:, 1:2, :, :], image_vis_ycrcb[:, 2:3, :, :]], dim=1)
            fusion_image = YCrCb2RGB(fusion_ycrcb)
            fusion_image = torch.clamp(fusion_image, 0, 1)

            optimizer.zero_grad()
            loss_fusion, loss_in, ssim_loss, loss_grad = criteria_fusion(
                image_vis=image_vis_ycrcb, image_ir=image_ir,
                generate_img=logits, i=fold, labels=None)
            loss_total = loss_fusion
            loss_total.backward()
            optimizer.step()
            train_losses.append(loss_total.item())

            ed = time.time()
            t_intv, glob_t_intv = ed - st, ed - glob_st
            now_it = len(train_loader) * epo + it + 1
            total_iters = len(train_loader) * args.epochs
            eta = int((total_iters - now_it) * (glob_t_intv / now_it))
            eta_str = str(datetime.timedelta(seconds=eta))

            if now_it % 10 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                msg = (f'Fold {fold} | Epoch {epo+1}/{args.epochs} | Step {it+1}/{len(train_loader)} | '
                       f'Loss: {loss_total.item():.4f} | In: {loss_in.item():.4f} | '
                       f'Grad: {loss_grad.item():.4f} | SSIM: {ssim_loss.item():.4f} | '
                       f'LR: {current_lr:.6f} | ETA: {eta_str}')
                logger.info(msg)
                st = ed

        fusionmodel.eval()
        val_losses = []
        with torch.no_grad():
            for it, (image_vis, image_ir, name) in enumerate(val_loader):
                image_vis = Variable(image_vis).cuda()
                image_vis_ycrcb = RGB2YCrCb(image_vis)
                image_vis_y = image_vis_ycrcb[:, 0:1, :, :]
                image_ir = Variable(image_ir).cuda()
                logits = fusionmodel(image_vis_y, image_ir)
                loss_fusion, _, _, _ = criteria_fusion(
                    image_vis=image_vis_ycrcb, image_ir=image_ir,
                    generate_img=logits, i=fold, labels=None)
                val_losses.append(loss_fusion.item())

        avg_train_loss = sum(train_losses) / len(train_losses)
        avg_val_loss = sum(val_losses) / len(val_losses)

        if args.lr_decay_type == 'plateau':
            scheduler.step(avg_val_loss)
        else:
            scheduler.step()

        logger.info(f'Fold {fold} | Epoch {epo+1} | Train Loss: {avg_train_loss:.4f} | '
                    f'Val Loss: {avg_val_loss:.4f} | LR: {optimizer.param_groups[0]["lr"]:.6f}')

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model_file = fold_dir / f'best_model.pth'
            torch.save({
                'epoch': epo + 1,
                'model_state_dict': fusionmodel.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_val_loss,
                'args': args,
            }, model_file)
            logger.info(f'Saved best model to: {model_file}')

        last_model_file = fold_dir / 'last_model.pth'
        torch.save({
            'epoch': epo + 1,
            'model_state_dict': fusionmodel.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_val_loss,
            'args': args,
        }, last_model_file)

    logger.info(f'Finished training fold {fold}')
    logger.info('-' * 80)


def main():
    args = parse_args()
    set_seed(args.seed)
    model_dir, log_dir = create_experiment_dirs(args)
    setup_logger(log_dir)
    logger = logging.getLogger(__name__)

    logger.info("=" * 80)
    logger.info("PRFusion Training Configuration")
    logger.info(f"Experiment: {args.exp_name} | Method: {args.method} | Model: {args.model_name}")
    logger.info(f"Folds: {args.num_folds} | Epochs: {args.epochs} | Batch: {args.batch_size}")
    logger.info(f"Optimizer: {args.optimizer} | LR: {args.lr_start} ({args.lr_decay_type})")
    logger.info(f"Data: {args.data_dir} | GPU: {args.gpu}")
    logger.info("=" * 80)

    for fold in range(args.num_folds):
        logger.info(f"\nStarting fold {fold}/{args.num_folds - 1}")
        train_fusion(args, fold, logger, model_dir)
        logger.info(f"Completed fold {fold}")

    logger.info("\nAll folds completed!")
    logger.info(f"Results saved to: {model_dir}")


if __name__ == "__main__":
    main()
