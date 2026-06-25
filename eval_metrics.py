# eval_metrics.py (完整修改版 - 支持 RGB 输入, 处理 256x256 尺寸和 SSIM 问题)

import torch
import numpy as np
from skimage.metrics import structural_similarity as ssim
# from skimage.util import view_as_windows # 未使用，可注释掉
from piq import VIFLoss
import csv
import os
import argparse
from PIL import Image
from torchvision import transforms  # 导入 transforms

# -------------------------
# 指标函数定义
# -------------------------

# --- 添加辅助函数：将 Tensor 转换为灰度 NumPy 数组 ---
def _to_grayscale_numpy(tensor: torch.Tensor) -> np.ndarray:
    """
    将 PyTorch Tensor 图像转换为灰度 NumPy 数组。
    Args:
        tensor (torch.Tensor): 形状为 [H, W, C] (C=3 for RGB) 或 [H, W] 的图像 Tensor。
    Returns:
        np.ndarray: 形状为 [H, W] 的灰度图像 NumPy 数组。
    """
    # 转换为 NumPy (在 CPU 上)
    np_img = tensor.cpu().numpy()
    # 检查维度
    if np_img.ndim == 3:
        # 假设是 [H, W, C] 格式
        if np_img.shape[2] == 3:
            # RGB to Grayscale: 使用标准权重
            gray = np.dot(np_img[...,:3], [0.2989, 0.5870, 0.1140])
            return gray
        elif np_img.shape[2] == 1:
            # 单通道，移除通道维度
            return np.squeeze(np_img, axis=2)
        else:
            # 无法处理的通道数，警告并尝试 squeeze
            print(f"Warning: Unexpected number of channels {np_img.shape[2]} in _to_grayscale_numpy. Attempting squeeze.")
            return np.squeeze(np_img)
    elif np_img.ndim == 2:
        # 已经是灰度图
        return np_img
    else:
        # 无法处理的维度
        raise ValueError(f"Unsupported image dimensions: {np_img.ndim}. Expected 2 or 3.")


# --- 修改 mutual_information ---
def mutual_information(x: torch.Tensor, y: torch.Tensor) -> float:
    """计算两个图像之间的互信息 (Mutual Information)"""
    x_gray = _to_grayscale_numpy(x) # [H, W]
    y_gray = _to_grayscale_numpy(y) # [H, W]
    x_flat = x_gray.flatten()
    y_flat = y_gray.flatten()
    hist_xy, _, _ = np.histogram2d(x_flat, y_flat, bins=256)
    p_xy = hist_xy / float(np.sum(hist_xy))
    p_x = np.sum(p_xy, axis=1)
    p_y = np.sum(p_xy, axis=0)
    p_xy = np.where(p_xy > 0, p_xy, 1e-10)
    p_x = np.where(p_x > 0, p_x, 1e-10)
    p_y = np.where(p_y > 0, p_y, 1e-10)
    mi = np.sum(p_xy * np.log(p_xy / (p_x[:, np.newaxis] * p_y[np.newaxis, :])))
    return float(mi)

# --- 修改 mean_squared_error ---
def mean_squared_error(fused_image: torch.Tensor, target_image: torch.Tensor) -> float:
    """计算均方误差 (Mean Squared Error)"""
    fused_gray = _to_grayscale_numpy(fused_image) # [H, W]
    target_gray = _to_grayscale_numpy(target_image) # [H, W]
    return float(np.mean((fused_gray - target_gray) ** 2))

# --- 修改 correlation_coefficient ---
def correlation_coefficient(fused_image: torch.Tensor, target_image: torch.Tensor) -> float:
    """计算相关系数 (Correlation Coefficient)"""
    fused_gray = _to_grayscale_numpy(fused_image) # [H, W]
    target_gray = _to_grayscale_numpy(target_image) # [H, W]
    return float(np.corrcoef(fused_gray.flatten(), target_gray.flatten())[0, 1])

# --- 修改 peak_signal_noise_ratio ---
def peak_signal_noise_ratio(fused_image: torch.Tensor, target_image: torch.Tensor) -> float:
    """计算峰值信噪比 (Peak Signal-to-Noise Ratio)"""
    fused_gray = _to_grayscale_numpy(fused_image) # [H, W]
    target_gray = _to_grayscale_numpy(target_image) # [H, W]
    mse = np.mean((fused_gray - target_gray) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 1.0
    return float(10.0 * np.log10((max_pixel ** 2) / mse))

# --- 修改 perceptual_quality (处理 SSIM 问题) ---
def perceptual_quality(fused_image: torch.Tensor, target_image: torch.Tensor) -> float:
    """使用 SSIM 计算感知质量 (Perceptual Quality)"""
    fused_gray = _to_grayscale_numpy(fused_image) # [H, W]
    target_gray = _to_grayscale_numpy(target_image) # [H, W]

    # --- 检查图像尺寸并处理 win_size ---
    min_height = min(fused_gray.shape[0], target_gray.shape[0])
    min_width = min(fused_gray.shape[1], target_gray.shape[1])
    min_dim = min(min_height, min_width)

    if min_dim < 3: # 最小边长至少为3才能计算SSIM
        print(f"Warning: Image dimensions too small for SSIM calculation. Min dimension: {min_dim}")
        return np.nan

    # 选择合适的 win_size (奇数且 <= min_dim)
    win_size = min(7, min_dim) # 默认最大7
    if win_size % 2 == 0:
        win_size -= 1 # 如果是偶数，减1变为奇数
    if win_size < 3:
        win_size = 3 # 确保至少为3

    try:
        return float(ssim(fused_gray, target_gray, data_range=1.0, win_size=win_size))
    except Exception as e: # 捕获所有可能的 SSIM 错误
        print(f"SSIM computation failed for images with dims {fused_gray.shape} and {target_gray.shape}: {e}")
        return np.nan
    # ----------------------------------
def visual_information_fidelity(fused_image: torch.Tensor, target_image: torch.Tensor) -> float:
    """计算视觉信息保真度 (Visual Information Fidelity) - 修复维度匹配问题"""
    try:
        from piq import vif_p
    except ImportError:
        print("piq库未安装，请运行: pip install piq")
        return np.nan
    
    def _prepare_for_piq(tensor: torch.Tensor) -> torch.Tensor:
        """
        准备张量用于piq库，确保输出为 [1, 1, H, W] 格式（NCHW）
        """
        # 确保在CPU上处理
        np_tensor = tensor.detach().cpu().numpy()
        
        # 处理不同的输入维度
        if np_tensor.ndim == 4:
            # [B, H, W, C] 或 [B, C, H, W]
            if np_tensor.shape[1] in [1, 3]:  # [B, C, H, W]
                np_tensor = np_tensor[0]  # 移除batch维度
            else:  # [B, H, W, C]
                np_tensor = np_tensor[0]  # 移除batch维度
        
        # 处理RGB转灰度
        if np_tensor.ndim == 3:
            if np_tensor.shape[-1] == 3:  # [H, W, C] RGB
                np_tensor = np.dot(np_tensor[...,:3], [0.2989, 0.5870, 0.1140])
            elif np_tensor.shape[0] == 3:  # [C, H, W] RGB
                np_tensor = np_tensor.transpose(1, 2, 0)  # [H, W, C]
                np_tensor = np.dot(np_tensor[...,:3], [0.2989, 0.5870, 0.1140])
            elif np_tensor.shape[0] == 1:  # [C, H, W] 灰度
                np_tensor = np_tensor.squeeze(0)  # [H, W]
            elif np_tensor.shape[-1] == 1:  # [H, W, C] 灰度
                np_tensor = np_tensor.squeeze(-1)  # [H, W]
        
        # 确保最终是2D灰度图像
        np_tensor = np.squeeze(np_tensor)
        
        # 转换为tensor并添加batch和channel维度 [1, 1, H, W]
        torch_tensor = torch.from_numpy(np_tensor).float()
        if torch_tensor.ndim == 2:
            torch_tensor = torch_tensor.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        
        return torch_tensor
    
    # 准备输入张量
    fused_prepared = _prepare_for_piq(fused_image)      # [1, 1, H, W]
    target_prepared = _prepare_for_piq(target_image)    # [1, 1, H, W]
    
    # 确保两个张量尺寸完全匹配
    if fused_prepared.shape != target_prepared.shape:
        # 调整尺寸以匹配
        min_height = min(fused_prepared.shape[2], target_prepared.shape[2])
        min_width = min(fused_prepared.shape[3], target_prepared.shape[3])
        
        fused_prepared = fused_prepared[:, :, :min_height, :min_width]
        target_prepared = target_prepared[:, :, :min_height, :min_width]
    
    try:
        # 使用vif_p计算VIF分数（值越大越好，范围[0,1]）
        score = vif_p(fused_prepared, target_prepared, data_range=1.0)
        return float(score.item())
    except Exception as e:
        print(f"VIF calculation failed: {e}")
        print(f"Fused shape: {fused_prepared.shape}, Target shape: {target_prepared.shape}")
        
        # 备选方案：使用简化的VIF计算
        try:
            from skimage.metrics import normalized_mutual_information
            fused_np = fused_prepared.squeeze().numpy()
            target_np = target_prepared.squeeze().numpy()
            nmi = normalized_mutual_information(fused_np, target_np)
            return float(nmi)
        except:
            return np.nan
# # --- 修改 visual_information_fidelity ---
# def visual_information_fidelity(fused_image: torch.Tensor, target_image: torch.Tensor) -> float:
#     """计算视觉信息保真度 (Visual Information Fidelity)"""
#     from piq import VIFLoss # 确保导入
#     def _prepare_for_piq(tensor: torch.Tensor) -> torch.Tensor:
#         np_tensor = tensor.cpu().numpy()
#         if np_tensor.ndim == 3 and np_tensor.shape[2] == 3:
#              np_tensor = np.dot(np_tensor[...,:3], [0.2989, 0.5870, 0.1140])
#         torch_tensor = torch.from_numpy(np_tensor).unsqueeze(0).unsqueeze(0) # [1, 1, H, W]
#         return torch_tensor.to(tensor.device)
#     fused_prepared = _prepare_for_piq(fused_image)      # [1, 1, H, W]
#     target_prepared = _prepare_for_piq(target_image)    # [1, 1, H, W]
#     vif_loss = VIFLoss(data_range=1.)
#     try:
#         score = vif_loss(fused_prepared, target_prepared)
#         return float(score.item())
#     except Exception as e:
#         print(f"VIF calculation failed: {e}")
#         return np.nan # 使用 nan 表示计算失败

# --- 修改 q_abf ---
def q_abf(fused_image: torch.Tensor, source1_image: torch.Tensor, source2_image: torch.Tensor) -> float:
    """计算 QABF 指标"""
    def gradient(img):
        return np.sqrt(np.gradient(img)[0] ** 2 + np.gradient(img)[1] ** 2)
    fused_gray = _to_grayscale_numpy(fused_image)       # [H, W]
    source1_gray = _to_grayscale_numpy(source1_image)   # [H, W]
    source2_gray = _to_grayscale_numpy(source2_image)   # [H, W]
    grad_fused = gradient(fused_gray)
    grad_source1 = gradient(source1_gray)
    grad_source2 = gradient(source2_gray)
    q_map = np.zeros_like(grad_fused)
    mask1 = grad_source1 > grad_source2
    mask2 = ~mask1
    q_map[mask1] = grad_fused[mask1] / (grad_source1[mask1] + 1e-8)
    q_map[mask2] = grad_fused[mask2] / (grad_source2[mask2] + 1e-8)
    q_map = np.clip(q_map, 0, 1)
    return float(np.mean(q_map))

# --- 修改 sum_correlation_differences ---
def sum_correlation_differences(fused_image: torch.Tensor, source1_image: torch.Tensor, source2_image: torch.Tensor) -> float:
    """计算 SCD 指标"""
    cc_fused_source1 = correlation_coefficient(fused_image, source1_image)
    cc_fused_source2 = correlation_coefficient(fused_image, source2_image)
    cc_source1_source2 = correlation_coefficient(source1_image, source2_image)
    scd = abs(cc_fused_source1 - cc_source1_source2) + abs(cc_fused_source2 - cc_source1_source2)
    return float(scd)

# -------------------------
# 工具函数
# -------------------------

# --- 修改 load_image ---
def load_image(image_path, target_size=(256, 256)): # 修改默认尺寸为 256x256
    """
    加载图像，调整尺寸，并确保是 RGB 模式。
    如果原始图像是灰度图，会自动转换为 RGB。
    Args:
        image_path (str): 图像文件路径。
        target_size (tuple): 目标尺寸 (宽, 高)。 # 修改注释
    Returns:
        torch.Tensor or None: 归一化到 [0, 1] 的 Tensor (形状 [H, W, C])，失败则返回 None。
    """
    try:
        img = Image.open(image_path) 
        if img.mode != 'RGB':
            # print(f"Converting image {image_path} from mode {img.mode} to RGB.")
            img = img.convert('RGB') 
        if img.size != target_size:
            # print(f"Resizing {image_path} from {img.size} to {target_size}")
            resize_transform = transforms.Resize(target_size)
            img = resize_transform(img)
        
        img_array = np.array(img) # Shape: [H, W, 3] for RGB
        img_array_normalized = img_array / 255.0 
        # 返回 [H, W, C] 格式的 Tensor
        img_tensor = torch.tensor(img_array_normalized, dtype=torch.float32) 
        return img_tensor
    except Exception as e:
        print(f"加载图像失败: {image_path}")
        print(f"错误信息: {e}")
        return None

def save_metrics_to_csv(metrics_dict, csv_path, append=False):
    directory = os.path.dirname(csv_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
    mode = 'a' if append else 'w'
    header = not append or not os.path.exists(csv_path)
    with open(csv_path, mode, newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=metrics_dict.keys())
        if header:
            writer.writeheader()
        writer.writerow(metrics_dict)

# -------------------------
# 主处理流程
# -------------------------
def process_images(fused_dir, source1_dir, source2_dir, csv_path, device):
    fused_files = sorted(os.listdir(fused_dir))
    source1_files = sorted(os.listdir(source1_dir))
    source2_files = sorted(os.listdir(source2_dir))
    if len(fused_files) != len(source1_files) or len(fused_files) != len(source2_files):
        print("错误: 不同目录中的图像文件数量不匹配")
        return
    metric_names = ['MI', 'MSE', 'CC', 'PSNR', 'SSIM', 'VIFF', 'SCD', 'Qabf']
    all_metrics = {name: [] for name in metric_names}
    for idx, (fused_file, source1_file, source2_file) in enumerate(zip(fused_files, source1_files, source2_files)):
        fused_path = os.path.join(fused_dir, fused_file)
        source1_path = os.path.join(source1_dir, source1_file)
        source2_path = os.path.join(source2_dir, source2_file)
        print(f"\n处理第 {idx+1} 张图像：{fused_file}")
        
        # --- 加载图像，确保尺寸一致并处理为 RGB ---
        # 修改 target_size 为 (256, 256)
        fused_image = load_image(fused_path, target_size=(256, 256))      # 融合图 (RGB)
        source1_image = load_image(source1_path, target_size=(256, 256))  # MRI 图 (转为 RGB)
        source2_image = load_image(source2_path, target_size=(256, 256))  # SPECT 图 (转为 RGB)
        # ---------------------------------------------------------------

        if fused_image is None or source1_image is None or source2_image is None:
            print("图像加载失败，跳过此组图像")
            continue
        if fused_image.shape != source1_image.shape or fused_image.shape != source2_image.shape:
            print(f"图像尺寸不匹配 - 跳过：{fused_file}")
            print(f"  融合图尺寸: {fused_image.shape}, MRI尺寸: {source1_image.shape}, SPECT尺寸: {source2_image.shape}")
            continue

        # --- 设备处理 ---
        # 简化设备选择逻辑
        compute_device = torch.device('cuda' if device == 'cuda' and torch.cuda.is_available() else 'cpu')
        if compute_device.type == 'cuda':
             print("使用GPU进行计算")
        else:
             print("使用CPU进行计算")
        fused_image = fused_image.to(compute_device)
        source1_image = source1_image.to(compute_device)
        source2_image = source2_image.to(compute_device)
        # -----------------

        print("计算评估指标...")
        metrics_source1 = {
            'MI_Source1': mutual_information(fused_image, source1_image),
            'MSE_Source1': mean_squared_error(fused_image, source1_image),
            'CC_Source1': correlation_coefficient(fused_image, source1_image),
            'PSNR_Source1': peak_signal_noise_ratio(fused_image, source1_image),
            'SSIM_Source1': perceptual_quality(fused_image, source1_image),
            'VIFF_Source1': visual_information_fidelity(fused_image, source1_image),
        }
        metrics_source2 = {
            'MI_Source2': mutual_information(fused_image, source2_image),
            'MSE_Source2': mean_squared_error(fused_image, source2_image),
            'CC_Source2': correlation_coefficient(fused_image, source2_image),
            'PSNR_Source2': peak_signal_noise_ratio(fused_image, source2_image),
            'SSIM_Source2': perceptual_quality(fused_image, source2_image),
            'VIFF_Source2': visual_information_fidelity(fused_image, source2_image),
        }
        scd_value = sum_correlation_differences(fused_image, source1_image, source2_image)
        qabf_value = q_abf(fused_image, source1_image, source2_image)
        
        # --- 汇总指标，处理可能的 nan 值 ---
        def safe_average(v1, v2):
             # 如果任一值为 nan，则结果为 nan
             if np.isnan(v1) or np.isnan(v2):
                 return np.nan
             return (v1 + v2) / 2.0

        metrics = {
            'MI': safe_average(metrics_source1['MI_Source1'], metrics_source2['MI_Source2']),
            'MSE': safe_average(metrics_source1['MSE_Source1'], metrics_source2['MSE_Source2']),
            'CC': safe_average(metrics_source1['CC_Source1'], metrics_source2['CC_Source2']),
            'PSNR': safe_average(metrics_source1['PSNR_Source1'], metrics_source2['PSNR_Source2']),
            'SSIM': safe_average(metrics_source1['SSIM_Source1'], metrics_source2['SSIM_Source2']),
            'VIFF': safe_average(metrics_source1['VIFF_Source1'], metrics_source2['VIFF_Source2']),
            'SCD': scd_value, # SCD 和 Qabf 本身不涉及 source1/source2 平均
            'Qabf': qabf_value,
        }
        # -----------------------------

        # 打印当前图像的指标 (保留 nan 值的显示)
        print("\n评估结果:")
        for metric, value in metrics.items():
            if np.isnan(value):
                print(f"{metric}: nan")
            else:
                print(f"{metric}: {value:.4f}")

        # 写入CSV
        metrics_with_name = {'ImageName': fused_file}
        # 格式化数值，nan 保持为 'nan' 字符串，数字保留4位小数
        formatted_metrics = {k: ('nan' if np.isnan(v) else f"{v:.4f}") for k, v in metrics.items()}
        metrics_with_name.update(formatted_metrics)
        save_metrics_to_csv(metrics_with_name, csv_path, append=True)

        # 存储用于平均值计算 (保留 nan，np.mean 会自动处理)
        for key, value in metrics.items():
            all_metrics[key].append(value)

    # 输出总体平均值
    print("\n所有图像处理完成，计算总体平均指标...")
    # 使用 nanmean 计算平均值，忽略 nan 值
    avg_metrics = {k: np.nanmean(v) for k, v in all_metrics.items()} 
    for metric, value in avg_metrics.items():
        if np.isnan(value):
             print(f"{metric}: nan")
        else:
             print(f"{metric}: {value:.4f}")
    
    # 追加写入总体平均值到 CSV
    avg_row = {'ImageName': 'Average'}
    # 格式化平均值
    formatted_avg_metrics = {k: ('nan' if np.isnan(v) else f"{v:.4f}") for k, v in avg_metrics.items()}
    avg_row.update(formatted_avg_metrics)
    
    fieldnames = ['ImageName'] + list(formatted_avg_metrics.keys())
    with open(csv_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        # 检查文件是否为空，如果为空则写入 header
        if os.stat(csv_path).st_size == 0:
            writer.writeheader()
        writer.writerow(avg_row)
    print(f"\n📊 总体平均指标已追加写入 {csv_path}")

# -------------------------
# 主函数入口
# -------------------------
def main():
    parser = argparse.ArgumentParser(description='图像融合质量评估（支持 RGB 输入, 256x256）')
    parser.add_argument('--fused', type=str, required=True, help='Fused images directory')
    parser.add_argument('--source1', type=str, required=True, help='Source image 1 directory (e.g., MRI)')
    parser.add_argument('--source2', type=str, required=True, help='Source image 2 directory (e.g., SPECT)')
    parser.add_argument('--csv', type=str, default='./metrics.csv', help='Output CSV path')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'], help='计算设备')
    args = parser.parse_args()
    for path in [args.fused, args.source1, args.source2]:
        if not os.path.exists(path):
            print(f"错误: 目录不存在 - {path}")
            return
    process_images(args.fused, args.source1, args.source2, args.csv, args.device)

if __name__ == "__main__":
    main()