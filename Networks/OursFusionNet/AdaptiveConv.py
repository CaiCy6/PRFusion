import torch
import torch.nn as nn
import torch.nn.functional as F

class AdaptiveDepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(AdaptiveDepthwiseSeparableConv, self).__init__()

        # 输入参数
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride

        # 选择一个自适应的卷积核大小 (3, 5, 或 7)
        self.kernel_size = self.select_kernel_size()

        # 深度可分离卷积
        self.depthwise = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, 
                                   kernel_size=self.kernel_size, stride=stride, padding=self.kernel_size // 2, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, bias=False)

    def select_kernel_size(self):
        """
        自动选择卷积核大小，可以根据条件自定义规则
        在这里，我们用随机选择作为示例，但可以根据实际需求进行修改
        """
        kernel_sizes = [3, 5, 7]
        # 这里可以加入更复杂的规则来选择合适的卷积核大小，例：根据输入图像的大小、计算资源等
        return kernel_sizes[torch.randint(0, len(kernel_sizes), (1,)).item()]

    def forward(self, x):
        # 深度卷积
        x = self.depthwise(x)
        # 点卷积
        x = self.pointwise(x)
        return x

# 测试自适应卷积模块
if __name__ == "__main__":
    # 假设输入图像为 [batch_size, in_channels, height, width]
    input_tensor = torch.rand(1, 32, 256, 256)  # [1, 32, 256, 256] 输入图像 (32通道，256x256大小)

    # 创建自适应深度可分离卷积模块实例
    model = AdaptiveDepthwiseSeparableConv(in_channels=32, out_channels=32)

    # 前向传播
    output_tensor = model(input_tensor)

    # 输出结果形状
    print(f"Output shape: {output_tensor.shape}")
