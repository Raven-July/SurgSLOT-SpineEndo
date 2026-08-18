import torch
import torch.nn as nn
import torch.nn.functional as F


class GaussianFilter2D(nn.Module):
    """Apply a depthwise Gaussian filter to tensors shaped [B, N, H, W]."""

    def __init__(self, sigma=1.0, kernel_size=5, constrained_th=0.0):
        super().__init__()
        if kernel_size % 2 == 0:
            kernel_size += 1
        self.sigma = float(sigma)
        self.kernel_size = int(kernel_size)
        self.constrained_th = float(constrained_th)

        center = self.kernel_size // 2
        xy = torch.arange(self.kernel_size, dtype=torch.float32) - center
        xx, yy = torch.meshgrid(xy, xy, indexing="ij")
        kernel = torch.exp(-(xx**2 + yy**2) / (2 * (self.sigma**2)))
        kernel = kernel / kernel.sum()

        # Register the kernel as [1, 1, K, K] so it follows device and DDP moves.
        self.register_buffer(
            "kernel", kernel.unsqueeze(0).unsqueeze(0), persistent=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype == torch.bool:
            x = x.float()
        _, num_channels, _, _ = x.shape

        # Expand depthwise weights to match the input dtype, device, and channels.
        weight = self.kernel.to(dtype=x.dtype, device=x.device)
        if weight.shape[0] != num_channels:
            weight = weight.expand(
                num_channels, 1, self.kernel_size, self.kernel_size
            ).contiguous()

        output = F.conv2d(
            x,
            weight,
            bias=None,
            stride=1,
            padding=self.kernel_size // 2,
            groups=num_channels,
        )

        if self.constrained_th > 0:
            output = self.constrained_gaussian_filter(
                output, self.constrained_th
            )
        return output

    @staticmethod
    def constrained_gaussian_filter(
        x: torch.Tensor, threshold: float = 0.1
    ) -> torch.Tensor:
        center = x.new_tensor(0.5)
        low = x.new_tensor(0.5 - threshold)
        high = x.new_tensor(0.5 + threshold)
        mask = (x - center).abs() < threshold
        return torch.where(mask, torch.where(x >= center, high, low), x)
