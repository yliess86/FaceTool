# -*- coding: utf-8 -*-
"""facetool.utils

The files provides common utilities between modules present in the repository.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GaussianFilter(nn.Module):
    """Gaussian Filter

    GaussianFilter applies Gaussian Blur with 2D conv gaussian kernel.

    Arguments:
        sigma {float} -- sigma of the gaussian filter (filter size will 
            automaticaly be computed based on this value)
        kernel_size {int} -- kernel size for the gaussian filter
            (default: {None})
    """

    def __init__(self, sigma: float, kernel_size: int = None) -> None:
        super(GaussianFilter, self).__init__()
        self.kernel_size = (
            int(max(np.round(sigma * 3) * 2 + 1, 3)) if kernel_size is None
            else kernel_size
        )
        self.sigma = sigma

        kernel_x = self._kernel(self.kernel_size, self.sigma)
        kernel_y = self._kernel(self.kernel_size, self.sigma)
        kernel = kernel_x.unsqueeze(-1) @ kernel_y.unsqueeze(-1).t()
        kernel = kernel.view(1, 1, self.kernel_size, self.kernel_size)

        self.kernel = nn.Parameter(kernel, requires_grad=False)
        self.padding = [(self.kernel_size - 1) // 2] * 2

    def _kernel(self, kernel_size: int, sigma: float) -> torch.Tensor:
        """Compute Kenerl Size based on Sigma Value"""
        gauss = lambda x: -(x - kernel_size // 2) ** 2 / (2 * sigma ** 2)
        kernel = torch.stack([
            torch.exp(torch.tensor(gauss(x))) for x in range(kernel_size)
        ])
        kernel = kernel / kernel.sum()
        return kernel

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward

        Arguments:
            inputs {torch.Tensor} -- batch of images tensors

        Returns:
            torch.Tensor -- batch of blurred images tensors
        """
        b, c, h, w = inputs.size()
        params = { "padding": self.padding, "stride": 1, "groups": c }
        kernel = self.kernel.repeat(c, 1, 1, 1)
        return F.conv2d(inputs, kernel, **params)