# -*- coding: utf-8 -*-
"""facecrop.masker

The file provides a wrapper for a segmentation mask model. This can be used to
improve the facial landmark detection. The mode segments the people present on
the frame and provides a mask to retriev the person and thus remove the
background. The model used is based on MobileNet UNet for its speed and has
been JIT traced from the following repository:
    * Paper: https://arxiv.org/pdf/1505.04597.pdf
    * Code: https://github.com/thuyngch/Human-Segmentation-PyTorch

The segmentation is not perfect but enough for its purpose here.
"""
from PIL import Image
from torchvision.transforms import Compose
from torchvision.transforms import Normalize
from torchvision.transforms import Resize
from torchvision.transforms import ToTensor

import numpy as np
import pkg_resources as pr
import torch
import torch.nn as nn
import torch.nn.functional as F


class GuassianBlur(nn.Module):
    """GuassianBlur

    GaussianBlur by Gaussian Kernel 2D Convolution.
    The implementation is a straight implementation of the solution provided
    by Tetratrio from the pytorch forums at the following address:
        https://discuss.pytorch.org/t/
        is-there-anyway-to-do-gaussian-filtering-for-an-image-2d-3d-in-pytorch/
        12351/3

    Arguments:
        kernel_size {int} -- size of the gaussian kernel
        sigma {float} -- sigma of the guassian pdf
        channels {int} -- number of channels of the input image
    """
    
    def __init__(self, kernel_size: int, sigma: float, channels: int) -> None:
        super(GuassianBlur, self).__init__()
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.channels = channels
        
        x_coord = torch.arange(kernel_size)
        x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
        y_grid = x_grid.t()
        xy_grid = torch.stack([x_grid, y_grid], dim=-1)

        mean = (kernel_size - 1.0) * 0.5
        variance = sigma ** 2

        c1 = 1.0 / (2.0 * np.pi * variance)
        c2 = 2.0 * variance
        kernel = c1 * torch.exp(-torch.sum((xy_grid - mean) ** 2, dim=-1) / c2)
        kernel = kernel / torch.sum(kernel)

        kernel = kernel.view(1, 1, kernel_size, kernel_size)
        kernel = kernel.repeat(channels, 1, 1, 1)
        
        self.kernel = nn.Parameter(kernel, requires_grad=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Apply Gaussian Blur to the given inputs"""
        return F.conv2d(inputs, self.kernel, groups=self.channels)


class BackgroundMasker:
    """Background Masker

    Background Masker computes a segmentation mask for given frames, thus
    allowing to differentiate a person from the background. The implementation
    is a wrapper of the following model JIT traces for shipping simplicity:
        * Paper: https://arxiv.org/pdf/1505.04597.pdf
        * Code: https://github.com/thuyngch/Human-Segmentation-PyTorch

    Arguments:
        device {str} -- inference device ("cpu" or "cuda")
    """

    def __init__(self, device: str) -> None:
        self.device = device
        self.model = torch.jit.load(self._model).to(device)
        self.blur = GuassianBlur(10, 2.0, 2).to(device)
        self.transform = Compose([
            lambda x: Image.fromarray(x),
            Resize((224, 224)),
            ToTensor(),
            Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

    @property
    def _model(self) -> str:
        """Return the model path"""
        return pr.resource_filename("facecrop", "model/masknet.pt")

    @torch.no_grad()
    def __call__(self, inputs: np.ndarray) -> np.ndarray:
        """Call

        Compute the segmentation masks for given set of frames.

        Arguments:
            inputs {np.ndarray} -- input frames

        Returns:
            np.ndarray -- output masks of size [N, H, W] where N is the number
                of frames, H the height and W the width of these frames.
        """
        b, h, w, c = inputs.shape
        size = h, w
        
        # Transform Inputs and Compute Segmentation
        x = torch.cat([self.transform(img).unsqueeze(0) for img in inputs])
        y = self.model(x.to(self.device))
        
        # Extract Mask and Resize
        masks = F.softmax(y, dim=1)
        masks = self.blur(masks)
        masks = F.interpolate(
            masks, size=size, mode="bilinear", align_corners=True
        )
        mask = masks[:, 1]
        mask = mask.cpu().numpy()

        return mask