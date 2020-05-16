# -*- coding: utf-8 -*-
"""facetool.xdoger

The file provides a countour filter based on the famous xDoG (eXtended
Difference of Gaussians). Combined to the mask provided by the FaceMasker, this
allow to compute drawing like countours of the subject present in the frames.
    * Paper: https://www.kyprianidis.com/p/cag2012/winnemoeller-cag2012.pdf
    * Code: https://github.com/alexpeattie/xdog-sketch
"""
from facetool.utils import GaussianFilter
from itertools import islice
from torchvision.transforms import ToTensor
from tqdm import tqdm
from typing import Any
from typing import Iterator
from typing import List

import moviepy.editor as mpe
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GrayScale(nn.Module):
    """Gray Scale

    GrayScale transform any batch of RGB tensors into grayscale tensors based
    on the channel ratio provided in: https://en.wikipedia.org/wiki/Grayscale
    """

    def __init__(self) -> None:
        super(GrayScale, self).__init__()
        self.r = 0.2126
        self.g = 0.7152
        self.b = 0.0722

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward

        Arguments:
            inputs {torch.Tensor} -- batch of rgb tensors [B, 3, H, W]

        Returns:
            torch.Tensor -- converted batch grayscale tensors [B, 1, H, W]
        """
        out  = inputs[:, 0] * self.r
        out += inputs[:, 1] * self.g
        out += inputs[:, 2] * self.b
        return out.unsqueeze(1)


class XDoG(nn.Module):
    """XDoG

    XDoG or eXtended Difference of Gaussians is a countour filtering algorithm
    based on the difference of the input image Blurred by 2D GaussianBlur at
    two different intensities. The resulting countours resemble inked drawings.
        * Paper: https://www.kyprianidis.com/p/cag2012/winnemoeller-cag2012.pdf
        * Code: https://github.com/alexpeattie/xdog-sketch

    Keyword Arguments:
        sigma1 {float} -- sigma of the first gaussian blur filter
            (default: {1.80})
        sigma2 {float} -- sigma of the second gaussian blur filter
            (default: {2.50})
        sharpen {float} -- sharpens the gaussians before computing difference
            (default: {28.0})
        phi {float} -- phi parameter for soft thresholding
            (default: {1.051})
        eps {float} -- epsilon parameter for soft thresholding
            (default: {10.70})
    """

    def __init__(
        self, sigma1: float = 1.80, sigma2: float = 2.50,
        sharpen: float = 28.0, phi: float = 1.051, eps: float = 10.70
    ) -> None:
        super(XDoG, self).__init__()
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.sharpen = sharpen
        self.phi = phi
        self.eps = eps
        
        self.gray = GrayScale()
        self.filter1 = GaussianFilter(self.sigma1)
        self.filter2 = GaussianFilter(self.sigma2)

    def forward(self, inputs: torch.Tensor) -> None:
        """Forward

        Applies xDoG on images batch. Will first convert them to grayscale.

        Arguments:
            inputs {torch.Tensor} -- input batch of rgb images [B, 3, H, W]

        Returns:
            torch.Tensor -- output batch of countour images [B, 1, H, W]
        """
        gray = self.gray(inputs)

        img1 = self.filter1(gray)
        img2 = self.filter2(gray)

        scaled_diff = (self.sharpen + 1) * img1 - self.sharpen * img2
        sharpened = (gray * scaled_diff) * 255.0
        mask = ((gray * scaled_diff) - self.eps) > 0

        soft = 1 + torch.tanh(self.phi * (sharpened - self.eps))
        res = ((~mask * soft) + mask) * 255.0
        res_scaled = res / res.max()

        return res_scaled


class XDoGer:
    """XDoGer

    XDoG or eXtended Difference of Gaussians is a countour filtering algorithm
    based on the difference of the input image Blurred by 2D GaussianBlur at
    two different intensities. The resulting countours resemble inked drawings.
        * Paper: https://www.kyprianidis.com/p/cag2012/winnemoeller-cag2012.pdf
        * Code: https://github.com/alexpeattie/xdog-sketch

    XDoGer uses XDoG to turn a full video into a inked countour version.

    Argument:
        batch {int} -- batch size
        device {str} -- device to use when computing, "cuda" or "cpu"

    Keyword Arguments:
        sigma1 {float} -- sigma of the first gaussian blur filter
            (default: {1.80})
        sigma2 {float} -- sigma of the second gaussian blur filter
            (default: {2.50})
        sharpen {float} -- sharpens the gaussians before computing difference
            (default: {28.0})
        phi {float} -- phi parameter for soft thresholding
            (default: {1.051})
        eps {float} -- epsilon parameter for soft thresholding
            (default: {10.70})
    """

    def __init__(
        self, batch_size: int, device: str,
        sigma1: float = 1.80, sigma2: float = 2.50,
        sharpen: float = 28.0, phi: float = 1.051, eps: float = 10.70
    ) -> None:
        self.batch_size = batch_size
        self.device = device
        self.xdog = XDoG(sigma1, sigma2, sharpen, phi, eps).to(device)
        self.transform = ToTensor()

    def __call__(self, path: str, dest: str) -> None:
        """Generate Inked Contour for a given Video and Produce an mp4 Clip"""
        video = mpe.VideoFileClip(path)
        video_frames = video.iter_frames()
        n_frames = int(np.floor(video.fps * video.duration))

        # Generator to Process Frames
        def generator() -> Iterator[np.ndarray]:
            for _ in range(0, n_frames, self.batch_size):
                frames = list(islice(video_frames, self.batch_size))
                contours = self.process(frames)
                for contour in contours:
                    yield contour

        # Produce Video
        cont_frames = generator()
        cont_video = mpe.VideoClip(make_frame=lambda t: next(cont_frames))
        cont_video = cont_video.set_fps(video.fps)
        cont_video = cont_video.set_duration(n_frames / video.fps)
        cont_video.write_videofile(dest)

    @torch.no_grad()
    def process(self, inputs: List[np.ndarray]) -> np.ndarray:
        """Process One Batch

        Arguments:
            inputs {List[np.ndarray]} -- input frames

        Returns:
            List[np.ndarray] -- output inked contour of size [N, H, W, C]
                where N is the number of frames, H the height, W the width,
                and C the number of channels for these frames.
        """
        x = torch.cat([self.transform(img).unsqueeze(0) for img in inputs])
        
        y = self.xdog(x.to(self.device))
        y = y.permute(0, 2, 3, 1).repeat(1, 1, 1, 3).mul_(255.0)
        y = y.cpu().numpy().astype(np.uint8)
        
        return y