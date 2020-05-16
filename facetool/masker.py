# -*- coding: utf-8 -*-
"""facetool.masker

The file provides a wrapper for a segmentation mask model. The mode segments
humans present on the frame and provides a mask to retrieve the person and thus
remove the background. The model used is based on MobileNet UNet for its speed
and has been JIT traced from the following repository:
    * Paper: https://arxiv.org/pdf/1505.04597.pdf
    * Code: https://github.com/thuyngch/Human-Segmentation-PyTorch

The segmentation is not perfect but enough for its purpose here.
"""
from facetool.utils import GaussianFilter
from itertools import chain
from itertools import islice
from PIL import Image
from torchvision.transforms import Compose
from torchvision.transforms import Normalize
from torchvision.transforms import Resize
from torchvision.transforms import ToTensor
from tqdm import tqdm
from typing import Any
from typing import Iterator
from typing import List

import moviepy.editor as mpe
import numpy as np
import pkg_resources as pr
import torch
import torch.nn as nn
import torch.nn.functional as F


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

    def __init__(self, batch_size: int, device: str) -> None:
        self.batch_size = batch_size
        self.device = device
        self.model = torch.jit.load(self._model).to(device)
        self.blur = GaussianFilter(2.0).to(device)
        self.transform = Compose([
            lambda x: Image.fromarray(x),
            Resize((224, 224)),
            ToTensor(),
            Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

    @property
    def _model(self) -> str:
        """Return the model path"""
        return pr.resource_filename("facetool", "model/masknet.pt")

    def __call__(self, path: str, dest: str) -> None:
        """Generate Mask for a given Video and Produce a Mask mp4 Clip"""
        video = mpe.VideoFileClip(path)
        video_frames = video.iter_frames()
        n_frames = int(np.floor(video.fps * video.duration))

        # Generator to Process Frames
        def generator() -> Iterator[np.ndarray]:
            for _ in range(0, n_frames, self.batch_size):
                frames = list(islice(video_frames, self.batch_size))
                masks = self.process(frames)
                for mask in masks:
                    yield mask
        
        # Produce Video
        mask_frames = generator()
        mask_video = mpe.VideoClip(make_frame=lambda t: next(mask_frames))
        mask_video = mask_video.set_fps(video.fps)
        mask_video = mask_video.set_duration(n_frames / video.fps)
        mask_video.write_videofile(dest)

    @torch.no_grad()
    def process(self, inputs: List[np.ndarray]) -> np.ndarray:
        """Process One Batch

        Arguments:
            inputs {List[np.ndarray]} -- input frames

        Returns:
            List[np.ndarray] -- output masks of size [N, H, W, C] where N is
                the number of frames, H the height, W the width, and C the
                number of channels for these frames.
        """
        h, w, c = inputs[0].shape
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
        mask = mask.unsqueeze(-1).repeat(1, 1, 1, 3).mul_(255.0)
        mask = mask.cpu().numpy().astype(np.uint8)

        return mask