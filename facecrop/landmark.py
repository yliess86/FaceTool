# -*- coding: utf-8 -*-
"""facecrop/landmark.py

The file provides a basic Wrapper to the face alignement library to allow
face landmark detection in batch as this functionnality is not provided by the
library by default. The face alignement implements FAN from the following
paper: https://arxiv.org/pdf/1712.02765.pdf, using the PyTorch library and can
thus be executed taking advantage of a cuda enabled GPU.
"""
from face_alignment.models import FAN
from PIL import Image
from torch.utils.model_zoo import load_url
from torchvision.transforms import Compose
from torchvision.transforms import Resize
from torchvision.transforms import ToTensor

import numpy as np
import torch


BASE = "https://www.adrianbulat.com/downloads/python-fan"
WEIGHTS = f"{BASE}/2DFAN4-11f355bf06.pth.tar"


class FaceLandmarker:
    """Face Landmarker

    Face Landmaker is a Wrapper to add Batch inference for the FAN 68 facial
    landmark detector. FAN paper: https://arxiv.org/pdf/1712.02765.pdf

    Attributes:
        device {str} -- device to run the model on, "cpu" or "cuda"
            (default: {"cpu"})
        fan {FAN} -- pytorch module containing the model
        transforms {Compose} -- torchvision transform pipeline

    Keyword Arguments:
        device {str} -- device to run the model on, "cpu" or "cuda"
    """

    def __init__(self, device: str = "cpu") -> None:
        weights = load_url(WEIGHTS, map_location=lambda s, l: s)
        self.device = device
        self.fan = FAN(4)
        self.fan.load_state_dict(weights)
        self.fan.to(self.device)
        self.fan.eval()
        self.transforms = Compose([Resize((256, 256)), ToTensor()])

    @torch.no_grad()
    def __call__(self, faces: np.ndarray) -> np.ndarray:
        """Compute 68 face landmarks given cropped faces: (B, N, 2)"""
        def flip(tensor: torch.Tensor) -> torch.Tensor:
            # Negative slicinf is not available yet
            return torch.flip(tensor, dims=(-2, ))

        h, w, c = faces[0].shape

        faces = torch.stack([
            self.transforms(Image.fromarray(face))
            for face in faces
        ]).to(self.device)

        # Flipped for redundancy. May increase stabilty.
        heatmaps = self.fan(faces)[-1] + flip(self.fan(flip(faces))[-1])
        heatmaps.mul_(0.5)

        B, N, W, H = heatmaps.size()

        preds = torch.argmax(heatmaps.view(B, N, W * H), -1)
        preds = preds.view(B, N, 1).float().repeat(1, 1, 2).cpu()
        preds[..., 0].apply_(lambda x: x % H)
        preds[..., 1].div_(W).floor_()

        for b in range(B):
            for n in range(N):
                hm_ = heatmaps[b, n, :]
                pX, pY = int(preds[b, n, 0]) - 1, int(preds[b, n, 1]) - 1
                if pX > 0 and pX < (W - 1) and pY > 0 and pY < (H - 1):
                    diff = torch.FloatTensor([
                        hm_[pY,     pX + 1] - hm_[pY,     pX - 1],
                        hm_[pY + 1, pX    ] - hm_[pY - 1, pX    ],
                    ])
                    preds[b, n].add_(diff.sign_().mul_(0.25))
        
        preds[..., 0].div_(H).mul_(256).mul_(w / 256)
        preds[..., 1].div_(W).mul_(256).mul_(h / 256)

        return preds.int()