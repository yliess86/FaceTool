# -*- coding: utf-8 -*-
"""facecrop.detector

The file provides a multi stage face detector to output a face box for given
frames. The process takes advantage of the GPU and Batch Inference.
The model used is MTCNN.
    * Paper: https://arxiv.org/pdf/1604.02878.pdf
    * Implementation: https://github.com/timesler/facenet-pytorch
"""
from facenet_pytorch import MTCNN
from itertools import islice
from PIL import Image
from tqdm import tqdm
from torchvision.transforms import Compose
from torchvision.transforms import Resize
from torchvision.transforms import ToTensor
from typing import Tuple

import moviepy.editor as mpe
import numpy as np
import torch


class FaceDetector:
    """Face Detector

    Face Detector infere face boxes for single person videos. It takes
    advantage of Batch Inference and Cuda optimization. The underlying model
    used is MTCNN.
        * Paper: https://arxiv.org/pdf/1604.02878.pdf
        * Implementation: https://github.com/timesler/facenet-pytorch

    Output Format: [frame_idx, x, y, w, h] instead of [x0, y0, x1, y1]

    Arguments:
        device {str} -- device to run inference on ("cpu" or "cuda")
        batch_size {int} -- batch size for batch inference
        size {Tuple[int, int]} -- resize input to a given size
    """

    def __init__(
        self, device: str, batch_size: int, size: Tuple[int, int]
    ) -> None:
        self.device = device
        self.batch_size = batch_size
        self.size = size

        ToPIL = lambda x: Image.fromarray(x)
        self.transform = Compose([ToPIL, Resize((size[1], size[0]))])
        self.mtcnn = MTCNN(device=device)
        self.mtcnn.select_largest = True

    def __call__(self, path: str) -> np.ndarray:
        """Call

        Compute the face boxe for each frame of a given video.

        Arguments:
            path {str} -- path to the video

        Returns:
            np.ndarray -- array of size [N, 5] where N is the amount of valid
                frames (frame with detected face) and 5 correspond to
                [frame_idx, x, y, w, h] of the box.
        """
        video = mpe.VideoFileClip(path)
        frame_iter = video.iter_frames()

        # Get Number of Frame and Scale
        n_frames = int(np.floor(video.fps * video.duration))
        scale = video.w / self.size[0], video.h / self.size[1]

        results = []
        iterator = range(0, n_frames, self.batch_size)
        for b in tqdm(iterator, desc="Face Detector"):
            # Get batch_size Frames and Detect Boxes
            frames = islice(frame_iter, self.batch_size)
            frames = map(self.transform, frames)
            boxes, _ = self.mtcnn.detect(list(frames), landmarks=False)

            # Handle Empty and Multiple Boxes
            is_empty = lambda box: box is None or not len(box)
            extract = lambda box: [0] * 4 if is_empty(box) else box[0]
            boxes = np.array([extract(box) for box in boxes], dtype=np.float32)

            # Rescale to Original Size
            np.multiply(boxes[:, 0], scale[0], out=boxes[:, 0])
            np.multiply(boxes[:, 2], scale[0], out=boxes[:, 2])
            np.multiply(boxes[:, 1], scale[1], out=boxes[:, 1])
            np.multiply(boxes[:, 3], scale[1], out=boxes[:, 3])

            # Turn [x0, y0, x1, y1] to [x, y, w, h]
            result = np.zeros_like(boxes, dtype=np.float32)
            np.subtract(boxes[:, 0], boxes[:, 2], out=result[:, 2]) # widht
            np.subtract(boxes[:, 1], boxes[:, 3], out=result[:, 3]) # height
            np.abs(result[:, 2], out=result[:, 2])                  # width
            np.abs(result[:, 3], out=result[:, 3])                  # height
            np.multiply(result[:, 2], 0.5, out=result[:, 0])        # x
            np.multiply(result[:, 3], 0.5, out=result[:, 1])        # y
            np.add(result[:, 0], boxes[:, 0], out=result[:, 0])     # x
            np.add(result[:, 1], boxes[:, 1], out=result[:, 1])     # y
            result = result.astype(int)

            # Add Frame Indexes
            idxs = np.arange(0, len(result), dtype=int) + b
            result = np.insert(result, 0, idxs, axis=-1)

            # Clean None Faces (0 is now frame)
            mask_pos  = (result[:, 1] != 0) & (result[:, 2] != 0)
            mask_size = (result[:, 3] != 0) & (result[:, 4] != 0)
            mask = mask_pos & mask_size

            results.append(result[mask])

        return np.concatenate(results)