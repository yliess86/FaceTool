# -*- coding: utf-8 -*-
"""facecrop/detector.py

The file provides a face detector and outputs boxes given frames.
Their should be only one face per frame.
If multiple are present, the boxer will return None for the given frame.
"""
from facenet_pytorch import MTCNN
from typing import Iterable
from typing import List

import numpy as np

class FaceDetector:
    """Face Detector

    Face Detector is a Wrapper for the MTCNN pytorch model to allow one face
    detection only returns one box per frame.

    Attributes:
        mtcnn {MTCNN} -- pytorch face detection model

    Keyword Arguments:
        margin {int} -- Margin to add to bounding box, in terms of pixels in 
            the final image. Note that the application of the margin differs 
            slightly from the davidsandberg/facenet repo, which applies the 
            margin to the original image before resizing, making the margin
            dependent on the original image size (this is a bug in 
            davidsandberg/facenet). (default: {4})
        factor {float} -- Factor used to create a scaling pyramid of face 
            sizes. (default: {0.6})
        keep_all {bool} -- If True, all detected faces are returned, in the 
            order dictated by the select_largest parameter. If a save_path is 
            specified, the first face is saved to that path and the remaining 
            faces are saved to <save_path>1, <save_path>2 etc. 
            (default: {False})
        device {str} -- device chosen for inference, cuda or cpu
            (default: {"cpu"})

    Examples:
        >>> from facecrop.boxer import FaceDetector
        >>>
        >>>
        >>> face_detector = FaceDetector(device="cuda)
        >>> boxes = face_detector(frames)
    """

    def __init__(
        self,
        margin: int = 4,
        factor: float = 0.6,
        keep_all: bool = False,
        device: str = "cpu"
    ) -> None:
        self.mtcnn = MTCNN(
            margin=margin, factor=factor, keep_all=keep_all, device=device
        )

    def __call__(self, frames: Iterable[np.ndarray]) -> List[np.ndarray]:
        """Call

        Arguments:
            frames {Iterable[np.ndarray]} -- frames to process

        Returns:
            List[np.ndarray] -- boxes for each given frame (None for multiple 
                or none faces)
        """
        boxes, _ = self.mtcnn.detect(frames)
        return [
            [None] * 4 if box is None or len(box) > 1 else box[0]
            for box in boxes
        ]