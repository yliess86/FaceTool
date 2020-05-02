# -*- coding: utf-8 -*-
"""facetool.annotator

The files provides a Face Annotator in charge of combining the result of the
Face Detector and Face Landmark in a single pandas DataFrame. This Face
Annotator is the API built to be used by the end user. 
"""
from facetool.detector import FaceDetector
from facetool.landmarker import FaceLandmarker
from tqdm import tqdm
from typing import Tuple

import numpy as np
import pandas as pd


class FaceAnnotator:
    """Face Annotator
    
    Face Annotator combine the boxes of a Face Detector and the landmarks of a
    Face Landmarker in a single DataFrame that can later be used for analysis,
    further computation, or visualization.

    Arguments:
        dbatch_size {int} -- batch size for the detector inference
        lbatch_size {int} -- batch size for the landmarker frame loading
        size {Tuple[int, int]} -- resize frames for detector
        n_process {int} -- number of threads used by the landmarker
        device {str} -- device to run the detector on ("cpu" or "cuda")
    """

    def __init__(
        self, dbatch_size: int, lbatch_size: int, size: Tuple[int, int],
        n_process: int, device: str,
    ) -> None:
        self.detector = FaceDetector(device, dbatch_size, size)
        self.landmarker = FaceLandmarker(n_process, lbatch_size)

    def __call__(self, path: str) -> pd.DataFrame:
        """Call

        Combines boxes and landmarks in a single DataFrame.

        Arguments:
            path {str} -- path to the video to be annotated

        Returns:
            pd.DataFrame -- dataframe containing boxes and landmarks
                informations of size [N, 1 + 4 + 68 * 2] where:
                    * N -> valid frames (frame with face detected)
                    * 1 -> frame_idx
                    * 4 -> box_x, box_y, box_w, box_h
                    * 68 * 2 -> landmark_i_x, landmark_i_y for i in range(68)
        """
        boxes = self.detector(path)
        landmarks = self.landmarker(path, boxes)

        N, B = boxes.shape        # Frames x Boxe Data
        N, L, P = landmarks.shape  # Frames x Landmark x Coords

        # Combine Data
        data = np.zeros((N, B + L * P), dtype=int)
        pbar = tqdm(enumerate(zip(boxes, landmarks)), desc="Face Annotator")
        for i, (box, landmark) in pbar:
            data[i, 0:(4 + 1)] = box       # t, x, y, w, h ->  5
            data[i, 5::2] = landmark[:, 0] # x_0 .... x_68 -> 68
            data[i, 6::2] = landmark[:, 1] # y_0 .... y_68 -> 68

        # Helpers to Name Landmarks Columns
        lpos = lambda k: "x" if k == 0 else "y"
        lname = lambda j, k: f"landmark_{j + 1}_{lpos(k)}"

        # Landmarks Column Names
        names = ["frame_idx", "box_x", "box_y", "box_w", "box_h"]
        names += [lname(j, k) for j in range(L) for k in range(P)]

        # Create DataFrame
        df = pd.DataFrame(data=data, columns=names)

        return df