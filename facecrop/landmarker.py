# -*- coding: utf-8 -*-
"""facecrop.landmarker

The file provides a wrapper to apply dlib 68 facial landmark detection
on multiple frames while taking advantage of threading to make faster
inferences.
    * dlib: http://dlib.net/
"""
from itertools import islice
from multiprocessing.pool import ThreadPool
from tqdm import tqdm
from typing import Tuple
from urllib import request

import bz2
import dlib
import moviepy.editor as mpe
import numpy as np
import os


class FaceLandmarker:
    """Face Landmarker
    
    Face Landmarker infere 68 facial landmarks for single person videos
    providing boxes (can use facecrop.detector or others). The class wrappes
    the dlib predictor an uses ThreadPool to accelrate inference. The process
    is batched for better memory usage.
        * dlib: http://dlib.net/face_landmark_detection.py.html

    Arguments:
        n_processes {int} -- number of process for multi threading pool
        batch_size {int} -- batch size correspond to the amount of frames
            loaded at a diven iteration
    """

    def __init__(self, n_process: int, batch_size: int) -> None:
        self.n_process = n_process
        self.batch_size = batch_size
        self.predictor = dlib.shape_predictor(self._predictor)

    @property
    def _predictor(self) -> str:
        """Donwload and return dlib Predictor Weights File Path"""
        # Define Bases for path and url
        base = "http://dlib.net/files"
        current = os.path.abspath(__file__)
        parent = os.path.dirname(current)
        root = os.path.join(parent, "model")

        # Define file names
        bz2_file = "shape_predictor_68_face_landmarks.dat.bz2"
        dat_file = "shape_predictor_68_face_landmarks.dat"
        
        # Define file paths and url
        url = f"{base}/{bz2_file}"
        bz2_path = f"{root}/{bz2_file}"
        dat_path = f"{root}/{dat_file}"

        if not os.path.isfile(dat_path):
            os.makedirs(root, exist_ok=True)

            # Read Compressed file
            request.urlretrieve(url, bz2_path)
            with open(bz2_path, "rb") as fh:
                compressed = fh.read()
                decompressed = bz2.decompress(compressed)
            os.remove(bz2_path)
            
            # Write Uncompressed file
            with open(dat_path, "wb") as fh:
                fh.write(decompressed)

        return dat_path


    def __call__(self, path: str, boxes: np.ndarray) -> np.ndarray:
        """Call

        Arguments:
            path {str} -- path to the video
            boxes {np.ndarray} -- array of size [N, 5] where N is the amount of
                valid frames (frame with detected face) and 5 correspond to
                [frame_idx, x, y, w, h] of the box.

        Returns:
            np.ndarray -- array of size [N, 68, 2] where N is the amount of
                valid frames (frame with detected face), 68 corresponds to the
                68 regressed facial landmarks and 2 to its x and y position.
        """
        video = mpe.VideoFileClip(path)
        frame_iter = video.iter_frames()

        # Get Number of Frame
        n_frames = int(np.floor(video.fps * video.duration))
        
        results = []
        iterator = range(0, n_frames, self.batch_size)
        for b in tqdm(iterator, desc="Face Landmarker"):
            # Get batch_size Frames
            frames = list(islice(frame_iter, self.batch_size))
            end = b + len(frames)
            
            # Select Accepted Frames
            mask = (boxes[:, 0] >= b) & (boxes[:, 0] < end) # Accepted Mask
            idxs = boxes[mask, 0] - b                       # Accepted Indexes
            bboxes = boxes[mask]                            # Batch Boxes
            
            def predict(box: np.ndarray) -> np.ndarray:
                i, x, y, w, h = box                   # Box
                x0, y0 = x - w // 2, y - w // 2       # Top Left Coords
                x1, y1 = x + w // 2, y + w // 2       # Bottom Right Coords
                rect = dlib.rectangle(x0, y0, x1, y1) # Rectangle
                
                frame = frames[i - b]                 # Frame of Box
                shape = self.predictor(frame, rect)   # Face Shape
                
                # Convert Shape to 68 Landmarks
                num_parts = shape.num_parts
                to_coords = lambda c: [c.x, c.y]
                prediction = np.array([
                    to_coords(shape.part(i)) for i in range(num_parts)
                ], dtype=int)

                return prediction
        
            # Predic Landmarks using Multiple Threads
            with ThreadPool(self.n_process) as pool:
                landmarks = np.array(list(pool.imap(predict, bboxes)))
            
            if len(landmarks):
                results.append(landmarks)

        return np.concatenate(results)