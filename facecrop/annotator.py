# -*- coding: utf-8 -*-
"""facecrop/annotator.py

The provides an FaceAnnotator using the Video and FaceDetector wrapper.
The FaceAnnotator computes detected face boxes annotations for videos and
classify each valid portion of the video by a sequence tag.

Examples:
    x {float} -- box x position
    y {float} -- box y position
    box_size {float} -- box size (max(max width, max height) of the sequence)
    mark_[i]_[x/y] {float} -- landmark id i, x or y component of the face
    frame_w {int} -- frame size width (resized when processed)
    frame_h {int} -- frame size height (resized when processed)

    +----------------------------------------------------------------+
    |time|    x|    y| box_size|mark_[i]_x|mark_[i]_y|frame_w|frame_h|
    |---:|----:|----:|--------:|---------:|---------:|------:|------:|
    |0.16|325.0|500.0|    230.0|     400.0|     528.0|   1280|    720|
    | ...|  ...|  ...|      ...|       ...|       ...|    ...|    ...|
    +----------------------------------------------------------------+
"""
from facecrop.detector import FaceDetector
from facecrop.landmark import FaceLandmarker
from facecrop.video import Video
from itertools import islice
from tqdm import tqdm
from typing import Tuple
from typing import Iterator

import numpy as np
import pandas as pd


class FaceAnnotator:
    """Box Annotator

    FaceAnnotator computes detected face boxes annotations for videos and
    classify each valid portion of the video by a sequence tag.

    Attributes:
        size {Tuple[int, int]} -- frame will be resized
        boxer {FaceDetector} -- face detector for boxing
        names {List[str]} -- initial column name when boxing

    Keyword Arguments:
        size {Tuple[int, int]} -- frame will be resized
        device {str} -- device for boxing, cpu or cuda (default: {"cpu"})

    Examples:
        x {float} -- box x position
        y {float} -- box y position
        box_size {float} -- box size (max(max width, max height) of the 
            sequence)
        mark_[i]_[x/y] {float} -- landmark id i, x or y component of the face
        frame_w {int} -- frame size width (resized when processed)
        frame_h {int} -- frame size height (resized when processed)

        >>> from facecrop.annotator import FaceAnnotator
        >>>
        >>>
        >>> box_annotator = FaceAnnotator((1280, 720), device)
        >>> df = box_annotator(path, batch_size)
        >>> df.head()
        +----------------------------------------------------------------+
        |time|    x|    y| box_size|mark_[i]_x|mark_[i]_y|frame_w|frame_h|
        |---:|----:|----:|--------:|---------:|---------:|------:|------:|
        |0.16|325.0|500.0|    230.0|     400.0|     528.0|   1280|    720|
        | ...|  ...|  ...|      ...|       ...|       ...|    ...|    ...|
        +----------------------------------------------------------------+
    """

    def __init__(
        self, size: Tuple[int, int], device: str = "cpu"
    ) -> None:
        self.size = size
        self.boxer = FaceDetector(device=device)
        self.landmarker = FaceLandmarker(device=device)
        self.names = ["x0", "y0", "x1", "y1"]

    def __call__(
        self, path: str, batch_size: int,
        n: int = 24, overshoot: float = 1.2, max_batch: int = None
    ) -> pd.DataFrame:
        """Call

        Arguments:
            path {str} -- path to the video
            batch_size {int} -- batch size to process faster
            overshoot {float} -- make box bigger by a factor (default: {1.2})
                usefull to get the hole face including hair and neck
            n {int} -- number of frame used for smoothing (default: {24})
            max_batch {int} -- max batch number to process (default: {None})
                usefull for testing

        Returns:
            pd.DataFrame -- annotations
        """
        video = Video(path, size=self.size)
        dt, n_batch = self._infos(video, batch_size)
        if max_batch:
            n_batch = min(n_batch, max_batch)

        with video as v:
            df = self._process_boxes(v, n_batch, batch_size)
        df = self._classify_boxes(df)
        df = self._transform_boxes(df, dt)
        df = self._clean_boxes(df)
        df = self._unify_boxes(df, overshoot)
        df = self._smooth_boxes(df, n)
        
        video.reset()
        df = self._landmarks(df, video, batch_size, dt)
        
        return df

    def _landmarks(
        self, df: pd.DataFrame, video: Video, batch_size: int, dt: float
    ) -> pd.DataFrame:
        """Landmark detection @TODO: Batch system should depend on sequence"""
        landmarks = []
        for sequence in tqdm(df.sequence.unique(), desc="Sequence"):
            df_sequence = df[df.sequence == sequence]

            # Compute number of batches for the sequence
            supplement = 0 if len(df_sequence) % batch_size == 0 else 1
            n_batch = len(df_sequence) // batch_size + supplement

            for b in tqdm(range(n_batch), "Landmarking Faces"):
                start = b * batch_size
                end = min(start + batch_size, len(df_sequence))

                # Collect faces and offsets
                face_list, offsets = [], []
                for _, row in df[start:end].iterrows():
                    x0 = int(np.floor(row.x - row.box_size * 0.5))
                    y0 = int(np.floor(row.y - row.box_size * 0.5))
                    x1 = int(np.floor(row.x + row.box_size * 0.5))
                    y1 = int(np.floor(row.y + row.box_size * 0.5))
                    frame = video[int(np.floor(row.time / dt))]
                    face_list.append(frame[y0:y1, x0:x1])
                    offsets.append([x0, y0])
                
                # Make sure faces are the same size to process in batch
                n_faces = len(face_list)
                size = np.max([max(face.shape) for face in face_list])
                faces = np.zeros((n_faces, size, size, 3), dtype=np.uint8)
                offsets = np.array(offsets)
                for i, face in enumerate(face_list):
                    h, w, c = face.shape 
                    faces[i, :h, :w, :c] = face[:, :, :]
                
                # Compute face landmarks
                preds = self.landmarker(faces)
                for n in range(preds.size(1)):
                    preds[:, n, 0] += offsets[:, 0]
                    preds[:, n, 1] += offsets[:, 1]
                landmarks.append(preds)
        landmarks = np.concatenate(landmarks)

        # Set landmark infos
        for i in range(68):
            df[f"mark_{i}_x"] = landmarks[:, i, 0]
            df[f"mark_{i}_y"] = landmarks[:, i, 1]

        return df

    def _smooth_boxes(self, df: pd.DataFrame, n: int) -> pd.DataFrame:
        """Smooth box positions by applying gaussian weighted convolution"""
        c = 1.0 / np.sqrt(2.0 * np.pi)
        gaussian = lambda n: c * np.exp(-0.5 * np.linspace(-4, 4, n) ** 2)

        def smooth(array: np.ndarray) -> np.ndarray:
            return np.convolve(array, gaussian(n), 'same') / n
            
        sequences = df.sequence.unique()
        for sequence in tqdm(sequences, desc="Smoothing Boxes"):
            df_sequence = df[df.sequence == sequence]
            df_sequence.x = smooth(df_sequence.x)
            df_sequence.y = smooth(df_sequence.y)
        return df

    def _unify_boxes(self, df: pd.DataFrame, overshoot: float) -> pd.DataFrame:
        """Unify box sizes"""
        sequences = df.sequence.unique()
        sizes = []
        for sequence in tqdm(sequences, desc="Unifying Boxes"):
            df_sequence = df[df.sequence == sequence]
            size = max(df_sequence.w.max(), df_sequence.h.max()) * overshoot
            sizes += [size] * len(df_sequence)
        df["box_size"] = sizes
        df.drop(labels=["w", "h"], axis=1, inplace=True)
        return df

    def _infos(self, video: Video, batch_size: int) -> Tuple[float, int]:
        """Get video infos"""
        dt = 1.0 / video.fps
        supplement = 0 if len(video) % batch_size == 0 else 1
        n_batch = len(video) // batch_size + supplement
        return dt, n_batch

    def _process_boxes(
        self, video: Iterator[np.ndarray], n_batch: int, batch_size: int
    ) -> pd.DataFrame:
        """Process frames for boxing"""
        pbar = tqdm(range(n_batch), desc="Boxing Faces")
        df = pd.DataFrame({name: [] for name in self.names})
        dfs = [self._process_batch_boxes(video, batch_size) for _ in pbar]
        df = pd.concat(dfs, ignore_index=True)
        return df

    def _process_batch_boxes(
        self, video: Iterator[np.ndarray], batch_size: int
    ) -> pd.DataFrame:
        """Process one batch for boxing"""
        frames = list(islice(video, batch_size))
        df = pd.DataFrame(data=self.boxer(frames), columns=self.names)
        return df

    def _clean_boxes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean dataframe"""
        df.drop(labels=self.names, axis=1, inplace=True)
        df.dropna(inplace=True)
        return df

    def _classify_boxes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Assign sequences"""
        seq = 0
        last = None
        sequences = []
        for i in tqdm(range(len(df)), "Assigning Sequence"):
            current = df.x0[i]
            if pd.isna(current):
                seq = seq if last is pd.isna(last) else seq + 1
            last = current
            sequences.append(seq)
        df["sequence"] = sequences
        return df

    def _transform_boxes(self, df: pd.DataFrame, dt: float) -> pd.DataFrame:
        """Compute position and size"""
        df["w"] = np.abs(df.x0 - df.x1)
        df["h"] = np.abs(df.y0 - df.y1)
        df["x"] = df.x0 + df.w * 0.5
        df["y"] = df.y0 + df.h * 0.5
        df["time"] = np.arange(0, len(df)) * dt
        df["frame_w"] = [self.size[0]] * len(df)
        df["frame_h"] = [self.size[1]] * len(df)
        return df