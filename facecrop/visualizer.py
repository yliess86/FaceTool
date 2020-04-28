# -*- coding: utf-8 -*-
"""facecrop/visualizer.py

The file provides a visualizer to help visualize the annotation results from
the facecrop annotator given a video and its corresponding annotation
dataframe. The result is directly displayed on the screen using cv2 and can
be saved as a gif. 
!Becareful the longer the video is, the heavier the gif will be.
"""
from facecrop.annotator import FaceAnnotator
from facecrop.landmark import FaceLandmarker
from facecrop.video import Video
from PIL import Image
from typing import Tuple

import cv2
import numpy as np
import pandas as pd


Box = Tuple[int, int, int, int]


class FaceVisualizer:
    """FaceVisualizer

    The visualizer allow visualization of face cropping and landmarks given a
    video and the resulting facecrop annotation dataframe from file.
    """

    ORANGE = (240, 151,  71)
    BLUE   = ( 21, 111, 248)

    @classmethod
    def get_box(cls, df: pd.Series, frame: np.ndarray) -> Box:
        """Get Box coordinates based on position dans box size"""
        x0 = int(df.x - df.box_size * 0.5)
        y0 = int(df.y - df.box_size * 0.5)
        x1 = int(df.x + df.box_size * 0.5)
        y1 = int(df.y + df.box_size * 0.5)
        return x0, y0, x1, y1

    @classmethod
    def draw_box(
        cls, box: Box, frame: np.ndarray, cropped: bool
    ) -> np.ndarray:
        """Draw box or crop face on frame"""
        x0, y0, x1, y1 = box
        if cropped:
            return frame[y0:y1, x0:x1]
        return cv2.rectangle(frame, (x0, y0), (x1, y1), cls.ORANGE, 6)

    @classmethod
    def draw_marks(
        cls, df: pd.Series, box: Box, frame: np.ndarray, cropped: bool
    ) -> np.ndarray:
        """Draw lanfmarks on frame"""
        x0, y0, _, _ = box
        for i in range(68):
            x = int(df[f"mark_{i}_x"] - (x0 if cropped else 0))
            y = int(df[f"mark_{i}_y"] - (y0 if cropped else 0))
            frame = cv2.circle(frame, (x, y), 3, cls.BLUE, -1)
        return frame

    @classmethod
    def visualize(
        cls, video: str, annotation: str,
        cropped: bool = False, save: str = None
    ) -> None:
        """Visulalize

        Arguments:
            video {str} -- path to the video
            annotation {str} -- path to the annotations

        Keyword Arguments:
            cropped {bool} -- croppe frame to the face if `True` else draw
                rectangle box (default: {False})
            save {str} -- path to save the gif. If `None` a gif file will not
                be created. (default: {None})
        """
        df = pd.read_csv(annotation)
        video = Video(video, (df.frame_w[0], df.frame_h[0]))
        
        if save is not None:
            frames = []

        dt = 1.0 / video.fps
        df.time = np.floor(df.time / dt).astype(int)
        for _, row in df.iterrows():
            frame = video[row.time]
            box = cls.get_box(row, frame)
            frame = cls.draw_box(box, frame, cropped)
            frame = cls.draw_marks(row, box, frame, cropped)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = frame.astype(np.uint8)

            cv2.imshow(f"FaceCrop: {video}", frame)
            if (cv2.waitKey(30) & 0xFF) == 27:
                break

            if save is not None:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)

        if save is not None:
            images = [Image.fromarray(frame) for frame in frames]
            images[0].save(
                save, append_images=images[1:], optimize=False, 
                save_all=True, duration=40, loop=0
            )