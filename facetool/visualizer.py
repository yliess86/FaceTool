# -*- coding: utf-8 -*-
"""facetool.visualizer

The file provides a visualizarion tool to produce detection and landmark 
visualization clip from a given video and annotation file.
"""
from tqdm import tqdm
from typing import Tuple

import cv2
import moviepy.editor as mpe
import numpy as np
import pandas as pd


BLUE = 21, 11, 248                # Blue of Logo
POINTS = [                        # Points Used in LINES
    1, 5, 9, 13, 17, 49, 55, 32, 34, 36, 31, 37, 38, 40, 46, 45, 43, 20, 25
]
LINES = [
    # Right     Left
    ( 9, 49), ( 9, 55),           # Chin to Mouth Corner
    ( 5, 49), (13, 55),           # Jaw to Mouth Corner
    ( 5,  9), (13,  9),           # Chin to Jaw
    (49, 55),                     # Mouth Corner
    (32, 34), (36, 34),           # Bottom Nose
    (32, 31), (36, 31),           # Top Nose
    (49, 32), (55, 36),           # Mouth Corner to Nose
    ( 5,  1), (13, 17),           # Jaw to Ear
    (32,  1), (36, 17),           # Nose to Ear
    (32, 40), (36, 43),           # Nose to Eye Inner Corner
    ( 1, 37), (17, 46),           # Ear to Eye Outer Corner
    ( 1, 20), (17, 25),           # Ear to Eye Brow
    (38, 20), (45, 25),           # Ear Top to Eye Brow
    (20, 25),                     # Eye Brow
    (37, 38), (38, 40), (40, 37), # Eye Right
    (46, 45), (45, 43), (43, 46), # Eye Left
]


class FaceVisualizer:
    @staticmethod
    def visualize(
        video: str, annotations: str, 
        save: str = None, size: Tuple[int, int] = None
    ) -> None:
        """Visualize

        Arguments:
            video {str} -- path to original video
            annotations {str} -- path to corresponding annotation csv file
            save {str} -- save path to save gif. If None, will not be saved
            size {Tuple[int, int]} -- resize video before saving to gif
        """
        video = mpe.VideoFileClip(video)
        annotations = pd.read_csv(annotations)

        # Gather Frames
        n_frames = int(np.round(video.fps * video.duration))
        frames = video.iter_frames()
        frames = list(tqdm(frames, desc="Loading Frames", total=n_frames))
        
        # Draw on Frames
        images = []
        rows = annotations.iterrows()
        pbar = tqdm(rows, desc="Processing Frames", total=len(annotations))
        for _, row in pbar:
            frame = frames[row.frame_idx]
            
            # Box Data with size expanded
            x, y = row.box_x, row.box_y       # Coords
            w, h = row.box_w, row.box_h       # Size
            w, h = int(w * 1.2), int(h * 1.2)

            # Box Data in Rectangle Format
            x0, y0 = x - w // 2, y - h // 2   # pt1
            x1, y1 = x + w // 2, y + h // 2   # pt2
            
            # Draw Box around Face
            frame = cv2.rectangle(
                frame, (x0, y0), (x1, y1),
                color=BLUE, thickness=12
            )

            # Gather X and Y Landmark Coords
            X = [row[f"landmark_{i + 1}_x"] for i in range(68)]
            Y = [row[f"landmark_{i + 1}_y"] for i in range(68)]
            
            # Draw Landmarks as Points (bigger if part of line)
            for i, (x, y) in enumerate(zip(X, Y)):
                s = 6 if i + 1 in POINTS else 2
                frame = cv2.circle(frame, (x, y), s, color=BLUE, thickness=-1)

            # Draw Lines to Join Key Landmark Points
            for p1, p2 in LINES:
                p1, p2 = p1 - 1, p2 - 1
                frame = cv2.line(
                    frame, (X[p1], Y[p1]), (X[p2], Y[p2]),
                    color=BLUE, thickness=2
                )

            images.append(frame)

        # Create and Preview Sequence
        clip = mpe.ImageSequenceClip(images, fps=video.fps)
        clip.preview()

        # Save to GIF if Asked
        if save is not None and size is not None:
            clip = clip.resize(width=size[0], height=size[1])
            clip.write_gif(save, fps=15)