# -*- coding: utf-8 -*-
"""facecrop/video.py

The file provides a simple wrapper for accessing video frames.
"""
from imutils.video import FileVideoStream
from typing import Iterator
from typing import Tuple

import numpy as np
import cv2


class Video:
    """Video

    Video File Stream Wrapper to provide Pythonic API to access video frames.

    Attributes:
        path {str} -- path to video file
        size {Tuple[int, int]} -- resize video (default: {None})
        video {FileVideoStream} -- video file stream

    Examples:
        >>> from facecrop import Video
        >>> from itertools import islice
        >>> 
        >>> 
        >>> video = Video(path, (1280, 720))
        >>> with video as v:
        >>>     frames = list(islice(v, 10))
    """

    def __init__(self, path: str, size: Tuple[int, int] = None) -> None:
        self.path = path
        self.size = size
        self.reset()

    def reset(self) -> None:
        """Reste video stream"""
        self.video = FileVideoStream(self.path)

    @property
    def fps(self) -> int:
        """Video Frame Per Second

        Returns:
            int -- video fps
        """
        return int(self.video.stream.get(cv2.CAP_PROP_FPS))

    def __getitem__(self, idx: int) -> np.ndarray:
        """Get Frame by id"""
        self.video.stream.set(1, idx)
        _, frame = self.video.stream.retrieve()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if self.size is not None:
            frame = cv2.resize(frame, self.size)
        return frame

    def __len__(self) -> int:
        """Video Frame Number

        Returns:
            int -- number of frames
        """
        return int(self.video.stream.get(cv2.CAP_PROP_FRAME_COUNT))

    def __enter__(self) -> Iterator[np.ndarray]:
        """Enter Video

        Starts video stream and returns an iterator when entering with block.

        Yields:
            Iterator[np.ndarray] -- generator to iterate over video frames
        """
        def frames() -> Iterator[np.ndarray]:
            for _ in range(len(self)):
                frame = self.video.read()
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if self.size is not None:
                    frame = cv2.resize(frame, self.size)
                yield frame

        self.video.start()
        return frames()

    def __exit__(self, *args, **kwargs) -> None:
        """Exit Video

        Exit video when out of with block.
        """
        self.video.stop()