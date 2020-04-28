# -*- coding: utf-8 -*-
"""facetorch/__main__.py"""
from argparse import ArgumentParser
from argparse import ArgumentTypeError
from facecrop.annotator import FaceAnnotator
from facecrop.visualizer import FaceVisualizer
from typing import Tuple

import os


def size(s: str) -> Tuple[int, int]:
    try:
        w, h = map(int, s.split(","))
        return w, h
    except:
        raise ArgumentTypeError("Size must be `w, h`")


parser = ArgumentParser()

parser.add_argument("--video_path", type=str, required=True)
parser.add_argument("--annot_path", type=str, required=True)

parser.add_argument("--video_size", type=size, default="1280, 720", nargs=2)
parser.add_argument("--batch_size", type=int,  default=16)
parser.add_argument("--max_batch",  type=int,  default=None)
parser.add_argument("--device",     type=str,  default="cpu")

parser.add_argument("-v", "--visualize", action="store_true")
parser.add_argument("-c", "--crop",      action="store_true")

args = parser.parse_args()

print(f"==== {args.video_path}")
if not os.path.isfile(args.annot_path):
    annotator = FaceAnnotator(args.video_size, args.device)
    annot = annotator(
        args.video_path, args.batch_size, max_batch=args.max_batch
    )
    annot.to_csv(args.annot_path)

if args.visualize:
    extension = args.video_path.split(".")[-1]
    path = args.video_path.replace(f".{extension}", ".gif")
    FaceVisualizer.visualize(
        args.video_path, args.annot_path, args.crop, save=path
    )