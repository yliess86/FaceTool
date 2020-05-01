# -*- coding: utf-8 -*-
"""facecrop.__main__"""
from facecrop.annotator import FaceAnnotator
from typing import Tuple

import argparse


def size(s: str) -> Tuple[int, int]:
    """Convert string argument to size Tuple"""
    elements = s.split(",")
    if len(elements) != 2:
        raise argparse.ArgumentTypeError("Should contain 2 ints `320, 155`")
    return tuple(map(int, elements[:2]))


# Define Command Parser
parser = argparse.ArgumentParser()
sub_parsers = parser.add_subparsers(dest="action")

# Define Annotate Command Arguments
annotate = sub_parsers.add_parser("annotate")
annotate.add_argument(
    "--video", type=str, required=True,
    help="video path to be annotated (mp4 by preference `video.mp4`)"
)
annotate.add_argument(
    "--annotations", type=str, required=True,
    help="annotations saving path (save to csv `annotations.csv`)"
)
annotate.add_argument(
    "--dbatch_size", type=int, required=True,
    help="batch_size for the detector inference"
)
annotate.add_argument(
    "--lbatch_size", type=int, required=True,
    help="batch_size for the landmark video loader"
)
annotate.add_argument(
    "--n_process", type=int, required=True,
    help="number of threads used for lanrmarking"
)
annotate.add_argument(
    "--size", type=size, default="320, 155", nargs=2,
    help="resize the video for the detector"
)
annotate.add_argument(
    "-d", "--device", type=str, default="cpu",
    help="device to run detector on, `cpu` or `cuda`"
)

# Parse and Process Commands
args = parser.parse_args()
if args.action == "annotate":
    face_annotator = FaceAnnotator(
        dbatch_size=args.dbatch_size, lbatch_size=args.lbatch_size,
        size=args.size, n_process=args.n_process, device=args.device
    )

    # Annotate Video and Save DataFrame
    annotations = face_annotator(args.video)
    annotations.to_csv(args.annotations)