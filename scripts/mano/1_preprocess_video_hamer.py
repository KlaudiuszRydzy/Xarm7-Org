import hamer
import imageio
import numpy as np
import local
import os
import os.path as osp
import argparse
from pprint import pprint

import cv2
import imageio
import jax
import jax.numpy as jnp
import numpy as np
import torch
import shutil
from pathlib import Path
import sys
xgym_path = Path(__file__).resolve().parents[2]  # Adjust path to ~/repos/xgym
sys.path.append(str(xgym_path))
from PIL import Image
from tqdm import tqdm
from smplx.utils import (Array, MANOOutput, SMPLHOutput, SMPLOutput,
                         SMPLXOutput, Struct, Tensor, to_np, to_tensor)
from tqdm import tqdm

from hamer.configs import CACHE_DIR_HAMER
from hamer.datasets.vitdet_dataset import (DEFAULT_MEAN, DEFAULT_STD,
                                           ViTDetDataset)
from hamer.models import (DEFAULT_CHECKPOINT, HAMER, MANO, download_models,
                          load_hamer)
from hamer.utils import SkeletonRenderer, recursive_to
from hamer.utils.geometry import perspective_projection
from hamer.utils.renderer import Renderer, cam_crop_to_full
from hamer.utils.render_openpose import render_openpose
from my_oakink_test_new import get_config, init_detector, infer, flatten
from vitpose_model import ViTPoseModel
import tensorflow as tf
from xgym.__init__ import MANO_1, MANO_1DONE, MANO_2

def extract_npz_files(i, img, detector, vitpose, device, model, model_cfg, renderer):

    out = infer(i, img, detector, vitpose, device, model, model_cfg, renderer)

    out = jax.tree.map(lambda x: x[0], out.data, is_leaf=lambda x: isinstance(x, list))

    out = flatten(out)
    clean = lambda x: x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else x
    out = jax.tree.map(clean, out)

    return out

def stack_sequence(sequence):
    """
    Stacks all frames in the sequence into arrays for saving.
    """
    stacked = {}
    for key in sequence[0].keys():
        stacked[key] = np.stack([step[key] for step in sequence], axis=0)
    return stacked

def process_sequence(sequence):
    """
    Filters frames to ensure only right-hand detections are included.
    """
    is_right = int(np.array([step["right"].mean() for step in sequence]).mean() > 0.5)

    if not is_right:
        return None  # Skip sequences with no right-hand detections

    def select_hand(x):
        if x.shape and x.shape[0] == 2:
            return x[is_right]
        if x.shape and x.shape[0] == 1:
            return x[0]
        return x

    processed_sequence = []
    for step in sequence:
        step = jax.tree_map(select_hand, step)
        processed_sequence.append(step)

    return processed_sequence

def pipeline(frame, out):

    # center crop to 224
    pass

def get_total_frames(video_path):
    """
    Get the total number of frames in the video using OpenCV.
    """
    cap = cv2.VideoCapture(str(video_path))  # Ensure video path is converted to string
    if not cap.isOpened():
        print(f"Error: Cannot open the video file: {video_path}")
        return -1
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return total_frames

def main():
    # Download and load models
    download_models(CACHE_DIR_HAMER)
    model, model_cfg = load_hamer()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    model.eval()

    # Initialize supporting models
    detector = init_detector()
    vitpose = ViTPoseModel(device)
    renderer = Renderer(model_cfg, faces=model.mano.faces)

    # Create output directories if not exist
    MANO_2.mkdir(exist_ok=True)
    MANO_1DONE.mkdir(exist_ok=True)

    # Process all videos in MANO_1
    video_files = list(MANO_1.glob("*.mp4"))
    for video_path in video_files:
        print(f"Processing video: {video_path.name}")
        sequence = []
        reader = imageio.get_reader(video_path)
        total_frames = get_total_frames(video_path)
        print(f"The video contains {total_frames} frames.")
        for i, frame in tqdm(enumerate(reader), desc=f"Processing {video_path.name}"):
            try:
                out = extract_npz_files(i, frame, detector, vitpose, device, model, model_cfg, renderer)
                sequence.append(out)
            except Exception as e:
                print(f"Error processing frame {i} in {video_path.name}: {e}")

        # Filter sequence for right-hand detections only
        filtered_sequence = process_sequence(sequence)
        if filtered_sequence is not None:
            stacked_sequence = stack_sequence(filtered_sequence)

            output_npz_path = MANO_2 / f"{video_path.stem}_filtered.npz"
            np.savez(output_npz_path, **stacked_sequence)

        # Move processed video to MANO_1DONE
        shutil.move(str(video_path), MANO_1DONE / video_path.name)
        print(f"Moved {video_path.name} to {MANO_1DONE}")


if __name__ == "__main__":
    main()
