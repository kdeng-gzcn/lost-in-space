import argparse
import logging
from typing import List, Tuple

import numpy as np
from pathlib import Path

from yacs.config import CfgNode as CN
from config.default import cfg

from src.logging.logging_config import setup_logging
from src.dataset.base_frame import Frame, FramePair
from src.dataset.utils import K_ss

setup_logging()
logger = logging.getLogger(__name__)


def arg_parser():
    parser = argparse.ArgumentParser(
        description="Filter 7 Scenes dataset according to different criteria."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to the output directory for the filtered 7-scene dataset.",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to the data directory for the filtered 7-scene dataset.",
    )
    parser.add_argument(
        "--yaml_path",
        type=str,
        required=True,
        help="Path to the YAML configuration file.",
    )
    return parser.parse_args()


def load_cfg(args):
    arg_dict = vars(args)
    arg_cfg = CN(arg_dict)
    cfg.merge_from_other_cfg(arg_cfg)
    cfg.merge_from_file(args.yaml_path)
    return cfg


def iterate_seq(data_dir: Path):
    for scene_dir in data_dir.iterdir():
        if scene_dir.is_dir():
            for seq_dir in scene_dir.iterdir():
                if seq_dir.is_dir():
                    yield scene_dir.name, seq_dir.name, seq_dir


def glob_color_img(seq_dir: Path):
    return sorted(seq_dir.glob("*.color.png"))


def glob_depth_img(seq_dir: Path):
    return sorted(seq_dir.glob("*.depth.png"))


def glob_pose(seq_dir: Path):
    return sorted(seq_dir.glob("*.pose.txt"))


def find_frame_pairs(cfg, frames: List[Frame]) -> List[Tuple[FramePair, str]]:
    stride_i = cfg.filter.stride_i
    stride_j = cfg.filter.stride_j
    min_gap = cfg.filter.min_gap
    max_gap = cfg.filter.max_gap
    theta_min, theta_max = cfg.filter.theta
    phi_min, phi_max = cfg.filter.phi
    psi_min, psi_max = cfg.filter.psi
    tx_min, tx_max = cfg.filter.tx
    ty_min, ty_max = cfg.filter.ty
    tz_min, tz_max = cfg.filter.tz

    label = ["theta", "phi", "psi", "tx", "ty", "tz"]
    threshold_min = [theta_min, phi_min, psi_min, tx_min, ty_min, tz_min]
    threshold_max = [theta_max, phi_max, psi_max, tx_max, ty_max, tz_max]

    frame_pairs = []
    i = 0
    while i < len(frames):
        found = False
        j = i + min_gap
        while j < min(len(frames), i + max_gap):
            frame_pair = FramePair(frames[i], frames[j])
            rpv_dict = frame_pair.cal_rpv()

            # Apply filtering criteria here based on cfg settings.
            accept = False

            # Iterate through [theta, phi, psi, tx, ty, tz] to find significant DOF
            for sig_dof, sig_value, th_max in zip(label, rpv_dict["value"], threshold_max):
                # Check if the current relative_pose is greater than the upper threshold
                if np.abs(sig_value) > th_max:
                    # Test other values to ensure they are below their lower thresholds
                    if all(
                        np.abs(value) < th_min
                        for dof, value, th_min in zip(label, rpv_dict["value"], threshold_min) if dof != sig_dof
                    ):
                        accept = True
                        break # break the for loop if one significant DOF is found
            
            if accept:
                frame_pairs.append((frame_pair, sig_dof))
                # greedy step: jump i to j and continue from there
                i += (j-i)//2
                found = True
                break

            j += stride_j

        if not found:
            # no acceptable j found for this i: advance i by stride_i
            i += stride_i

    return frame_pairs


def main(cfg):
    logger.info(f"Print configuration:\n{cfg}")

    output_dir = Path(cfg.output_dir)
    data_dir = Path(cfg.data_dir)

    sample_id = 0

    for scene_name, seq_name, seq_dir in iterate_seq(data_dir):
        logger.info(f"Processing scene: {scene_name}, sequence: {seq_name}")
        color_imgs = glob_color_img(seq_dir)
        depth_imgs = glob_depth_img(seq_dir)
        poses = glob_pose(seq_dir)
        
        frames = [
            Frame(
                color_path=color_img, 
                depth_path=depth_img, 
                pose_path=pose,
                K=K_ss,
                scene_name=scene_name,
                seq_name=seq_name,
            ) for color_img, depth_img, pose in zip(color_imgs, depth_imgs, poses)
        ]

        frame_pairs = find_frame_pairs(cfg, frames)
        for frame_pair, sig_dof in frame_pairs:
            # Save or log the selected frame pairs as needed.
            logger.info(f"Selected {sample_id+1}-th frame pair ({sig_dof}): {frame_pair.src_frame.color_path.stem} <-> {frame_pair.tgt_frame.color_path.stem}")
            frame_pair.save(output_dir, f"ss/{sig_dof}", sample_id)
            sample_id += 1

    # Save cfg to actual output dir
    cfg_save_path = output_dir / f"ss" / "config.yaml"
    try:
        with open(cfg_save_path, "w") as f:
            f.write(cfg.dump())
    except Exception as e:
        logger.error(f"Failed to save config to {cfg_save_path}: {e}")
        logger.error("Probably due to no valid pair, hence the folder not created.")


if __name__ == "__main__":
    args = arg_parser()
    cfg = load_cfg(args)
    main(cfg)
    