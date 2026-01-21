import argparse
import logging
from typing import List

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
        description="Filter 7-scene dataset according to different criteria."
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
            yield scene_dir.name, scene_dir


def glob_color_img(seq_dir: Path):
    return sorted((seq_dir / "color").glob("*.jpg"), key=lambda x: int(x.stem))


def glob_depth_img(seq_dir: Path):
    return sorted((seq_dir / "depth").glob("*.png"), key=lambda x: int(x.stem))


def glob_pose(seq_dir: Path):
    return sorted((seq_dir / "pose").glob("*.txt"), key=lambda x: int(x.stem))


def glob_intrinsic(seq_dir: Path) -> np.ndarray:
    return np.loadtxt(seq_dir / "intrinsic" / "intrinsic_color.txt")[:3, :3]


def find_frame_pairs(cfg, frames: List[Frame]):
    stride_i = cfg.filter.stride_i
    stride_j = cfg.filter.stride_j
    min_gap = cfg.filter.min_gap
    max_gap = cfg.filter.max_gap
    tau_limit = cfg.filter.tau_limit
    cpd_limit = cfg.filter.cpd_limit
    theta_limit = cfg.filter.theta_limit

    frame_pairs = []
    i = 0
    while i < len(frames):
        found = False
        j = i + min_gap
        while j < min(len(frames), i + max_gap):
            frame_pair = FramePair(frames[i], frames[j])
            rpv_dict = frame_pair.cal_rpv()
            tau_cpd_dict = frame_pair.cal_tau_and_cpd()

            # Apply filtering criteria here based on cfg settings.
            accept = False
            if tau_limit <= tau_cpd_dict["tau"] <= tau_limit + 5 and tau_cpd_dict["cpd"] <= cpd_limit:
                theta, phi, psi, tx, ty, tz = rpv_dict["value"]
                if all([
                    np.sign(phi) != np.sign(tx),
                    abs(theta) <= theta_limit,
                    abs(phi) >= 2*abs(theta),
                    abs(tx) >= 2*abs(ty)
                ]):
                    accept = True
            
            if accept:
                frame_pairs.append(frame_pair)
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

    for scene_name, scene_dir in iterate_seq(data_dir):
        logger.info(f"Processing scene: {scene_name}")
        color_imgs = glob_color_img(scene_dir)
        depth_imgs = glob_depth_img(scene_dir)
        poses = glob_pose(scene_dir)
        K_sn = glob_intrinsic(scene_dir)

        frames = [
            Frame(
                color_path=color_img, 
                depth_path=depth_img, 
                pose_path=pose,
                K=K_sn,
                scene_name=scene_name,
            ) for color_img, depth_img, pose in zip(color_imgs, depth_imgs, poses)
        ]

        frame_pairs = find_frame_pairs(cfg, frames)
        for frame_pair in frame_pairs:
            # Save or log the selected frame pairs as needed.
            logger.info(f"Selected {sample_id+1}-th frame pair: {frame_pair.src_frame.color_path.stem} <-> {frame_pair.tgt_frame.color_path.stem}")
            frame_pair.save(output_dir, f"sn-{int(cfg.filter.tau_limit)}", sample_id) # TODO: can be combined with tau value
            sample_id += 1

    # Save cfg to actual output dir
    cfg_save_path = output_dir / f"sn-{int(cfg.filter.tau_limit)}" / "config.yaml"
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
    