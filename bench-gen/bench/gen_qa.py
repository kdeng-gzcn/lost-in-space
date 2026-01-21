import logging
from typing import List, Any
from pathlib import Path
import json

import numpy as np
from datasets import Dataset
from sympy import arg

from src.dataset.utils import dataset_map, prompt_template
from src.dataset.base_frame import Frame, FramePair
from src.logging.logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

### Set Seed
import random
import argparse
random.seed(42)
np.random.seed(42)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate QA dataset from benchmark")
    parser.add_argument(
        "--hf_id",
        type=str,
        required=False,
        default=None,
        help="Hugging Face dataset name"
    )
    parser.add_argument(
        "--bench_dir",
        type=str,
        required=False,
        default="benchmark/bench",
        help="Path to benchmark directory"
    )
    return parser.parse_args()


def read_ss_framepair(pari_dir: Path) -> FramePair:
    src_color_path = list((pari_dir / "src").glob("*.color.png"))[0]
    src_depth_path = list((pari_dir / "src").glob("*.depth.png"))[0]
    src_pose_path = list((pari_dir / "src").glob("*.pose.txt"))[0]

    tgt_color_path = list((pari_dir / "tgt").glob("*.color.png"))[0]
    tgt_depth_path = list((pari_dir / "tgt").glob("*.depth.png"))[0]
    tgt_pose_path = list((pari_dir / "tgt").glob("*.pose.txt"))[0]

    src_frame = Frame(
        color_path=src_color_path,
        depth_path=src_depth_path,
        pose_path=src_pose_path,
    )

    tgt_frame = Frame(
        color_path=tgt_color_path,
        depth_path=tgt_depth_path,
        pose_path=tgt_pose_path,
    )

    return FramePair(src_frame=src_frame, tgt_frame=tgt_frame)


def read_sn_framepair(pari_dir: Path) -> FramePair:
    src_color_path = list((pari_dir / "src").glob("*.jpg"))[0]
    src_depth_path = list((pari_dir / "src").glob("*.png"))[0]
    src_pose_path = list((pari_dir / "src").glob("*.txt"))[0]

    tgt_color_path = list((pari_dir / "tgt").glob("*.jpg"))[0]
    tgt_depth_path = list((pari_dir / "tgt").glob("*.png"))[0]
    tgt_pose_path = list((pari_dir / "tgt").glob("*.txt"))[0]

    src_frame = Frame(
        color_path=src_color_path,
        depth_path=src_depth_path,
        pose_path=src_pose_path,
    )

    tgt_frame = Frame(
        color_path=tgt_color_path,
        depth_path=tgt_depth_path,
        pose_path=tgt_pose_path,
    )

    return FramePair(src_frame=src_frame, tgt_frame=tgt_frame)


def read_frame_pair(pari_dir: Path, dataset: str) -> FramePair:
    if dataset == "ss":
        return read_ss_framepair(pari_dir)
    elif dataset == "sn":
        return read_sn_framepair(pari_dir)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def iter_sample(bench_dir: Path):
    """
    When iterting, limit 100 samples per level, and shuffle the samples.
    """
    for level_dir in bench_dir.iterdir():
        if level_dir.is_dir():
            dataset, level = level_dir.name.split("-")
            pair_dirs = [p for p in level_dir.iterdir() if p.is_dir()]
            random.shuffle(pair_dirs)
            pair_dirs = pair_dirs[:100]  # Limit to 100 samples
            for pair_dir in pair_dirs:
                if pair_dir.is_dir():
                    yield dataset, level, pair_dir


def generate_qa(rpv: dict):
    MOVE_LEFT = "Move left while yawing right"
    MOVE_RIGHT = "Move right while yawing left"
    TRAP = "The camera is stationary"

    if rpv["text"][-3] == "translate left":
        gt_text = MOVE_LEFT
    elif rpv["text"][-3] == "translate right":
        gt_text = MOVE_RIGHT
    else:
        raise ValueError(f"Unknown rpv text: {rpv['text']}")

    
    ### zero-shot prompt first
    shuf_opt_cand = random.sample([MOVE_LEFT, MOVE_RIGHT], 2)
    options = "\n".join(
        [f"{idx}. {opt}" for idx, opt in enumerate(shuf_opt_cand)]
    )
    prompt = prompt_template.format(options=options)

    zs = {
        "prompt": prompt,
        "gt_idx": shuf_opt_cand.index(gt_text),
        "gt_text": gt_text,
    }

    ### with trap prompt second
    shuf_opt_cand = random.sample([MOVE_LEFT, MOVE_RIGHT, TRAP], 3)
    options = "\n".join(
        [f"{idx}. {opt}" for idx, opt in enumerate(shuf_opt_cand)]
    )
    prompt = prompt_template.format(options=options)

    wt = {
        "prompt": prompt,
        "gt_idx": shuf_opt_cand.index(gt_text),
        "gt_text": gt_text,
        "trap_idx": shuf_opt_cand.index(TRAP),
    }

    return {
        "zero-shot": zs,
        "w/ trap": wt,
    }
    

def main(args) -> Dataset:
    bench_dir = Path(args.bench_dir)
    sample_id = 0
    data = []
    for dataset, level, pair_dir in iter_sample(bench_dir):
        full_dataset_name = dataset_map[dataset]
        frame_pair = read_frame_pair(pair_dir, dataset)
        rpv = json.load(open(pair_dir / "rpv.json"))
        tau_and_cpd = json.load(open(pair_dir / "tau_cpd.json"))
        tau = tau_and_cpd["tau"]
        cpd = tau_and_cpd["cpd"]

        qa = generate_qa(rpv)
        
        data.append({
            "id": sample_id,
            "level": f"{level}-{int(level)+5}",
            "dataset": full_dataset_name,
            "metadata": json.load(open(pair_dir / "metadata.json")),
            "images": [frame_pair.src_frame.color, frame_pair.tgt_frame.color],
            "qa": qa,
            "depth_images": [frame_pair.src_frame.depth, frame_pair.tgt_frame.depth],
            "poses": [frame_pair.src_frame.pose, frame_pair.tgt_frame.pose],
            "intrinsic": np.loadtxt(str(pair_dir / "K.txt")),
            "relative_pose_vector": json.load(open(pair_dir / "rpv.json")),
            "tau": tau,
            "central_point_deviation": cpd,
        })
        sample_id += 1
                    
    return Dataset.from_list(data)


if __name__ == "__main__":
    args = parse_args()
    hf_data = main(args)
    print(f"Finished with QA generation, dataset size: {len(hf_data)}")
    if args.hf_id is not None:
        hf_data.push_to_hub(args.hf_id)
    