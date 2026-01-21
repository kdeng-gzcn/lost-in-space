import logging
import argparse
from typing import List, Any, Union
from pathlib import Path

import numpy as np
from datasets import Dataset

from src.dataset.utils import prompt_template_diag, prompt_template_diag_cot
from src.dataset.base_frame import Frame, FramePair
from src.logging.logging_config import setup_logging

### Set Seed
import random
random.seed(42)
np.random.seed(42)

setup_logging()
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate QA dataset from benchmark")
    parser.add_argument(
        "--hf_id",
        type=str,
        default=None,
        required=False,
        help="Hugging Face dataset name"
    )
    parser.add_argument(
        "--bench_dir",
        type=str,
        default="benchmark/diag",
        help="Path to benchmark directory"
    )
    return parser.parse_args()


def read_framepair(scp: Path, sdp: Union[Path, None], spp: Path, 
                   tcp: Path, tdp: Union[Path, None], tpp: Path) -> FramePair:
    src_frame = Frame(
        color_path=scp,
        depth_path=sdp,
        pose_path=spp,
    )

    tgt_frame = Frame(
        color_path=tcp,
        depth_path=tdp,
        pose_path=tpp,
    )
    return FramePair(src_frame=src_frame, tgt_frame=tgt_frame)


def iter_ss_sample(bench_dir: Path):
    """
    When iterting, limit 100 samples per level, and shuffle the samples.
    """
    bench_dir = Path(f"{bench_dir}/ss")
    for dof_dir in bench_dir.iterdir():
        if dof_dir.is_dir():
            dof = dof_dir.name.split("_")[0]
            scp_l = sorted(list(dof_dir.glob("**/src/*.color.png")))
            pair_list = [scp.parent.parent for scp in scp_l]
            random.shuffle(pair_list)
            for pair_dir in pair_list[:100]:
                scp = list((pair_dir / "src").glob("*.color.png"))[0]
                sdp = list((pair_dir / "src").glob("*.depth.png"))[0]
                spp = list((pair_dir / "src").glob("*.pose.txt"))[0]
                tcp = list((pair_dir / "tgt").glob("*.color.png"))[0]
                tdp = list((pair_dir / "tgt").glob("*.depth.png"))[0]
                tpp = list((pair_dir / "tgt").glob("*.pose.txt"))[0]

                yield dof, {"scene_name": pair_dir.parent.parent.name, "seq_name": pair_dir.parent.name}, read_framepair(scp, sdp, spp, tcp, tdp, tpp)


def iter_sn_sample(bench_dir: Path):
    """
    When iterting, limit 100 samples per level, and shuffle the samples.
    """
    bench_dir = Path(f"{bench_dir}/sn")
    for dof_dir in bench_dir.iterdir():
        if dof_dir.is_dir():
            dof = dof_dir.name.split("_")[0]
            scp_l = sorted(list(dof_dir.glob("**/source/*.jpg")))
            pair_list = [scp.parent.parent for scp in scp_l]
            random.shuffle(pair_list)
            for pair_dir in pair_list[:100]:
                scp = list((pair_dir / "source").glob("*.jpg"))[0]
                sdp = list((pair_dir / "source").glob("*.png"))[0]
                spp = list((pair_dir / "source").glob("*.txt"))[0]
                tcp = list((pair_dir / "target").glob("*.jpg"))[0]
                tdp = list((pair_dir / "target").glob("*.png"))[0]
                tpp = list((pair_dir / "target").glob("*.txt"))[0]

                yield dof, {"scene_name": pair_dir.parent.name}, read_framepair(scp, sdp, spp, tcp, tdp, tpp)


def iter_snpp_sample(bench_dir: Path):
    """
    When iterting, limit 100 samples per level, and shuffle the samples.
    """
    bench_dir = Path(f"{bench_dir}/snpp")
    for dof_dir in bench_dir.iterdir():
        if dof_dir.is_dir():
            dof = dof_dir.name.split("_")[0]
            scp_l = sorted(list(dof_dir.glob("**/source/*.jpg")))
            pair_list = [scp.parent.parent for scp in scp_l]
            random.shuffle(pair_list)
            for pair_dir in pair_list[:100]:
                scp = list((pair_dir / "source").glob("*.jpg"))[0]
                # sdp = list((pair_dir / "source").glob("*.depth.png"))[0]
                spp = list((pair_dir / "source").glob("*.txt"))[0]
                tcp = list((pair_dir / "target").glob("*.jpg"))[0]
                # tdp = list((pair_dir / "target").glob("*.depth.png"))[0]
                tpp = list((pair_dir / "target").glob("*.txt"))[0]

                yield dof, {"hash_id": pair_dir.parent.name}, read_framepair(scp, None, spp, tcp, None, tpp)


MOVE_LEFT = "Translate left"
MOVE_RIGHT = "Translate right"
MOVE_FORWARD = "Translate forward"
MOVE_BACKWARD = "Translate backward"
MOVE_UP = "Translate up"
MOVE_DOWN = "Translate down"

ROT_LEFT = "Rotate left"
ROT_RIGHT = "Rotate right"
ROT_UP = "Rotate up"
ROT_DOWN = "Rotate down"
ROT_CLOCK = "Rotate clockwise"
ROT_COUNTER = "Rotate counterclockwise"

DOF_CON_MAP = {
    "theta": "rotation along x-axis (pitch), i.e., rotate up or down",
    "phi": "rotation along y-axis (yaw), i.e., rotate left or right",
    "psi": "rotation along z-axis (roll), i.e., rotate clockwise or counterclockwise",
    "tx": "translation along x-axis (left/right)",
    "ty": "translation along y-axis (up/down)",
    "tz": "translation along z-axis (forward/backward)",
}


def _get_dof_con_and_options(dof: str):
    if dof == "theta":
        return DOF_CON_MAP[dof], [ROT_UP, ROT_DOWN]
    elif dof == "phi":
        return DOF_CON_MAP[dof], [ROT_LEFT, ROT_RIGHT]
    elif dof == "psi":
        return DOF_CON_MAP[dof], [ROT_CLOCK, ROT_COUNTER]
    elif dof == "tx":
        return DOF_CON_MAP[dof], [MOVE_LEFT, MOVE_RIGHT]
    elif dof == "ty":
        return DOF_CON_MAP[dof], [MOVE_UP, MOVE_DOWN]
    elif dof == "tz":
        return DOF_CON_MAP[dof], [MOVE_FORWARD, MOVE_BACKWARD]
    else:
        raise ValueError(f"Unknown dof: {dof}")


def _get_gt_text(dof: str, rpv: dict):
    if dof == "theta":
        return ROT_UP if rpv["value"][0] > 0 else ROT_DOWN
    elif dof == "phi":
        return ROT_RIGHT if rpv["value"][1] > 0 else ROT_LEFT
    elif dof == "psi":
        return ROT_CLOCK if rpv["value"][2] > 0 else ROT_COUNTER
    elif dof == "tx":
        return MOVE_RIGHT if rpv["value"][3] > 0 else MOVE_LEFT
    elif dof == "ty":
        return MOVE_DOWN if rpv["value"][4] > 0 else MOVE_UP
    elif dof == "tz":
        return MOVE_FORWARD if rpv["value"][5] > 0 else MOVE_BACKWARD
    else:
        raise ValueError(f"Unknown dof: {dof}")


def generate_qa(dof: str, rpv: dict):
    ### zero-shot prompt first
    con, opt_cand = _get_dof_con_and_options(dof)
    gt_text = _get_gt_text(dof, rpv)
    shuf_opt_cand = random.sample(opt_cand, 2)
    options = "\n".join(
        [f"{idx}. {opt}" for idx, opt in enumerate(shuf_opt_cand)]
    )
    prompt = prompt_template_diag.format(dof_constrain=con, options=options)

    zs = {
        "prompt": prompt,
        "gt_idx": shuf_opt_cand.index(gt_text),
        "gt_text": gt_text,
    }

    ### CoT prompt
    cot_prompt = prompt_template_diag_cot.format(dof_constrain=con, options=options)
    cot = {
        "prompt": cot_prompt,
        "gt_idx": shuf_opt_cand.index(gt_text),
        "gt_text": gt_text,
    }

    return {
        "zero-shot": zs,
        "CoT": cot,
    }
    

def main(args) -> Dataset:
    bench_dir = Path(args.bench_dir)
    sample_id = 0
    data = []
    try:
        for dof, metadata, frame_pair in iter_ss_sample(bench_dir):
            logger.info(f"Processing DOF: {dof}, sample id: {sample_id}")
            rpv = frame_pair.cal_rpv()
            qa = generate_qa(dof, rpv)
            data.append({
                "id": sample_id,
                "level": dof,
                "dataset": "7 Scenes",
                "metadata": metadata,
                "images": [frame_pair.src_frame.color, frame_pair.tgt_frame.color],
                "qa": qa,
                "depth_images": [frame_pair.src_frame.depth, frame_pair.tgt_frame.depth],
                "poses": [frame_pair.src_frame.pose, frame_pair.tgt_frame.pose],
                "relative_pose_vector": rpv,
            })
            sample_id += 1
    except Exception as e:
        logger.error(f"Error processing 7 Scenes samples: {e}")
    
    try:
        for dof, metadata, frame_pair in iter_sn_sample(bench_dir):
            logger.info(f"Processing DOF: {dof}, sample id: {sample_id}")
            rpv = frame_pair.cal_rpv()
            qa = generate_qa(dof, rpv)
            data.append({
                "id": sample_id,
                "level": dof,
                "dataset": "ScanNet",
                "metadata": metadata,
                "images": [frame_pair.src_frame.color, frame_pair.tgt_frame.color],
                "qa": qa,
                "depth_images": [frame_pair.src_frame.depth, frame_pair.tgt_frame.depth],
                "poses": [frame_pair.src_frame.pose, frame_pair.tgt_frame.pose],
                "relative_pose_vector": rpv,
            })
            sample_id += 1
    except Exception as e:
        logger.error(f"Error processing ScanNet samples: {e}")

    try: 
        for dof, metadata, frame_pair in iter_snpp_sample(bench_dir):
            logger.info(f"Processing DOF: {dof}, sample id: {sample_id}")
            rpv = frame_pair.cal_rpv()
            qa = generate_qa(dof, rpv)
            data.append({
                "id": sample_id,
                "level": dof,
                "dataset": "ScanNet++",
                "metadata": metadata,
                "images": [frame_pair.src_frame.color, frame_pair.tgt_frame.color],
                "qa": qa,
                # "depth_images": [frame_pair.src_frame.depth, frame_pair.tgt_frame.depth],
                "poses": [frame_pair.src_frame.pose, frame_pair.tgt_frame.pose],
                "relative_pose_vector": rpv,
            })
            sample_id += 1
    except Exception as e:
        logger.error(f"Error processing ScanNet++ samples: {e}")

    return Dataset.from_list(data)


if __name__ == "__main__":
    args = parse_args()
    hf_data = main(args)
    print(f"Finished with QA generation, dataset size: {len(hf_data)}")
    if args.hf_id is not None:
        hf_data.push_to_hub(args.hf_id)
    