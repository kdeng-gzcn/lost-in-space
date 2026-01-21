import argparse
import logging

import numpy as np
from pathlib import Path
import jsonlines
from datasets import load_dataset
from yacs.config import CfgNode as CN
from tqdm import tqdm

from config.default import cfg # default configuration file
from src.logging.logging_config import setup_logging
from src.model.cv import SIFT, LoFTR
from src.dataset.base_eval import cv_eval

setup_logging()
logger = logging.getLogger(__name__)


def arg_parser():
    parser = argparse.ArgumentParser(
        description=""
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["SIFT", "LoFTR"],
        help="",
    )
    parser.add_argument(
        "--dataset_id",
        type=str,
        required=True,
        help="",
    )
    return parser.parse_args()


def _load_args(cfg, args):
    arg_dict = vars(args)
    arg_cfg = CN(arg_dict)
    cfg.merge_from_other_cfg(arg_cfg)
    return cfg


def load_cfg(args):
    _cfg = _load_args(cfg, args)
    # Potential additional cfg, e.g., merge from file
    return _cfg


def load_model(cfg, model_name: str):
    if model_name == "SIFT":
        return SIFT(cfg=cfg)
    elif model_name == "LoFTR":
        return LoFTR(cfg=cfg)
    else:
        raise NotImplementedError(f"Unknown model: {model_name}")


def _get_output_path(output_dir: str) -> str:
    output_path = Path(output_dir) / "cv.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return str(output_path)


def main(cfg):
    logger.info(f"Print cfg:\n{cfg}")

    model = load_model(cfg, cfg.model)
    dataset = load_dataset(cfg.dataset_id, "default", split="train")
    output_path = _get_output_path(cfg.output_dir)

    for data in tqdm(dataset):
        images = data["images"]
        K = np.array(data["intrinsic"])
        R, t = model.pipeline(images, K)
        eval_res = cv_eval(data["relative_pose_vector"], R, t)
        sample_res = {
            "id": data["id"],
            "level": data["level"],
            "dataset": data["dataset"],
            "vlm_id": cfg.model,
            "type": "cv",
            "prompt_mode": None,
            **eval_res,
        }
        with jsonlines.open(output_path, "a") as f:
            f.write(sample_res)

    logger.info(f"Saved evaluation results to {output_path} with model {cfg.model}")


if __name__ == "__main__":
    args = arg_parser()
    cfg = load_cfg(args)
    main(cfg)
    