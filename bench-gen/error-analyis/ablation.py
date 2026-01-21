import logging
from typing import List, Any, Union
from pathlib import Path

import numpy as np
from datasets import Dataset
import jsonlines
from PIL import Image

# from src.dataset.base_frame import Frame, FramePair
from src.logging.logging_config import setup_logging

### Set Seed
import random
random.seed(42)
np.random.seed(42)

setup_logging()
logger = logging.getLogger(__name__)


HF_ID = "kdeng03/ablation-VRRPI-Diag"
BENCH_FILE = "../spatial-reasoning-of-LMs/result/new-error-diagnosis/error-diag-s3.jsonl"


def _get_gt_text(dof: str, sign: str) -> str:
    if dof == "theta":
        if sign == "+":
            return "Rotate up"
        else:
            return "Rotate down"
    elif dof == "phi":
        if sign == "+":
            return "Rotate right"
        else:
            return "Rotate left"
    elif dof == "psi":
        if sign == "+":
            return "Rotate clockwise"
        else:
            return "Rotate counterclockwise"
    elif dof == "tx":
        if sign == "+":
            return "Translate right"
        else:
            return "Translate left"
    elif dof == "ty":
        if sign == "+":
            return "Translate down"
        else:
            return "Translate up"
    elif dof == "tz":
        if sign == "+":
            return "Translate forward"
        else:
            return "Translate backward"


def iter_sample():
    """
    Preprocessed benchmark file in jsonlines format.
    """
    bench_file = Path(BENCH_FILE)
    with jsonlines.open(bench_file) as reader:
        for obj in reader:
            src_img_path = Path("../spatial-reasoning-of-LMs") / Path(obj["src_img_path"])
            tgt_img_path = Path("../spatial-reasoning-of-LMs") / Path(obj["tgt_img_path"])
            img1 = Image.open(src_img_path).convert("RGB")
            img2 = Image.open(tgt_img_path).convert("RGB")
            images = [img1, img2]
            dof = obj["dof"]
            sign = obj["sign"]

            gt_text = _get_gt_text(dof, sign)

            ### zs
            prompt = obj["prompt0"]
            gt_idx = obj["cor_idx"]
            zs = {
                "prompt": prompt,
                "gt_idx": gt_idx,
                "gt_text": gt_text,
            }
            ### h
            prompt = obj["prompt1"]
            h = {
                "prompt": prompt,
                "gt_idx": gt_idx,
                "gt_text": gt_text,
            }
            ### h+
            prompt = obj["prompt2"]
            hp = {
                "prompt": prompt,
                "gt_idx": gt_idx,
                "gt_text": gt_text,
            }
            ### h++
            prompt = obj["prompt3"]
            hpp = {
                "prompt": prompt,
                "gt_idx": gt_idx,
                "gt_text": gt_text,
            }

            qa = {
                "zero-shot": zs,
                "H": h,
                "H+": hp,
                "H++": hpp,
            }
            yield dof, images, qa


def main():
    sample_id = 0
    data = []
    for dof, imgs, qa in iter_sample():
        logger.info(f"Processing type: {dof}, sp: {qa['zero-shot']['gt_text']}, sample id: {sample_id}")

        data.append({
            "id": sample_id,
            "level": dof,
            "images": imgs,
            "qa": qa,
        })
        sample_id += 1

    return Dataset.from_list(data)


if __name__ == "__main__":
    hf_data = main()
    hf_data.push_to_hub(HF_ID)
    