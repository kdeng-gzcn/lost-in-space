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


HF_ID = "kdeng03/cross-image-sr"
BENCH_FILE = "../spatial-reasoning-of-LMs/result/new-error-diagnosis/error-diag-s2.jsonl"


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

            prompt = obj["prompt0"]
            # gt_idx = obj["cor_idx_shuf"]
            gt_list = obj["label"]
            # level = obj["type"]

            zs = {
                "prompt": prompt,
                "gt_list": gt_list,
            }

            prompt = obj["prompt1"]
            # gt_idx = obj["cor_idx_shuf"]
            ro = obj["ref_obj"]
            gt_list = obj["label"]
            # level = obj["type"]

            wro = {
                "prompt": prompt,
                "gt_list": gt_list,
                "ref_obj": ro,
            }

            qa = {
                "w/o ref obj": zs,
                "w/ ref obj": wro,
            }
            
            yield images, qa


def main():
    sample_id = 0
    data = []
    for images, qa in iter_sample():
        logger.info(f"Processing sample id: {sample_id}")

        data.append({
            "id": sample_id,
            # "level": level,
            "images": images,
            "qa": qa,
        })
        sample_id += 1

    return Dataset.from_list(data)


if __name__ == "__main__":
    hf_data = main()
    hf_data.push_to_hub(HF_ID)
    