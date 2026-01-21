import logging
from typing import List, Any, Union
from pathlib import Path

import numpy as np
from datasets import Dataset
import jsonlines
from PIL import Image

from src.dataset.base_frame import Frame, FramePair
from src.logging.logging_config import setup_logging

### Set Seed
import random
random.seed(42)
np.random.seed(42)

setup_logging()
logger = logging.getLogger(__name__)


HF_ID = "kdeng03/intra-image-sr"
BENCH_FILE = "../spatial-reasoning-of-LMs/result/new-error-diagnosis/basic-rela-v2.jsonl"


def iter_sample():
    """
    Preprocessed benchmark file in jsonlines format.
    """
    bench_file = Path(BENCH_FILE)
    with jsonlines.open(bench_file) as reader:
        for obj in reader:
            img_path = Path("../spatial-reasoning-of-LMs") / Path(obj["img_path"])
            img = Image.open(img_path).convert("RGB")
            prompt = obj["prompt_shuf"]
            gt_idx = obj["cor_idx_shuf"]
            gt_text = obj["cor_rela"]
            level = obj["type"]
            qa = {
                "prompt": prompt,
                "gt_idx": gt_idx,
                "gt_text": gt_text,
            }
            yield img, qa, level


def main():
    sample_id = 0
    data = []
    for img, qa, level in iter_sample():
        logger.info(f"Processing type: {level}, sp: {qa['gt_text']}, sample id: {sample_id}")

        data.append({
            "id": sample_id,
            "level": level,
            "images": img,
            "qa": qa,
        })
        sample_id += 1

    return Dataset.from_list(data)


if __name__ == "__main__":
    hf_data = main()
    