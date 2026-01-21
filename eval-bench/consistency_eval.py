import argparse
import logging

from pathlib import Path
import jsonlines
from datasets import load_dataset
from yacs.config import CfgNode as CN
from tqdm import tqdm

from config.default import cfg # default configuration file
from src.logging.logging_config import setup_logging
from src.model.utils import load_model
from src.dataset.base_eval import consistency_eval

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
        "--vlm_id",
        type=str,
        required=True,
        help="",
    )
    parser.add_argument(
        "--dataset_id",
        type=str,
        required=True,
        help="",
    )
    parser.add_argument(
        "--subset",
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
    

def _get_output_path(output_dir: str) -> str:
    output_path = Path(output_dir) / "consistency.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return str(output_path)


def dual_infer(vlm, images, prompt: str):
    images_orig = images
    images_reversed = list(reversed(images_orig))

    messages_orig = vlm.format_msg(images_orig, prompt)
    output_orig = vlm.qa(images_orig, messages_orig)

    messages_reversed = vlm.format_msg(images_reversed, prompt)
    output_reversed = vlm.qa(images_reversed, messages_reversed, max_new_tokens=4096 if "thinking" in cfg.vlm_id.lower() else 1024)
    return output_orig, output_reversed


def main(cfg):
    logger.info(f"Print cfg:\n{cfg}")

    vlm = load_model(cfg.vlm_id)
    dataset = load_dataset(cfg.dataset_id, cfg.subset, split="train")
    output_path = _get_output_path(cfg.output_dir)

    for data in tqdm(dataset):
        qa = data["qa"]["zero-shot"]
        prompt = qa["prompt"]
        images = data["images"]
        output, output_rev = dual_infer(vlm, images, prompt)
        eval_res = consistency_eval(qa["gt_idx"],  1-qa["gt_idx"], output, output_rev)
        sample_res = {
            "id": data["id"],
            "level": data["level"],
            "dataset": data["dataset"],
            "vlm_id": cfg.vlm_id,
            "type": "vlm",
            "prompt_mode": "zero-shot",
            "output_orig": output,
            "output_rev": output_rev,
            **eval_res,
        }
        with jsonlines.open(output_path, "a") as f:
            f.write(sample_res)

    logger.info(f"Saved evaluation results to {output_path}")


if __name__ == "__main__":
    args = arg_parser()
    cfg = load_cfg(args)
    main(cfg)
    