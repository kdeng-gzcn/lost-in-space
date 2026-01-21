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
from src.dataset.base_eval import deval, trap_eval

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
        "--prompt_mode",
        type=str,
        default="zero-shot",
        choices=["zero-shot", "w/ trap"],
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


def _get_eval_fn(cfg, qa, output: str):
    if cfg.prompt_mode == "w/ trap":
        return trap_eval(
            gt_idx=qa["gt_idx"],
            trap_idx=qa["trap_idx"],
            output=output,
        )
    elif cfg.prompt_mode == "zero-shot":
        return deval(
            gt_idx=qa["gt_idx"],
            output=output,
        )
    else:
        raise NotImplementedError(f"Unknown prompt mode: {cfg.prompt_mode}")
    

def _get_output_path(cfg, output_dir: str) -> str:
    MAP = {
        "zero-shot": "zs.jsonl",
        "w/ trap": "wt.jsonl",
    }
    output_path = Path(output_dir) / MAP[cfg.prompt_mode]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return str(output_path)


def main(cfg):
    logger.info(f"Print cfg:\n{cfg}")

    vlm = load_model(cfg.vlm_id)
    dataset = load_dataset(cfg.dataset_id, "default", split="train")
    output_path = _get_output_path(cfg, cfg.output_dir)

    for data in tqdm(dataset):
        qa = data["qa"][cfg.prompt_mode]
        prompt = qa["prompt"]
        images = data["images"]
        messages = vlm.format_msg(images, prompt)
        output = vlm.qa(images, messages, max_new_tokens=4096 if "thinking" in cfg.vlm_id.lower() else 1024)
        eval_res = _get_eval_fn(cfg, qa, output)
        sample_res = {
            "id": data["id"],
            "level": data["level"],
            "dataset": data["dataset"],
            "vlm_id": cfg.vlm_id,
            "type": "vlm",
            "prompt_mode": cfg.prompt_mode,
            "output": output,
            **eval_res,
        }
        with jsonlines.open(output_path, "a") as f:
            f.write(sample_res)

    logger.info(f"Saved evaluation results to {output_path}")


if __name__ == "__main__":
    args = arg_parser()
    cfg = load_cfg(args)
    main(cfg)
    