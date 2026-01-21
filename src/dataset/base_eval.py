from typing import Union, List

import numpy as np

from .utils import _parse_ans_tags, _parse_thinking_tags, _pose2rpv, _parse_ans_text_tags

def qeval(gt_idx: int, output: str) -> bool:
    """Quick Evaluate the answer from VLM output.

    Args:
        gt_idx (int): The ground truth index.
        output (str): The output string from VLM.

    Returns:
        bool: Whether the answer is correct.
    """
    pred_idx = _parse_ans_tags(output)
    return pred_idx == str(gt_idx)


def deval(gt_idx: int, output: str) -> str:
    """Detailed Evaluate the output from VLM.

    Args:
        gt_idx (int): The ground truth index.
        output (str): The output string from VLM.

    Returns:
        str: The extracted reasoning text.
    """
    thinking = _parse_thinking_tags(output)
    pred_idx = _parse_ans_tags(output)
    return {
        "gt_idx": gt_idx,
        "pred_idx": pred_idx,
        "is_correct": pred_idx == str(gt_idx),
        "has_thinking": len(thinking) > 0,
        "has_ans": len(pred_idx) > 0,
        "thinking": thinking,
    }


def list_eval(gt_list: List[str], output: str) -> bool:
    """Evaluate whether the text answer from VLM matches the ground truth.

    Args:
        gt_text (str): The ground truth text answer.
        output (str): The output string from VLM.

    Returns:
        bool: Whether the text answer matches the ground truth.
    """
    pred_text = _parse_ans_text_tags(output)
    is_correct = any(gt_text.lower() in pred_text.lower() for gt_text in gt_list)
    return {
        "gt_list": gt_list,
        "pred_text": pred_text,
        "is_correct": is_correct,
        "has_ans": bool(pred_text),
    }


def cv_eval(rpv: dict, R: Union[np.ndarray, None], t: Union[np.ndarray, None]) -> bool:
    """Evaluate whether the relative pose estimation is correct.

    Args:
        rpv (dict): The relative pose estimation result containing 'rotation' and 'translation'.
        R (np.ndarray): The ground truth rotation matrix (3x3).
        t (np.ndarray): The ground truth translation vector (3x1).
        threshold (float): The error threshold to consider the estimation as correct.

    Returns:
        bool: Whether the relative pose estimation is correct.
    """
    if R is None or t is None:
        return {
            "gt_rpv": rpv,
            "pred_rpv": None,
            "is_correct": None,
            "is_phi_correct": None,
            "is_tx_correct": None,
        }

    pred_rpv = _pose2rpv(R, t)

    _, gt_phi, _, gt_tx, _, _ = rpv["value"]
    _, pred_phi, _, pred_tx, _, _ = pred_rpv["value"]

    is_phi_correct = bool(np.sign(gt_phi) == np.sign(pred_phi))
    is_tx_correct = bool(np.sign(gt_tx) == np.sign(pred_tx))

    return {
        "gt_rpv": rpv,
        "pred_rpv": pred_rpv,
        "is_correct": is_phi_correct and is_tx_correct,
        "is_phi_correct": is_phi_correct,
        "is_tx_correct": is_tx_correct,
    }


def trap_eval(gt_idx: int, trap_idx: int, output: str) -> dict:
    """Evaluate whether model mistakenly selects the trap option.

    Args:
        gt_idx (int): Ground truth option index.
        trap_idx (int): The trap option index.
        output (str): The raw model output string.

    Returns:
        dict: {
            "gt_idx": str,
            "trap_idx": str,
            "pred_idx": str,
            "is_correct": bool,
            "is_trap_selected": bool,
            "has_thinking": bool(thinking),
            "has_ans": bool(pred_idx),
            "thinking": thinking,
        }
    """
    thinking = _parse_thinking_tags(output)
    pred_idx = _parse_ans_tags(output)

    return {
        "gt_idx": str(gt_idx),
        "pred_idx": pred_idx,        
        "is_correct": pred_idx == str(gt_idx),
        "trap_idx": str(trap_idx),
        "is_trap_selected": pred_idx == str(trap_idx),
        "has_thinking": bool(thinking),
        "has_ans": bool(pred_idx),
        "thinking": thinking,
    }


def consistency_eval(gt_idx_orig: int, gt_idx_rev: int, output_orig: str, output_rev: str) -> dict:
    """Evaluate the consistency of model answers under image order reversal."""
    thinking_orig = _parse_thinking_tags(output_orig)
    pred_idx_orig = _parse_ans_tags(output_orig)

    thinking_rev = _parse_thinking_tags(output_rev)
    pred_idx_rev = _parse_ans_tags(output_rev)

    has_pred_orig = bool(pred_idx_orig)
    has_pred_rev = bool(pred_idx_rev)
    has_both = has_pred_orig and has_pred_rev

    is_consistent = (pred_idx_orig != pred_idx_rev) if has_both else None
    is_correct_orig = (pred_idx_orig == str(gt_idx_orig)) if has_pred_orig else None
    is_correct_rev = (pred_idx_rev == str(gt_idx_rev)) if has_pred_rev else None
    is_all_correct = (is_correct_orig and is_correct_rev) if has_both else None

    return {
        "gt_idx_orig": str(gt_idx_orig),
        "pred_idx_orig": pred_idx_orig if has_pred_orig else None,
        "gt_idx_rev": str(gt_idx_rev),
        "pred_idx_rev": pred_idx_rev if has_pred_rev else None,
        "is_consistent": is_consistent,
        "is_all_correct": is_all_correct,
        "is_correct_orig": is_correct_orig,
        "is_correct_rev": is_correct_rev,
        "has_thinking_orig": bool(thinking_orig),
        "has_thinking_rev": bool(thinking_rev),
        "has_ans_orig": has_pred_orig,
        "has_ans_rev": has_pred_rev,
        "thinking_orig": thinking_orig if thinking_orig else None,
        "thinking_rev": thinking_rev if thinking_rev else None,
    }