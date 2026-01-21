"""
Some geometric utility functions for generating VRRPI dataset.
"""
import numpy as np
import re

dataset_map = {
    "ss": "7 Scenes",
    "sn": "ScanNet",
}


#### Version 1
# prompt_template = """<input>
# You are given two consecutive frames from a scene. The first image shows the **source viewpoint**, and the second image shows the **target viewpoint**. The camera moves and rotates simultaneously between these two frames. Use the visual difference between them to reason about the camera motion in 3D space.
# </input>

# <task>
# Select the correct description of the camera motion between these viewpoints.
# </task>

# <ans-candidates>
# {options}
# </ans-candidates>

# <output-format>
# <thinking>Provide your spatial reasoning here</thinking>
# <ans>Only index of your option here</ans>
# Do not output anything else.
# </output-format>"""


### Version 2
prompt_template = """<input>
You are given two consecutive frames from a scene. The first image shows the **source viewpoint**, and the second image shows the **target viewpoint**. The camera moves and rotates simultaneously between these two frames. Use the visual difference between them to reason about the camera motion in 3D space.
</input>

<task>
Select the correct description of the camera motion between these viewpoints.
</task>

<ans-candidates>
{options}
</ans-candidates>

<output-format>
Please stricly follow the format: Provide your spatial reasoning inside <thinking></thinking> XML tags, and provide **only index of your option** inside <ans></ans> XML tags, e.g., <thinking>...</thinking><ans>...</ans>.
Do not output anything else.
</output-format>"""


### Template for VRRPI-Diag
prompt_template_diag = """<input>
You are given two consecutive frames from a scene. The first image shows the **source viewpoint**, and the second image shows the **target viewpoint**. The camera movement is usually described in 6 degree of freedom (6DoF)---translation and rotation along x, y, and z axes. We now have a constrain on camera motion between source and target viewpoints, that is, the significant movement is only {dof_constrain}.
</input>

<task>
Use the visual difference between them to reason about the direction of this significant camera motion in 3D space. Select the correct description of the camera motion between these viewpoints.
</task>

<ans-candidates>
{options}
</ans-candidates>

<output-format>
Please stricly follow the format: Provide your spatial reasoning inside <thinking></thinking> XML tags, and provide **only index of your option** inside <ans></ans> XML tags, e.g., <thinking>...</thinking><ans>...</ans>.
Do not output anything else.
</output-format>"""


### Template for VRRPI-Diag
prompt_template_diag_cot = """<input>
You are given two consecutive frames from a scene. The first image shows the **source viewpoint**, and the second image shows the **target viewpoint**. The camera movement is usually described in 6 degree of freedom (6DoF)---translation and rotation along x, y, and z axes. We now have a constrain on camera motion between source and target viewpoints, that is, the significant movement is only {dof_constrain}.
</input>

<task>
Use the visual difference between them to reason about the direction of this significant camera motion in 3D space. Select the correct description of the camera motion between these viewpoints.
</task>

<ans-candidates>
{options}
</ans-candidates>

<chain-of-thought>
Note that the camera-perspective and instance-perspective is different. Let's think step by step.
</chain-of-thought>

<output-format>
Please stricly follow the format: Provide your spatial reasoning inside <thinking></thinking> XML tags, and provide **only index of your option** inside <ans></ans> XML tags, e.g., <thinking>...</thinking><ans>...</ans>.
Do not output anything else.
</output-format>"""


### 7 Scenes Intrinsic Matrix
K_ss = np.array(
    [[585., 0., 320.],
     [0., 585., 240.],
     [0., 0., 1.]],
)


def _parse_thinking_tags(output: str) -> str:
    """Extract reasoning text from <thinking> tags (robust version).

    Handles:
        1. <thinking> ... </thinking>  — standard closed tags
        2. <thinking> ... <ans>        — left-open only (stops before <ans>)
        3. ... </thinking>             — right-open only (grabs text before it)
        4. fallback: returns "" if no reasoning found.

    Args:
        output (str): The model output possibly containing <thinking> tags.

    Returns:
        str: Extracted reasoning text, stripped of whitespace.
    """

    # --- 1. Standard closed tags ---
    match = re.search(r"<thinking>(.*?)</thinking>", output, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()

    # --- 2. Left-open tag: <thinking> ... (before <ans> or end)
    match = re.search(r"<thinking>(.*?)(?:<ans>|$)", output, re.DOTALL | re.IGNORECASE)
    if match:
        content = match.group(1).strip()
        if content:
            return content

    # --- 3. Right-open tag: ... </thinking>
    match = re.search(r"(.*?)</thinking>", output, re.DOTALL | re.IGNORECASE)
    if match:
        content = match.group(1).strip()
        if content:
            return content

    # --- 4. Fallback: no thinking section found
    return ""


def _parse_ans_text_tags(output: str) -> str:
    """Extract answer text from <ans> tags (robust version).

    Handles:
        1. <ans> ... </ans>  — standard closed tags
        2. <ans> ... (no closing tag) — left-open only
        3. ... </ans> (no opening tag) — right-open only
        4. fallback: returns "" if no answer found.

    Args:
        output (str): The model output possibly containing <ans> tags.

    Returns:
        str: Extracted answer text, stripped of whitespace.
    """

    # --- 1. Standard closed tags ---
    match = re.search(r"<ans>(.*?)</ans>", output, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()

    # --- 2. Left-open tag: <ans> ... (no closing tag)
    match = re.search(r"<ans>(.*)", output, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()

    # --- 3. Right-open tag: ... </ans>
    match = re.search(r"(.*?)</ans>", output, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()

    # --- 4. Fallback: no answer section found
    return ""


def _parse_ans_tags(output: str) -> str:
    """Robustly extract numeric answer index from <ans> tags or fallback text."""
    
    def _parse_idx(text: str, take_last: bool = False) -> str:
        matches = re.findall(r"\b(\d)\b", text)
        if not matches:
            return ""
        return matches[-1].strip() if take_last else matches[0].strip()

    # --- 1. <ans> ... </ans> ---
    match = re.search(r"<ans>(.*?)</ans>", output, re.DOTALL | re.IGNORECASE)
    if match:
        return _parse_idx(match.group(1))

    # --- 2. <ans> ... (no closing tag) ---
    match = re.search(r"<ans>(.*)", output, re.DOTALL | re.IGNORECASE)
    if match:
        idx = _parse_idx(match.group(1))
        if idx:
            return idx

    # --- 3. ... </ans> (no opening tag) ---
    match = re.search(r"(.*?)</ans>", output, re.DOTALL | re.IGNORECASE)
    if match:
        idx = _parse_idx(match.group(1), take_last=True)
        if idx:
            return idx

    # --- 4. last fallback: any digit near the end ---
    match = re.findall(r"\b([0-9])\b", output)
    return match[-1].strip() if match else ""


def _pose2rpv(R: np.ndarray, t: np.ndarray) -> dict:
    """Convert rotation matrix and translation vector to relative pose dict.

    Args:
        R (np.ndarray): Rotation matrix (3x3).
        t (np.ndarray): Translation vector (3x1).

    Returns:
        dict: Relative pose with 'rotation' (quaternion) and 'translation' (3-vector).
    """

    def _rpv_text(rpv: np.ndarray):
        theta, phi, psi, tx, ty, tz = rpv
        return [
            "pitch up" if theta > 0 else "pitch down",
            "yaw right" if phi > 0 else "yaw left",
            "roll clockwise" if psi > 0 else "roll counterclockwise",
            "translate right" if tx > 0 else "translate left",
            "translate down" if ty > 0 else "translate up",
            "translate forward" if tz > 0 else "translate backward",
        ]

    theta = np.degrees(np.arctan2(R[2, 1], R[2, 2]))
    phi = np.degrees(np.arcsin(-R[2, 0]))
    psi = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
    translation = t.flatten().tolist()

    value = [theta, phi, psi, translation[0], translation[1], translation[2]]
    text = _rpv_text(value)

    return {
        "value": value,
        "text": text,
    }
    