"""Shared data schemas for evaluation results."""

from enum import Enum


class FailureAttribution(str, Enum):
    """5-way failure attribution for diagnosis-driven coaching.

    Adapted from EvoW's failure categories (evow/evaluation/openai_vlm_evaluator.py)
    with the addition of TASK_FAILURE for ill-defined instructions.
    """

    SUCCESS = "success"
    TASK_FAILURE = "task_failure"  # Task instruction is ill-defined or impossible
    SCENE_FAILURE = "scene_failure"  # Initial scene incompatible with task
    EXPERT_FAILURE = "expert_failure"  # Policy chose wrong actions
    WM_FAILURE = "wm_failure"  # World model produced inconsistent predictions
