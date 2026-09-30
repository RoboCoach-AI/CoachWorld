"""Abstract evaluator interface.

An evaluator takes an episode (video + instruction) and produces a score
plus failure attribution. This drives the coaching loop's diagnosis step.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

from coachworld.evaluator.schemas import FailureAttribution


@dataclass
class EvalInput:
    """Input to the evaluator."""

    instruction: str
    video_path: Optional[str] = None  # Path to episode video
    frames: Optional[list] = None  # Or direct frame list: list of (H, W, 3)
    success_criteria: Optional[list[str]] = None
    task_id: Optional[str] = None  # Used by TwoStageEvaluator for reference latent lookup


@dataclass
class EvalResult:
    """Evaluation result with failure attribution."""

    reward: float  # Scalar reward in [0, 1]
    success: bool
    failure_attribution: FailureAttribution
    physical_consistency: float = 0.0  # 0-1 score for WM quality
    reasoning: str = ""  # VLM explanation


class BaseEvaluator(ABC):
    """Abstract VLM evaluator: episode -> score + failure attribution."""

    @abstractmethod
    def evaluate(self, eval_input: EvalInput) -> EvalResult:
        """Evaluate an episode for task success and failure attribution."""
        ...
