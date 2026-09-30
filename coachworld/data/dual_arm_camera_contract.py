"""Dataset contracts for dual-arm static-camera calibration.

The calibration scripts in this repository are still audit tools, but they
must not silently share dataset-specific assumptions.  This module records the
small set of assumptions that affect K/T and EEF projection:

* which camera keys are static third-person candidates,
* where current-frame qpos comes from,
* how slot0/slot1 map to left/right arms, and
* whether the current geometry pipeline has passed review.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class DualArmCameraContract:
    dataset: str
    robot_family: str
    static_camera_priority: tuple[str, ...]
    wrist_camera_keys: tuple[str, ...]
    slot_sides: tuple[str, str]
    qpos_source: str
    qpos_layout: str
    eef_source_for_condition: str
    eef_source_for_current_projection: str
    intrinsic_policy: str
    extrinsic_policy: str
    review_status: str
    best_known_review: str
    notes: tuple[str, ...]

    def qpos_slice(self, slot: int) -> slice:
        slot = int(slot)
        if self.dataset == "robotwin":
            if slot == 0:
                return slice(0, 6)
            if slot == 1:
                return slice(6, 12)
            raise ValueError(f"dual-arm slot must be 0 or 1, got {slot}")
        if slot == 0:
            return slice(0, 7)
        if slot == 1:
            return slice(7, 14)
        raise ValueError(f"dual-arm slot must be 0 or 1, got {slot}")

    def to_summary(self) -> dict[str, Any]:
        out = asdict(self)
        slot0 = self.qpos_slice(0)
        slot1 = self.qpos_slice(1)
        out["slot_qpos_slices"] = {
            "slot0": [int(slot0.start), int(slot0.stop)],
            "slot1": [int(slot1.start), int(slot1.stop)],
        }
        return out


ROBOCOIN_CONTRACT = DualArmCameraContract(
    dataset="robocoin",
    robot_family=(
        "RoboCOIN LeRobot dual arm; dominant robot_type is AgileX/Cobot Magic, "
        "with a small number of aloha-tagged subsets"
    ),
    static_camera_priority=(
        "observation.images.cam_head_rgb",
        "observation.images.cam_high_rgb",
        "observation.images.cam_front_rgb",
    ),
    wrist_camera_keys=(
        "observation.images.cam_left_wrist_rgb",
        "observation.images.cam_right_wrist_rgb",
        "observation.images.cam_left_wrist_rgb_rgb",
        "observation.images.cam_right_wrist_rgb_rgb",
    ),
    slot_sides=("left", "right"),
    qpos_source=(
        "LeRobot parquet observation.state names; current Piper-FK smoke extracts "
        "left_arm_joint_1..6 + left_gripper_open, then right_arm_joint_1..6 + right_gripper_open"
    ),
    qpos_layout="slot0=left qpos[0:7], slot1=right qpos[7:14] after name-based extraction",
    eef_source_for_condition="eef_sim_pose_state + gripper_open_scale_action",
    eef_source_for_current_projection=(
        "qpos cache -> current RoboCOIN robot-model FK for skeleton/heatmap review "
        "(historical cam_head_rgb canary was visually useful but is not a production sidecar)"
    ),
    intrinsic_policy="VGGT K for current smoke unless an explicit fixed K is supplied",
    extrinsic_policy="episode/view-level K/T must be solved and reviewed; no raw K/T in metadata",
    review_status="historical_canary_only_not_production_sidecar",
    best_known_review="",
    notes=(
        "RoboCOIN and ViFailBack need separate episode-level adapters and review overlays.",
        "observation.state/action can be 26D or 14D; EEF condition should prefer eef_sim_pose_*.",
    ),
)


VIFAILBACK_CONTRACT = DualArmCameraContract(
    dataset="vifailback",
    robot_family="ViFailBack README states ALOHA dual-arm platform; current Piper-style projection is unaccepted",
    static_camera_priority=("cam_high",),
    wrist_camera_keys=("cam_left_wrist", "cam_right_wrist"),
    slot_sides=("left", "right"),
    qpos_source="HDF5 observations/qpos",
    qpos_layout=(
        "README mapping: slot0=left qpos[0:7], slot1=right qpos[7:14]; "
        "must still be audited episode-level before sidecar admission"
    ),
    eef_source_for_condition="not finalized; action_eef is target EEF and not current-frame truth",
    eef_source_for_current_projection=(
        "observations/qpos -> correct ALOHA/Piper-family URDF FK; current Piper approximation must not be trusted"
    ),
    intrinsic_policy="fixed cam_high K candidate or VGGT K; provenance must be written per image space",
    extrinsic_policy=(
        "episode/view-level SAM3 mask + robot-model render/refine; current Piper-style results not accepted"
    ),
    review_status="not_accepted",
    best_known_review="",
    notes=(
        "ViFailBack has not passed manual projection review.",
        "README says ALOHA platform, so robot model/URDF semantics must be audited before blaming refine steps.",
        "Known failure mode: SAM masks mix left/right arms or miss a visible slot.",
        "Dabai RGB/depth are not spatially aligned, so depth cannot be treated as aligned RGB-D truth.",
    ),
)


ROBOTWIN_CONTRACT = DualArmCameraContract(
    dataset="robotwin",
    robot_family="RoboTwin2 Aloha-AgileX simulated dual-arm benchmark",
    static_camera_priority=("head_camera", "front_camera", "video/episode*.mp4"),
    wrist_camera_keys=("left_wrist_camera", "right_wrist_camera"),
    slot_sides=("left", "right"),
    qpos_source="randomized-500 zip _traj_data/episode*.pkl left_joint_path/right_joint_path",
    qpos_layout="slot0=left fl_joint1..6, slot1=right fr_joint1..6; gripper is not present in local randomized-500 qpos archive",
    eef_source_for_condition=(
        "preferred: official data_type.endpose HDF5 left/right_endpose + gripper; "
        "fallback for current local archive: Aloha-AgileX URDF FK from 6D qpos to fl/fr_link6"
    ),
    eef_source_for_current_projection=(
        "simulator camera config + current qpos/endpose; local MP4-only archive lacks production per-frame K/T metadata"
    ),
    intrinsic_policy="use simulator camera intrinsic_cv when collecting HDF5; MP4-only review is state-space only",
    extrinsic_policy="use simulator extrinsic_cv/cam2world_gl when collecting HDF5; no real-world SAM/refine needed",
    review_status="state_space_review_ready_not_real_camera_sidecar",
    best_known_review="",
    notes=(
        "Official RoboTwin supports qpos and ee action modes; ee mode is already close to CoachWorld arm_slot_eef_pose.",
        "The local randomized-500 zip currently stores left/right 6D joint paths and one external MP4 view.",
        "Left-only and right-only episodes must set slot_exists instead of hallucinating the missing arm.",
    ),
)


ROBOMIND_AGILEX_CONTRACT = DualArmCameraContract(
    dataset="robomind_agilex",
    robot_family="RoboMind AgileX dual-arm real data",
    static_camera_priority=("camera_front", "camera_top", "camera_left", "camera_right"),
    wrist_camera_keys=("camera_left_wrist", "camera_right_wrist"),
    slot_sides=("left", "right"),
    qpos_source="HDF5 puppet/joint_position_left and puppet/joint_position_right",
    qpos_layout="slot0=left puppet/joint_position_left[:7], slot1=right puppet/joint_position_right[:7]",
    eef_source_for_condition=(
        "not accepted from current local raw puppet/end_effector_left/right; "
        "use *_align/data fields if present, otherwise verify qpos->robot-model FK in a shared world frame"
    ),
    eef_source_for_current_projection=(
        "qpos -> verified AgileX/Piper-family FK + episode/view camera K/T; raw puppet/end_effector fields are audit-only"
    ),
    intrinsic_policy="official/static intrinsics if available, otherwise estimate and record per camera/image scale",
    extrinsic_policy="episode/view-level SAM3 mask + render/refine only after qpos/FK world-frame check passes",
    review_status="not_accepted_current_raw_eef_frame",
    best_known_review="",
    notes=(
        "Current local h5_agilex_3rgb samples inspected in this audit did not expose *_align/data fields.",
        "Raw puppet/end_effector_left/right trajectories are useful for visualization but may be local/offset-reduced.",
        "RoboMind Franka remains separate and is already much cleaner because its xyz/gripper ranges are usable.",
    ),
)


GIAGAI_CONTRACT = DualArmCameraContract(
    dataset="giagai",
    robot_family="GigaAI/GigaBrain challenge real-data candidate; visually Piper-like, not yet admitted",
    static_camera_priority=("cam_high",),
    wrist_camera_keys=("cam_left_wrist", "cam_right_wrist"),
    slot_sides=("left", "right"),
    qpos_source="train/trajectories/episode_*.pkl ndarray (T,14)",
    qpos_layout="candidate: slot0=left qpos[0:7], slot1=right qpos[7:14]",
    eef_source_for_condition="candidate qpos -> Piper FK(gripper_base) -> dual mount",
    eef_source_for_current_projection="same candidate qpos/FK path; must pass projected EEF trace review before training admission",
    intrinsic_policy="VGGT K candidate unless explicit camera calibration is found later",
    extrinsic_policy="episode/view-level SAM3 mask + Piper render/refine candidate; not production sidecar yet",
    review_status="pre_camera_state_review_passed_top_right_only",
    best_known_review="",
    notes=(
        "The top-right shared rig trajectory appears plausible against raw cam_high.",
        "Bottom-left local FK looked reversed, but it is a per-slot local frame diagnostic and not a rejection criterion.",
        "Only cam_high is a static-camera candidate; wrist cameras are excluded from current static K/T line.",
    ),
)


def dual_arm_camera_contract(dataset: str) -> DualArmCameraContract:
    key = str(dataset).strip().lower()
    if key == "robocoin":
        return ROBOCOIN_CONTRACT
    if key in {"vifailback", "vifail"}:
        return VIFAILBACK_CONTRACT
    if key in {"robotwin", "robotwin2", "robotwin2_randomized_500"}:
        return ROBOTWIN_CONTRACT
    if key in {"robomind_agilex", "robomind-agilex", "robomind"}:
        return ROBOMIND_AGILEX_CONTRACT
    if key in {"giagai", "gigabrain", "gigaai"}:
        return GIAGAI_CONTRACT
    raise KeyError(f"unknown dual-arm dataset contract: {dataset!r}")


def dual_arm_contract_summary(dataset: str) -> dict[str, Any]:
    return dual_arm_camera_contract(dataset).to_summary()


def dual_arm_qpos_slice(dataset: str, slot: int) -> slice:
    return dual_arm_camera_contract(dataset).qpos_slice(slot)


def select_static_camera_key(dataset: str, available_keys: list[str] | tuple[str, ...]) -> str | None:
    available = set(str(x) for x in available_keys)
    for key in dual_arm_camera_contract(dataset).static_camera_priority:
        if key in available:
            return key
    return None
