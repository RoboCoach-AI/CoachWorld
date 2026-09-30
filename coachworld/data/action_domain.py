"""Domain and embodiment ids for CoachWorld action conditioning.

These ids are part of the model/data contract.  They are deliberately kept in
one registry so dataset builders, readers, and action encoders cannot silently
invent incompatible ids.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ActionDomainSpec:
    name: str
    domain_id: int
    dataset: str
    embodiment: str
    embodiment_id: int
    camera_setup: str
    camera_setup_id: int
    action_frame: str
    slot_layout: tuple[str, ...]
    gripper_semantics: str

    def to_payload(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "domain_id": int(self.domain_id),
            "dataset": self.dataset,
            "embodiment": self.embodiment,
            "embodiment_id": int(self.embodiment_id),
            "camera_setup": self.camera_setup,
            "camera_setup_id": int(self.camera_setup_id),
            "action_frame": self.action_frame,
            "slot_layout": list(self.slot_layout),
            "gripper_semantics": self.gripper_semantics,
        }


ACTION_DOMAIN_SPECS: tuple[ActionDomainSpec, ...] = (
    ActionDomainSpec(
        name="droid_franka_exterior2",
        domain_id=0,
        dataset="droid",
        embodiment="franka",
        embodiment_id=0,
        camera_setup="droid_exterior2_left",
        camera_setup_id=0,
        action_frame="droid_world_cartesian",
        slot_layout=("left", "inactive"),
        gripper_semantics="droid_gripper_position",
    ),
    ActionDomainSpec(
        name="robomind_franka_primary",
        domain_id=1,
        dataset="robomind",
        embodiment="franka",
        embodiment_id=0,
        # Legacy name kept stable for existing RoboMind Franka camera_right
        # video_latent roots.  New camera-aware roots should use distinct
        # camera_setup ids while sharing the same Franka action domain.
        camera_setup="robomind_franka_primary",
        camera_setup_id=1,
        action_frame="robomind_cartesian",
        slot_layout=("left", "inactive"),
        gripper_semantics="robomind_gripper_position",
    ),
    ActionDomainSpec(
        name="robomind_agilex_front",
        domain_id=2,
        dataset="robomind",
        embodiment="agilex",
        embodiment_id=1,
        camera_setup="robomind_agilex_front",
        camera_setup_id=2,
        action_frame="robomind_cartesian",
        slot_layout=("left", "right"),
        gripper_semantics="robomind_agilex_joint_position_gripper_unit_scale_5p35",
    ),
    ActionDomainSpec(
        name="robocoin_agilex_head",
        domain_id=3,
        dataset="robocoin",
        embodiment="agilex",
        embodiment_id=1,
        camera_setup="robocoin_cam_head_rgb",
        camera_setup_id=3,
        action_frame="robocoin_eef_sim_pose_state",
        slot_layout=("left", "right"),
        gripper_semantics="robocoin_gripper_open_scale_action",
    ),
    ActionDomainSpec(
        name="robocoin_agilex_high",
        domain_id=3,
        dataset="robocoin",
        embodiment="agilex",
        embodiment_id=1,
        camera_setup="robocoin_cam_high_rgb",
        camera_setup_id=7,
        action_frame="robocoin_eef_sim_pose_state",
        slot_layout=("left", "right"),
        gripper_semantics="robocoin_gripper_open_scale_action",
    ),
    ActionDomainSpec(
        name="robocoin_agilex_front",
        domain_id=3,
        dataset="robocoin",
        embodiment="agilex",
        embodiment_id=1,
        camera_setup="robocoin_cam_front_rgb",
        camera_setup_id=8,
        action_frame="robocoin_eef_sim_pose_state",
        slot_layout=("left", "right"),
        gripper_semantics="robocoin_gripper_open_scale_action",
    ),
    ActionDomainSpec(
        name="droid_franka_exterior1",
        # Same DROID Franka action semantics as exterior2.  Keep the action
        # domain shared so checkpoints trained with 4 domains can reuse the
        # learned DROID action adapter when we switch the camera setup to ext1.
        domain_id=0,
        dataset="droid",
        embodiment="franka",
        embodiment_id=0,
        camera_setup="droid_exterior1_left",
        camera_setup_id=4,
        action_frame="droid_world_cartesian",
        slot_layout=("left", "inactive"),
        gripper_semantics="droid_gripper_position",
    ),
    ActionDomainSpec(
        name="robomind_franka_camera_left",
        domain_id=1,
        dataset="robomind",
        embodiment="franka",
        embodiment_id=0,
        camera_setup="robomind_franka_camera_left",
        camera_setup_id=5,
        action_frame="robomind_cartesian",
        slot_layout=("left", "inactive"),
        gripper_semantics="robomind_gripper_position",
    ),
    ActionDomainSpec(
        name="robomind_franka_camera_top",
        domain_id=1,
        dataset="robomind",
        embodiment="franka",
        embodiment_id=0,
        camera_setup="robomind_franka_camera_top",
        camera_setup_id=6,
        action_frame="robomind_cartesian",
        slot_layout=("left", "inactive"),
        gripper_semantics="robomind_gripper_position",
    ),
    ActionDomainSpec(
        name="robotwin2_agilex_head_dual",
        domain_id=4,
        dataset="robotwin2",
        embodiment="agilex",
        embodiment_id=1,
        camera_setup="robotwin2_head_camera",
        camera_setup_id=9,
        action_frame="robotwin2_clean_endpose_wxyz",
        slot_layout=("left", "right"),
        gripper_semantics="robotwin2_normalized_gripper",
    ),
    ActionDomainSpec(
        name="robotwin2_agilex_head_left_only",
        domain_id=4,
        dataset="robotwin2",
        embodiment="agilex",
        embodiment_id=1,
        camera_setup="robotwin2_head_camera",
        camera_setup_id=9,
        action_frame="robotwin2_clean_endpose_wxyz",
        slot_layout=("left", "inactive"),
        gripper_semantics="robotwin2_normalized_gripper",
    ),
    ActionDomainSpec(
        name="robotwin2_agilex_head_right_only",
        domain_id=4,
        dataset="robotwin2",
        embodiment="agilex",
        embodiment_id=1,
        camera_setup="robotwin2_head_camera",
        camera_setup_id=9,
        action_frame="robotwin2_clean_endpose_wxyz",
        slot_layout=("inactive", "right"),
        gripper_semantics="robotwin2_normalized_gripper",
    ),
    ActionDomainSpec(
        name="vifailback_piper_cam_high_dual",
        domain_id=5,
        dataset="vifailback",
        embodiment="piper",
        embodiment_id=2,
        camera_setup="vifailback_cam_high",
        camera_setup_id=10,
        action_frame="vifailback_qpos_piper_fk_dual_mount",
        slot_layout=("left", "right"),
        gripper_semantics="vifailback_qpos_gripper_open_unit_or_5p35_scale",
    ),
    ActionDomainSpec(
        name="vifailback_piper_cam_high_left_only",
        domain_id=5,
        dataset="vifailback",
        embodiment="piper",
        embodiment_id=2,
        camera_setup="vifailback_cam_high",
        camera_setup_id=10,
        action_frame="vifailback_qpos_piper_fk_dual_mount",
        slot_layout=("left", "inactive"),
        gripper_semantics="vifailback_qpos_gripper_open_unit_or_5p35_scale",
    ),
    ActionDomainSpec(
        name="vifailback_piper_cam_high_right_only",
        domain_id=5,
        dataset="vifailback",
        embodiment="piper",
        embodiment_id=2,
        camera_setup="vifailback_cam_high",
        camera_setup_id=10,
        action_frame="vifailback_qpos_piper_fk_dual_mount",
        slot_layout=("inactive", "right"),
        gripper_semantics="vifailback_qpos_gripper_open_unit_or_5p35_scale",
    ),
    ActionDomainSpec(
        name="libero_franka_agentview",
        domain_id=6,
        dataset="libero",
        embodiment="franka",
        embodiment_id=0,
        camera_setup="libero_agentview",
        camera_setup_id=11,
        action_frame="libero_robosuite_world_axis_angle",
        slot_layout=("left", "inactive"),
        gripper_semantics="libero_panda_gripper_qpos_first_joint_unit_0p04",
    ),
    ActionDomainSpec(
        name="giagai_piper_cam_high_dual",
        domain_id=7,
        dataset="giagai",
        embodiment="piper",
        embodiment_id=2,
        camera_setup="giagai_cam_high",
        camera_setup_id=12,
        action_frame="giagai_qpos_piper_fk_dual_mount",
        slot_layout=("left", "right"),
        gripper_semantics="giagai_qpos_gripper_open_unit_or_5p35_scale",
    ),
    ActionDomainSpec(
        name="giagai_piper_cam_high_left_only",
        domain_id=7,
        dataset="giagai",
        embodiment="piper",
        embodiment_id=2,
        camera_setup="giagai_cam_high",
        camera_setup_id=12,
        action_frame="giagai_qpos_piper_fk_dual_mount",
        slot_layout=("left", "inactive"),
        gripper_semantics="giagai_qpos_gripper_open_unit_or_5p35_scale",
    ),
    ActionDomainSpec(
        name="giagai_piper_cam_high_right_only",
        domain_id=7,
        dataset="giagai",
        embodiment="piper",
        embodiment_id=2,
        camera_setup="giagai_cam_high",
        camera_setup_id=12,
        action_frame="giagai_qpos_piper_fk_dual_mount",
        slot_layout=("inactive", "right"),
        gripper_semantics="giagai_qpos_gripper_open_unit_or_5p35_scale",
    ),
    ActionDomainSpec(
        name="lab_franka_external",
        domain_id=8,
        dataset="lab_franka",
        embodiment="franka",
        embodiment_id=0,
        camera_setup="lab_franka_0726_static_external",
        camera_setup_id=13,
        action_frame="franka_base_robotiq_tcp",
        slot_layout=("left", "inactive"),
        gripper_semantics="lab_franka_raw_3_open_230_closed_to_open_unit",
    ),
)

_BY_NAME = {spec.name: spec for spec in ACTION_DOMAIN_SPECS}
_BY_ID = {spec.domain_id: spec for spec in ACTION_DOMAIN_SPECS}


def num_action_domains() -> int:
    return max(_BY_ID) + 1


def domain_spec(name: str) -> ActionDomainSpec:
    try:
        return _BY_NAME[str(name)]
    except KeyError as exc:
        raise KeyError(f"unknown action domain {name!r}; known={sorted(_BY_NAME)}") from exc


def droid_domain_for_video_keys(video_keys: list[str]) -> ActionDomainSpec:
    keys = [str(k) for k in video_keys]
    if keys == ["observation.images.exterior_1_left"]:
        return domain_spec("droid_franka_exterior1")
    if keys == ["observation.images.exterior_2_left"]:
        return domain_spec("droid_franka_exterior2")
    raise ValueError(
        "No registered DROID action domain for video_keys="
        f"{keys!r}. Register the camera setup in coachworld.data.action_domain first."
    )


def robomind_domain_for_camera(embodiment: str, camera: str) -> ActionDomainSpec:
    emb = str(embodiment)
    cam = str(camera).split(".")[-1]
    if emb == "h5_franka_3rgb":
        if cam == "camera_right":
            return domain_spec("robomind_franka_primary")
        if cam == "camera_left":
            return domain_spec("robomind_franka_camera_left")
        if cam == "camera_top":
            return domain_spec("robomind_franka_camera_top")
    if emb == "h5_agilex_3rgb" and cam == "camera_front":
        return domain_spec("robomind_agilex_front")
    raise ValueError(
        f"No registered RoboMind action domain for embodiment={emb!r}, camera={cam!r}. "
        "Register it in coachworld.data.action_domain before building video_latent."
    )


def robocoin_domain_for_video_key(video_key: str) -> ActionDomainSpec:
    key = str(video_key)
    if key == "observation.images.cam_head_rgb":
        return domain_spec("robocoin_agilex_head")
    if key == "observation.images.cam_high_rgb":
        return domain_spec("robocoin_agilex_high")
    if key == "observation.images.cam_front_rgb":
        return domain_spec("robocoin_agilex_front")
    raise ValueError(
        f"No registered RoboCOIN action domain for video_key={key!r}. "
        "Register it in coachworld.data.action_domain before building video_latent."
    )


def robotwin2_domain_for_camera(camera: str, pattern: str = "dual") -> ActionDomainSpec:
    cam = str(camera)
    if cam == "head_camera":
        pat = str(pattern)
        if pat == "dual":
            return domain_spec("robotwin2_agilex_head_dual")
        if pat == "left_only":
            return domain_spec("robotwin2_agilex_head_left_only")
        if pat == "right_only":
            return domain_spec("robotwin2_agilex_head_right_only")
        raise ValueError(f"unsupported RoboTwin2 arm pattern={pat!r}")
    raise ValueError(
        f"No registered RoboTwin2 action domain for camera={cam!r}. "
        "Only head_camera is currently admitted into the CoachWorld training pool."
    )


def vifailback_domain_for_camera(camera: str, pattern: str = "dual") -> ActionDomainSpec:
    cam = str(camera).split("/")[-1]
    if cam == "cam_high":
        pat = str(pattern)
        if pat == "dual":
            return domain_spec("vifailback_piper_cam_high_dual")
        if pat == "left_only":
            return domain_spec("vifailback_piper_cam_high_left_only")
        if pat == "right_only":
            return domain_spec("vifailback_piper_cam_high_right_only")
        raise ValueError(f"unsupported ViFailBack arm pattern={pat!r}")
    raise ValueError(
        f"No registered ViFailBack action domain for camera={cam!r}. "
        "Only cam_high is currently admitted for the filtered Piper subset."
    )


def libero_domain_for_camera(camera: str) -> ActionDomainSpec:
    cam = str(camera).split("/")[-1]
    if cam == "agentview":
        return domain_spec("libero_franka_agentview")
    raise ValueError(
        f"No registered LIBERO action domain for camera={cam!r}. "
        "Only static agentview is currently admitted."
    )


def giagai_domain_for_camera(camera: str, pattern: str = "dual") -> ActionDomainSpec:
    cam = str(camera).split("/")[-1]
    if cam == "cam_high":
        pat = str(pattern)
        if pat == "dual":
            return domain_spec("giagai_piper_cam_high_dual")
        if pat == "left_only":
            return domain_spec("giagai_piper_cam_high_left_only")
        if pat == "right_only":
            return domain_spec("giagai_piper_cam_high_right_only")
        raise ValueError(f"unsupported GigaAI arm pattern={pat!r}")
    raise ValueError(
        f"No registered GigaAI action domain for camera={cam!r}. "
        "Only cam_high is currently admitted as a candidate."
    )


def validate_domain_payload(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError(f"domain must be an object, got {type(payload).__name__}")
    required = {
        "name",
        "domain_id",
        "dataset",
        "embodiment",
        "embodiment_id",
        "camera_setup",
        "camera_setup_id",
        "action_frame",
        "slot_layout",
        "gripper_semantics",
    }
    missing = required - set(payload)
    if missing:
        raise ValueError(f"domain missing keys: {sorted(missing)}")
    spec = domain_spec(str(payload["name"]))
    expected = spec.to_payload()
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(
                f"domain.{key}={payload.get(key)!r} does not match registry "
                f"{spec.name}.{key}={value!r}"
            )
    return payload
