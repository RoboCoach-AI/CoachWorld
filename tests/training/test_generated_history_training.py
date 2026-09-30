from types import SimpleNamespace

import torch

from coachworld.data.video_latent_collate import collate_video_latent_batch
from coachworld.world_model.action_contract import (
    latent_window_indices_for_first_future_frame,
)
from coachworld.world_model.generated_history import (
    generated_history_step_is_active,
    select_generated_history_unroll_chunks,
)
from coachworld.world_model.trainer import WMTrainer


def test_sparse_next_window_consumes_generated_source_latents() -> None:
    source = latent_window_indices_for_first_future_frame(
        first_future_frame=40,
        video_length=200,
        history_frames=5,
        future_frames=3,
        history_selector="sparse",
        history_dilation=2,
    )
    following = latent_window_indices_for_first_future_frame(
        first_future_frame=52,
        video_length=200,
        history_frames=5,
        future_frames=3,
        history_selector="sparse",
        history_dilation=2,
    )

    assert source == [1, 3, 5, 7, 9, 10, 11, 12]
    assert following == [4, 6, 8, 10, 12, 13, 14, 15]
    assert set(source[5:]) & set(following[:5]) == {10, 12}


def test_generated_history_replaces_only_matching_eligible_slots() -> None:
    trainer = object.__new__(WMTrainer)
    trainer.device = torch.device("cpu")
    trainer.dtype = torch.float32
    trainer.tcfg = SimpleNamespace(
        generated_history_dataset_keys=["droid_ext1"],
        generated_history_blend=0.5,
        generated_history_denoise_steps=2,
    )
    generated = torch.tensor(
        [
            [[[[10.0]], [[20.0]], [[30.0]]]],
            [[[[40.0]], [[50.0]], [[60.0]]]],
        ]
    )
    trainer._generate_history_source_future = lambda _batch: generated

    next_latent = torch.zeros(2, 8, 1, 1, 1)
    batch = {
        "dataset_key": ["droid_ext1", "droid_ext1"],
        "future_latent_ids": torch.tensor(
            [[10, 11, 12], [10, 11, 12]]
        ),
        "target": {
            "latent": next_latent,
            "history_latent_ids": torch.tensor(
                [[4, 6, 8, 10, 12], [4, 6, 8, 10, 12]]
            ),
            "_generated_history_chain_active_mask": torch.tensor(
                [True, False]
            ),
        },
    }

    result, logs = trainer._advance_generated_history_batch(
        batch,
        batch["target"],
    )

    assert result["latent"][0, 3, 0, 0, 0].item() == 5.0
    assert result["latent"][0, 4, 0, 0, 0].item() == 15.0
    assert torch.count_nonzero(result["latent"][1]).item() == 0
    assert result["_generated_history_sample_mask"].tolist() == [True, False]
    assert logs["generated_history_replaced_frames"] == 2.0
    assert logs["generated_history_sample_fraction"] == 0.5


def test_generated_history_unroll_metrics_average_all_steps() -> None:
    trainer = object.__new__(WMTrainer)
    trainer._global_step = 0
    trainer.tcfg = SimpleNamespace(
        generated_history_unroll_chunks=[2],
        generated_history_seed=11,
    )
    step_logs = iter(
        [
            {
                "generated_history_active": 1.0,
                "generated_history_eligible_fraction": 1.0,
                "generated_history_sample_fraction": 0.5,
                "generated_history_replaced_frames": 2.0,
                "generated_history_delta": 0.25,
            },
            {
                "generated_history_active": 1.0,
                "generated_history_eligible_fraction": 0.0,
                "generated_history_sample_fraction": 0.0,
                "generated_history_replaced_frames": 0.0,
                "generated_history_delta": 0.0,
            },
        ]
    )
    trainer._advance_generated_history_batch = (
        lambda source, _target: (source, next(step_logs))
    )
    batch = {
        "generated_history_chain": [{}, {}],
        "generated_history_available_chunks": torch.tensor([2, 1]),
    }

    _, logs = trainer._build_generated_history_batch(batch)

    assert logs["generated_history_eligible_fraction"] == 0.5
    assert logs["generated_history_sample_fraction"] == 0.25
    assert logs["generated_history_replaced_frames"] == 2.0
    assert logs["generated_history_delta"] == 0.25
    assert logs["generated_history_unroll_chunks"] == 2.0
    assert logs["generated_history_available_chunks"] == 1.5


def test_generated_history_unroll_depth_is_deterministic_and_available() -> None:
    expected_prefix = [2, 4, 8, 1, 2, 4, 8, 1]
    actual_prefix = [
        select_generated_history_unroll_chunks(
            global_step=step,
            configured_horizons=[1, 2, 4, 8],
            available_chunks=8,
            seed=11,
        )
        for step in range(len(expected_prefix))
    ]
    assert actual_prefix == expected_prefix

    seen = set()
    for step in range(40):
        first = select_generated_history_unroll_chunks(
            global_step=step,
            configured_horizons=[1, 2, 4, 8],
            available_chunks=8,
            seed=11,
        )
        second = select_generated_history_unroll_chunks(
            global_step=step,
            configured_horizons=[1, 2, 4, 8],
            available_chunks=8,
            seed=11,
        )
        assert first == second
        seen.add(first)
    assert seen == {1, 2, 4, 8}

    assert select_generated_history_unroll_chunks(
        global_step=3,
        configured_horizons=[1, 2, 4, 8],
        available_chunks=3,
        seed=11,
    ) in {1, 2}


def test_generated_history_collate_preserves_long_items_with_active_masks() -> None:
    def window() -> dict:
        return {
            "latent": torch.zeros(8, 1, 1, 1),
            "action": torch.zeros(4, 1),
            "text": "",
            "dataset_key": "test",
            "dataset_index": 0,
            "domain_id": 0,
            "embodiment_id": 0,
            "camera_setup_id": 0,
            "history_latent_ids": torch.arange(5),
            "future_latent_ids": torch.arange(5, 8),
        }

    long_item = window()
    long_item["generated_history_chain"] = [window() for _ in range(8)]
    long_item["generated_history_max_unroll_chunks"] = 8
    short_item = window()
    short_item["generated_history_chain"] = [window() for _ in range(2)]
    short_item["generated_history_max_unroll_chunks"] = 8

    result = collate_video_latent_batch([long_item, short_item])
    assert len(result["generated_history_chain"]) == 8
    assert result["generated_history_chain"][1][
        "_generated_history_chain_active_mask"
    ].tolist() == [True, True]
    assert result["generated_history_chain"][2][
        "_generated_history_chain_active_mask"
    ].tolist() == [True, False]
    assert result["generated_history_chain"][7][
        "_generated_history_chain_active_mask"
    ].tolist() == [True, False]


def test_generated_history_schedule_is_step_deterministic() -> None:
    kwargs = {
        "enabled": True,
        "start_step": 20,
        "probability": 0.35,
        "seed": 7,
    }
    assert generated_history_step_is_active(global_step=19, **kwargs) is False

    decisions = []
    for step in range(20, 120):
        first = generated_history_step_is_active(global_step=step, **kwargs)
        second = generated_history_step_is_active(global_step=step, **kwargs)
        assert first == second
        decisions.append(first)
    assert 20 <= sum(decisions) <= 50
