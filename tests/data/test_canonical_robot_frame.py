#!/usr/bin/env python3
"""Unit checks for canonical robot-frame transforms."""

from __future__ import annotations

import unittest

import numpy as np

from coachworld.data.canonical_robot_frame import (
    camera_from_canonical,
    libero_base_from_world,
    matrix_to_rot6d,
    rot6d_to_matrix,
    transform_arm_slot_condition,
    transform_xyz,
)


class CanonicalRobotFrameTest(unittest.TestCase):
    def test_libero_robot_base_translation(self) -> None:
        xml = '<mujoco><worldbody><body name="robot0_base" pos="-0.66 0 0.912"/></worldbody></mujoco>'
        base_from_world = libero_base_from_world(xml)
        point_world = np.array([[-0.16, 0.2, 1.212]], dtype=np.float64)
        np.testing.assert_allclose(
            transform_xyz(point_world, base_from_world),
            [[0.5, 0.2, 0.3]],
            atol=1.0e-10,
        )

    def test_rot6d_round_trip(self) -> None:
        rotation = np.array(
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        np.testing.assert_allclose(rot6d_to_matrix(matrix_to_rot6d(rotation)), rotation, atol=1.0e-10)

    def test_condition_and_camera_projection_invariance(self) -> None:
        condition = np.zeros((2, 2, 11), dtype=np.float32)
        condition[:, 0, :3] = [[0.4, 0.1, 0.3], [0.5, -0.1, 0.35]]
        condition[:, 0, 3:9] = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        condition[:, 0, 9] = 0.5
        condition[:, 0, 10] = 1.0
        target_from_source = np.array(
            [[0.0, -1.0, 0.0, 0.2], [1.0, 0.0, 0.0, -0.3], [0.0, 0.0, 1.0, 0.1], [0.0, 0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        camera_from_source = np.array(
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        canonical = transform_arm_slot_condition(condition, target_from_source)
        camera_from_target = camera_from_canonical(camera_from_source, target_from_source)
        source_points = np.c_[condition[:, 0, :3], np.ones(2)]
        target_points = np.c_[canonical[:, 0, :3], np.ones(2)]
        np.testing.assert_allclose(
            (camera_from_source @ source_points.T).T,
            (camera_from_target @ target_points.T).T,
            atol=1.0e-6,
        )
        np.testing.assert_array_equal(canonical[:, 1], 0.0)


if __name__ == "__main__":
    unittest.main()
