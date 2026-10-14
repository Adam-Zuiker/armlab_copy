"""Synthetic RGB/depth regression tests; no RealSense or robot is started.

Run: MPLCONFIGDIR=/tmp/armlab-mpl python -m unittest discover -s tests -v
"""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from camera import Camera, WORKSPACE_TAG_PIXELS, BOARD_EXTENSION_MM


class BlockDetectorTests(unittest.TestCase):
    def test_busy_camera_reports_startup_error_without_stopping_another_pipeline(self):
        real_sense = Mock()
        real_sense.context.return_value.query_devices.return_value = [object()]
        pipeline = real_sense.pipeline.return_value
        pipeline.start.side_effect = RuntimeError("Device or resource busy")
        with patch.dict(sys.modules, {"pyrealsense2": real_sense}):
            camera = Camera()
        self.assertFalse(camera.camera_connected)
        self.assertEqual(camera.connection_error, "Device or resource busy")
        pipeline.stop.assert_not_called()

    def test_partial_camera_setup_failure_releases_its_own_pipeline(self):
        real_sense = Mock()
        real_sense.context.return_value.query_devices.return_value = [object()]
        pipeline = real_sense.pipeline.return_value
        sensor = pipeline.start.return_value.get_device.return_value.first_depth_sensor.return_value
        sensor.get_depth_scale.side_effect = RuntimeError("Depth sensor setup failed")
        with patch.dict(sys.modules, {"pyrealsense2": real_sense}):
            camera = Camera()
        self.assertFalse(camera.camera_connected)
        self.assertFalse(camera._pipeline_running)
        pipeline.stop.assert_called_once()

    def make_scene(self, blocks=(), bias=False):
        camera = Camera(start_pipeline=False)
        camera.depth_scale_mm = 0.25
        camera.intrinsic_matrix = np.array([[850., 0, 665], [0, 850., 332], [0, 0, 1]])
        tilt = np.deg2rad(19)
        rotation = np.array([[0, -1., 0], [-np.cos(tilt), 0, -np.sin(tilt)],
                             [np.sin(tilt), 0, -np.cos(tilt)]])
        world_to_camera = np.eye(4)
        world_to_camera[:3, :3] = rotation
        world_to_camera[:3, 3] = -rotation @ np.array([250.-900*np.tan(tilt), 0, 900.])
        camera.extrinsic_matrix_inv = world_to_camera
        camera.extrinsic_matrix = np.linalg.inv(world_to_camera)
        points = np.array([camera.tag_world_points[i] for i in (1, 2, 3, 4)])
        camera_points = points @ rotation.T + world_to_camera[:3, 3]
        projected = camera_points @ camera.intrinsic_matrix.T
        projected = (projected[:, :2] / projected[:, 2:3]).astype(np.float32)
        camera.workspace_homography, camera.workspace_homography_inv = (
            camera._calculate_workspace_transform(projected))
        camera.camera_calibrated = True
        shape = camera.depth_frame_raw.shape
        v, u = np.indices(shape, dtype=np.float32)
        raw_pixels = np.stack((u, v), axis=-1)
        workspace = cv2.perspectiveTransform(raw_pixels, camera.workspace_homography)
        ray_z = camera.extrinsic_matrix[2, :3] @ np.linalg.inv(camera.intrinsic_matrix)
        slope = ray_z[0]*u + ray_z[1]*v + ray_z[2]
        table = -camera.extrinsic_matrix[2, 3] / slope
        depth = table.copy()
        if bias:
            # Smooth camera elevation bias, 1..13 mm across the board, plus side tilt.
            depth += 1 + 12*(workspace[..., 1]-190)/285 + 2*(workspace[..., 0]-380)/570
        rgb_workspace = np.full((*shape, 3), 100, np.uint8)
        for center, side_mm, angle, hue, levels in blocks:
            scale = 0.95
            mask = np.zeros(shape, np.uint8)
            box = cv2.boxPoints((center, (side_mm*scale, side_mm*scale), angle))
            cv2.fillConvexPoly(mask, np.rint(box).astype(np.int32), 255)
            raw_mask = cv2.warpPerspective(mask, camera.workspace_homography_inv,
                                          (shape[1], shape[0]), flags=cv2.INTER_NEAREST) > 0
            depth[raw_mask] -= (side_mm*levels) / -slope[raw_mask]
            rgb = cv2.cvtColor(np.uint8([[[hue, 220, 210]]]), cv2.COLOR_HSV2RGB)[0, 0]
            rgb_workspace[mask > 0] = rgb
        camera.depth_frame_raw = np.rint(depth/camera.depth_scale_mm).astype(np.uint16)
        camera.video_frame = cv2.warpPerspective(rgb_workspace, camera.workspace_homography_inv,
                                                 (shape[1], shape[0]))
        return camera

    def detect(self, camera):
        camera.update_workspace_frame()
        camera.warp_depth_image()
        camera.update_block_detections()
        return camera.block_detections

    def test_six_colors_both_sizes_orientation_and_original_world_depth(self):
        colors = ("red", "orange", "yellow", "green", "blue", "purple")
        hues = (174, 16, 29, 60, 110, 145)
        blocks = [((x, y), size, angle, hue, 1)
                  for x, y, size, angle, hue in zip(
                      (490, 660, 830, 490, 660, 830), (260, 260, 260, 405, 405, 405),
                      (25, 38, 25, 38, 25, 38), (0, 17, 30, 45, 60, 75), hues)]
        camera = self.make_scene(blocks)
        records = self.detect(camera)
        self.assertEqual(len(records), 6, camera.block_rejections or camera.detector_status)
        by_color = {record["color"]: record for record in records}
        self.assertEqual(set(by_color), set(colors))
        for color, (_, size, angle, _, _) in zip(colors, blocks):
            record = by_color[color]
            self.assertEqual(record["size_mm"], size)
            self.assertEqual(record["size"], "small" if size == 25 else "large")
            angle_error = (record["orientation_deg"] - angle + 45) % 90 - 45
            self.assertLess(abs(angle_error), 4)
            self.assertAlmostEqual(record["height_mm"], size, delta=1)
            x, y = record["image_center_px"]
            original_mm = camera.depth_frame_raw[y, x] * camera.depth_scale_mm
            point = np.linalg.inv(camera.intrinsic_matrix) @ np.array([x, y, 1]) * original_mm
            expected = camera.extrinsic_matrix @ np.r_[point, 1]
            np.testing.assert_allclose(record["world_position_mm"], expected[:3])
        self.assertEqual(set(camera.detector_debug_frames),
                         {"warped depth", "flattened depth", "height", "mask", "contours", "rectangles"})

    def test_empty_tilted_table_tags_arm_and_invalid_depth_are_rejected(self):
        camera = self.make_scene()
        # Large gray arm-like strip raised 38 mm, well outside the size/shape filters.
        mask = np.zeros(camera.depth_frame_raw.shape, np.uint8)
        cv2.rectangle(mask, (530, 310), (780, 342), 255, -1)
        raw = cv2.warpPerspective(mask, camera.workspace_homography_inv,
                                  (mask.shape[1], mask.shape[0]), flags=cv2.INTER_NEAREST) > 0
        camera.depth_frame_raw[raw] -= 160
        # A raised colorful tag lies inside the expanded board, in a local exclusion.
        tag = np.zeros_like(mask)
        cv2.circle(tag, tuple(WORKSPACE_TAG_PIXELS[0].astype(int)), 12, 255, -1)
        raw_tag = cv2.warpPerspective(tag, camera.workspace_homography_inv,
                                     (mask.shape[1], mask.shape[0]), flags=cv2.INTER_NEAREST) > 0
        camera.depth_frame_raw[raw_tag] -= 120
        camera.video_frame[raw_tag] = (200, 20, 20)
        camera.depth_frame_raw[300:305, 580:600] = 0
        self.assertEqual(self.detect(camera), [])
        valid_table = np.isfinite(camera.block_height_mm) & (camera.block_mask == 0)
        self.assertLess(np.median(np.abs(camera.block_height_mm[valid_table])), 0.5)
        invalid = camera.depth_frame_topdown_raw == 0
        self.assertFalse(np.any(camera.block_mask[invalid]))

    def test_nearest_warp_preserves_raw_values_and_zeros(self):
        camera = Camera(start_pipeline=False)
        camera.depth_scale_mm = 0.5
        camera.depth_frame_raw[300:310, 500:510] = 1001
        camera.workspace_homography = np.array([[1., 0, 0.4], [0, 1., 0.6], [0, 0, 1.]])
        camera.warp_depth_image()
        self.assertEqual(set(np.unique(camera.depth_frame_topdown_raw)), {0, 500.5})

    def test_board_extension_detects_blocks_past_tags_on_all_four_sides(self):
        outside_x = int(380-0.95*BOARD_EXTENSION_MM-35)
        blocks = [((330, 330), 25, 0, 60, 1), ((1000, 330), 25, 0, 110, 1),
                  ((660, 140), 25, 0, 16, 1), ((660, 525), 25, 0, 145, 1),
                  ((outside_x, 330), 25, 0, 174, 1)]  # beyond the configured physical board
        camera = self.make_scene(blocks)
        records = self.detect(camera)
        self.assertEqual({r["color"] for r in records}, {"green", "blue", "orange", "purple"},
                         camera.block_rejections or camera.detector_status)
        # Full-view flattening preserves the depth of the out-of-board object,
        # while the detection mask excludes it.
        self.assertTrue(np.isfinite(camera.depth_frame_flattened[330, outside_x]))
        self.assertGreater(camera.block_height_mm[330, outside_x], 15)
        self.assertEqual(camera.block_mask[330, outside_x], 0)
        self.assertEqual(camera.depth_frame_flattened.shape, camera.depth_frame_topdown_raw.shape)

    def test_cached_table_plane_survives_covered_corner_and_hidden_tags(self):
        camera = self.make_scene([((650, 330), 38, 15, 60, 1)])
        self.assertEqual(len(self.detect(camera)), 1)
        plane = camera.table_plane_coefficients.copy()
        sample = camera._get_detector_geometry()["samples"][0]
        cover = np.zeros(camera.depth_frame_raw.shape, np.uint8)
        x, y = np.rint(sample).astype(int)
        cv2.rectangle(cover, (x-10, y-10), (x+10, y+10), 255, -1)
        raw_cover = cv2.warpPerspective(cover, camera.workspace_homography_inv,
                                       (cover.shape[1], cover.shape[0]), flags=cv2.INTER_NEAREST) > 0
        camera.depth_frame_raw[raw_cover] -= 150
        camera.tag_detections = []
        self.assertEqual(len(self.detect(camera)), 1)
        self.assertIn("cached plane", camera.detector_status)
        np.testing.assert_array_equal(camera.table_plane_coefficients, plane)
        self.assertAlmostEqual(camera.block_detections[0]["height_mm"], 38, delta=1)
        # The exact place-height path still gets zero table height while tags/corners are hidden.
        from state_machine import StateMachine
        sm = StateMachine(None, camera)
        x, y = np.rint(camera.workspace_pixel_to_image(700, 380)).astype(int)
        self.assertAlmostEqual(sm._click_surface_height(x, y, -13, None), 0, delta=1)

    def test_recalibration_invalidates_cached_plane_and_old_height_map(self):
        camera = self.make_scene()
        self.detect(camera)
        self.assertIsNotNone(camera.table_plane_coefficients)
        camera.camera_connected = True  # calibration consumes frames only; no hardware is started
        camera.tag_detections = [SimpleNamespace(tag_id=i, center=camera.workspace_pixel_to_image(*pixel))
                                 for i, pixel in zip((1, 2, 3, 4), WORKSPACE_TAG_PIXELS)]
        ok, message = camera.estimate_extrinsics_from_tags()
        self.assertTrue(ok, message)
        self.assertIsNone(camera.table_plane_coefficients)
        self.assertFalse(np.any(np.isfinite(camera.block_height_mm)))
        self.detect(camera)
        self.assertIsNotNone(camera.table_plane_coefficients)

    def test_initial_table_fit_uses_remaining_table_when_all_corners_are_covered(self):
        camera = self.make_scene([((650, 330), 38, 15, 60, 1)])
        cover = np.zeros(camera.depth_frame_raw.shape, np.uint8)
        for sample in camera._get_detector_geometry()["samples"]:
            x, y = np.rint(sample).astype(int)
            cv2.rectangle(cover, (x-10, y-10), (x+10, y+10), 255, -1)
        raw_cover = cv2.warpPerspective(cover, camera.workspace_homography_inv,
                                       (cover.shape[1], cover.shape[0]), flags=cv2.INTER_NEAREST) > 0
        camera.depth_frame_raw[raw_cover] -= 150
        self.assertIsNone(camera.table_plane_coefficients)
        records = self.detect(camera)
        self.assertEqual(len(records), 1, camera.detector_status)
        self.assertIn("visible table", camera.detector_status)
        self.assertAlmostEqual(records[0]["height_mm"], 38, delta=1)

    def test_no_initial_table_depth_does_not_invent_a_plane(self):
        camera = self.make_scene()
        camera.depth_frame_raw[:] = 0
        self.assertEqual(self.detect(camera), [])
        self.assertIsNone(camera.table_plane_coefficients)
        self.assertFalse(np.any(np.isfinite(camera.block_height_mm)))

    def test_touching_blocks_are_flagged(self):
        camera = self.make_scene([((640, 330), 25, 0, 60, 1),
                                  ((663, 330), 25, 0, 60, 1)])
        self.assertEqual(self.detect(camera), [])
        self.assertTrue(any("touching" in rejection["reason"] for rejection in camera.block_rejections))

    def test_partial_large_block_passes_relaxed_shape_and_side_tolerance(self):
        camera = self.make_scene([((650, 330), 38, 0, 60, 1)])
        cut = np.zeros(camera.depth_frame_raw.shape, np.uint8)
        cv2.rectangle(cut, (658, 309), (673, 351), 255, -1)
        raw_cut = cv2.warpPerspective(cut, camera.workspace_homography_inv,
                                     (cut.shape[1], cut.shape[0]), flags=cv2.INTER_NEAREST) > 0
        camera.depth_frame_raw[raw_cut] = 0
        records = self.detect(camera)
        self.assertEqual(len(records), 1, camera.block_rejections)
        record = records[0]
        ratio = min(record["dimensions_mm"])/max(record["dimensions_mm"])
        self.assertGreaterEqual(ratio, 0.60)
        self.assertLess(ratio, 0.72)  # previously rejected as non-square
        self.assertEqual(record["size"], "large")
        self.assertGreater(max(abs(d-38) for d in record["dimensions_mm"]), 7)

    def test_missing_corner_passes_relaxed_rectangular_fill(self):
        camera = self.make_scene([((650, 330), 38, 0, 60, 1)])
        cut = np.zeros(camera.depth_frame_raw.shape, np.uint8)
        cv2.rectangle(cut, (646, 309), (673, 328), 255, -1)
        raw_cut = cv2.warpPerspective(cut, camera.workspace_homography_inv,
                                     (cut.shape[1], cut.shape[0]), flags=cv2.INTER_NEAREST) > 0
        camera.depth_frame_raw[raw_cut] = 0
        records = self.detect(camera)
        self.assertEqual(len(records), 1, camera.block_rejections)
        contour = cv2.findContours(camera.block_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0][0]
        _, sides, _ = cv2.minAreaRect(contour)
        fill = cv2.contourArea(contour)/(sides[0]*sides[1])
        self.assertGreaterEqual(fill, 0.60)
        self.assertLess(fill, 0.72)

    def test_zero_center_depth_rejects_position(self):
        camera = self.make_scene([((650, 330), 38, 0, 60, 1)])
        record = self.detect(camera)[0]
        x, y = record["image_center_px"]
        camera.depth_frame_raw[y-1:y+2, x-1:x+2] = 0
        self.assertEqual(self.detect(camera), [])
        self.assertTrue(any("center depth" in r["reason"] for r in camera.block_rejections))

    def test_cached_bias_rejects_blocks_and_recovers_stack_levels(self):
        camera = self.make_scene([((520, 290), 38, 20, 16, 2),
                                  ((780, 380), 38, 40, 110, 3)], bias=True)
        camera.depth_bias_enabled = True
        camera.detector_max_stack_levels = 3
        records = self.detect(camera)
        self.assertEqual(len(records), 2, camera.block_rejections or camera.detector_status)
        self.assertEqual({r["stack_levels"] for r in records}, {2, 3})
        for record in records:
            self.assertAlmostEqual(record["height_mm"], 38*record["stack_levels"], delta=1)
        fitted = camera.depth_bias_coefficients.copy()
        self.detect(camera)
        np.testing.assert_array_equal(camera.depth_bias_coefficients, fitted)

    def test_red_wrap_and_highlights_use_interior_median(self):
        for hue in (3, 176):
            camera = self.make_scene([((650, 330), 38, 0, hue, 1)])
            camera.update_workspace_frame()
            camera.warp_depth_image()
            ok, message = camera.flatten_depth_topdown()
            self.assertTrue(ok, message)
            contour = camera.threshold_block_heights()[0]
            x, y, width, height = cv2.boundingRect(contour)
            camera.workspace_frame[y:y+height, x:x+width//3] = 255
            self.assertEqual(camera.classify_block_color(contour), "red")

    def test_uncalibrated_frames_have_no_detections(self):
        camera = Camera(start_pipeline=False)
        self.assertEqual(self.detect(camera), [])
        self.assertIn("Calibrate", camera.detector_status)


if __name__ == "__main__":
    unittest.main()
