"""
Camera wrapper for the Intel RealSense L515.

Runtime behavior:
- use RealSense intrinsics directly
- provide live RGB/depth for the control station
- expose a student-facing extrinsic calibration entrypoint that uses
  AprilTags + solvePnP at runtime
"""
import time
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs
from pupil_apriltags import Detector
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtGui import QImage
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

COLOR_W, COLOR_H = 1280, 720
DEPTH_W, DEPTH_H = 1024, 768
TAG_SIZE_MM = 25.0
WORKSPACE_MARGIN_MM = 50.0
# Known workspace tag centers in millimeters in the robot/world frame.
# Students can adjust these for their physical station.
# TODO: student lab - adjust thest to match new workspace tag locations.
TAG_WORLD_POINTS = {
    1: (100.0,  300.0, 0.0),
    2: (100.0, -300.0, 0.0),
    3: (400.0, -300.0, 0.0),
    4: (400.0, 300.0, 0.0),
}
# TODO: fill in from the fixture spec for your station.
FIXTURE_OFFSET_MM = np.array([0, 0, 60], dtype=float)
CHECK_3D_TARGETS_MM = np.array([
    [ -50, 350, 100],
    [ -50, 350, 150],
    [ -50, 350, 200],
    [ 300,   0, 100],
    [ 300,   0, 150],
    [ 300,   0, 200],
    [   0,-350, 100],
    [   0,-350, 150],
    [   0,-350, 200],
    [ 350,-150, 100],
    [ 350,-150, 150],
    [ 350,-150, 200],
], dtype=float)
CHECK_3D_CSV = (Path(__file__).resolve().parent.parent / "results" / "check_3d_results.csv")
FIXTURE_TAG_ID = 5

class Camera:
    """Holds live frame data and owns the RealSense pipeline."""

    def __init__(self):
        self.video_frame = np.zeros((COLOR_H, COLOR_W, 3), dtype=np.uint8)
        self.depth_frame_raw = np.zeros((COLOR_H, COLOR_W), dtype=np.uint16)
        self.depth_frame_rgb = np.zeros((COLOR_H, COLOR_W, 3), dtype=np.uint8)
        self.tag_image_frame = np.zeros((COLOR_H, COLOR_W, 3), dtype=np.uint8)
        self.workspace_frame = np.zeros((COLOR_H, COLOR_W, 3), dtype=np.uint8)
        self.heatmap = np.zeros((COLOR_H, COLOR_W, 3), dtype=np.uint8)

        # Camera Calibration: extrinsic_matrix maps world-frame homogeneous points
        # to camera-frame homogeneous points; extrinsic_matrix_inv does the reverse.
        self.camera_calibrated = False
        self.intrinsic_matrix = np.eye(3, dtype=np.float64)
        self.extrinsic_matrix = np.eye(4, dtype=np.float64) # world to camera
        self.extrinsic_matrix_inv = np.eye(4, dtype=np.float64) # camera to world (cached inverse)
        # Workspace view: raw image pixel -> top-down workspace pixel.
        self.workspace_homography = None
        # Inverse lets mouse clicks in workspace view map back to raw image pixels.
        self.workspace_homography_inv = None

        # Click inspection state used by the GUI.
        self.last_click = np.array([0, 0], dtype=int)
        self.new_click = False

        # AprilTag state
        self.tag_world_points = dict(TAG_WORLD_POINTS)
        self.tag_detections = []
        self.board_tag_centers_px = None
        self.tag_detect_interval = 6  # ~5 Hz at 30 FPS
        self.depth_scale_mm = 1.0   # raw depth units -> mm; overwritten once the pipeline starts
        self._pipeline_running = False
        self.camera_connected = False
        self.check_3d_records = []
        self.board_check_requested = False

        if len(rs.context().query_devices()) == 0:
            print("WARNING: No RealSense device found - running without video.")
            return

        try:
            self.pipeline = rs.pipeline()
            cfg = rs.config()
            cfg.enable_stream(rs.stream.color, COLOR_W, COLOR_H, rs.format.bgr8, 30)
            cfg.enable_stream(rs.stream.depth, DEPTH_W, DEPTH_H, rs.format.z16, 30)
            profile = self.pipeline.start(cfg)
            self.align = rs.align(rs.stream.color)
            depth_sensor = profile.get_device().first_depth_sensor()
            self.depth_scale_mm = depth_sensor.get_depth_scale() * 1000.0
            # L515 visual_preset: 0=Custom, 1=Default (deprecated), 2=No Ambient,
            # 3=Low Ambient, 4=Max Range, 5=Short Range.
            # Short Range lowers the firmware depth floor toward the 250 mm spec minimum.
            # Otherwise the depth cut off from 411 mm.
            if depth_sensor.supports(rs.option.visual_preset):
                try:
                    depth_sensor.set_option(rs.option.visual_preset, 5)
                except RuntimeError:
                    pass
            if depth_sensor.supports(rs.option.min_distance):
                depth_sensor.set_option(rs.option.min_distance, 0)

            intr = (
                profile.get_stream(rs.stream.color)
                .as_video_stream_profile()
                .intrinsics
            )
            self.intrinsic_matrix = np.array([
                [intr.fx, 0.0, intr.ppx],
                [0.0, intr.fy, intr.ppy],
                [0.0, 0.0, 1.0],
            ], dtype=np.float64)

            self.tag_detector = Detector(families="tagStandard41h12")

            self.spatial_filter = rs.spatial_filter()
            self.spatial_filter.set_option(rs.option.filter_magnitude, 2)
            self.spatial_filter.set_option(rs.option.filter_smooth_alpha, 0.5)
            self.spatial_filter.set_option(rs.option.filter_smooth_delta, 20)

            self.temporal_filter = rs.temporal_filter()
            self.temporal_filter.set_option(rs.option.filter_smooth_alpha, 0.1)
            self.temporal_filter.set_option(rs.option.filter_smooth_delta, 90)
            self.temporal_filter.set_option(rs.option.holes_fill, 3)

            self.hole_filling = rs.hole_filling_filter()
            self.hole_filling.set_option(rs.option.holes_fill, 2)

            self._pipeline_running = True
            self.camera_connected = True

            # self.estimate_extrinsics_from_tags()
        except RuntimeError as exc:
            print(
                f"WARNING: Camera unavailable ({exc})\n"
                "  -> Connect the L515 via USB 3.2 for video."
            )

    def stop(self):
        if not self._pipeline_running:
            return
        self.pipeline.stop()
        self._pipeline_running = False

    def colorize_depth_frame(self, depth_lo_mm=300.0, depth_hi_mm=1200.0):
        # Jet colormap (blue=near, red=far); pixels outside range masked black.
        depth_mm = self.depth_frame_raw.astype(np.float32) * self.depth_scale_mm
        in_range = (depth_mm >= depth_lo_mm) & (depth_mm <= depth_hi_mm)

        normalized = np.zeros(depth_mm.shape, dtype=np.uint8)
        normalized[in_range] = np.clip(
            (depth_mm[in_range] - depth_lo_mm) / (depth_hi_mm - depth_lo_mm) * 255.0,
            0, 255,
        ).astype(np.uint8)

        colorized = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
        colorized = cv2.cvtColor(colorized, cv2.COLOR_BGR2RGB)
        colorized[~in_range] = 0

        self.depth_frame_rgb = colorized

    # 2.8a
    def draw_gridlines_in_rgb_image(self):
        if not self.camera_calibrated:
            return
        # (-450,450) in y
        # (450 in x)
        intersections = np.array([])
        projection_matrix = np.array([
            [1,0,0,0],
            [0,1,0,0],
            [0,0,1,0],
        ])
        # self.estimate_extrinsics_from_tags()

        for x in range(0, 450, 50):
            for y in range(-450, 450, 50):
                world_point = np.array([x, y, 0, 1])
                camera_point = self.extrinsic_matrix_inv @ world_point
                pixel = 1 / camera_point[2] * self.intrinsic_matrix @ projection_matrix @ camera_point
                #breakpoint()
                cv2.drawMarker(self.tag_image_frame, [int(i) for i in pixel[:2]], [0,255,0], markerSize=30, thickness=2)

    # 2.8b
    # def check_board_plane(self):
    #     # pass
    #     if not self.camera_calibrated:
    #         return False, "Calibrate the camera first."
    #     rows, columns = self.video_frame.shape[:2]
    #     local_heatmap = np.zeros((rows, columns))
    #     try:                        
    #         for row in range(rows):
    #             for col in range(columns):
    #                 # breakpoint()
    #                 world_point = self.image_to_world(col, row)
    #                 local_heatmap[row][col] = int(world_point[2])
    #     except IndexError as e:
    #         error = e
    #         breakpoint()

    #     plt.imshow(local_heatmap, cmap="viridis", interpolation="nearest")
    #     plt.colorbar()
    #     plt.title('Z Error')
    #     plt.savefig(fname="z_error_heatmap")
    

    def check_board_plane(self, step=1):
        if not self.camera_calibrated:
            return False, "Calibrate the camera first."

        rows, columns = self.depth_frame_raw.shape
        sample_rows = range(0, rows, step)
        sample_cols = range(0, columns, step)

        # NaN means no valid depth measurement, not zero error.
        z_map = np.full( (len(sample_rows), len(sample_cols)), np.nan, dtype=float)
        
        records = []
        for map_row, image_row in enumerate(sample_rows):
            for map_col, image_col in enumerate(sample_cols):
                if self.depth_frame_raw[image_row, image_col] == 0:
                    continue

                world_point = self.image_to_world(image_col, image_row)
                if world_point is None:
                    continue

                z_mm = float(world_point[2])
                z_map[map_row, map_col] = z_mm
                records.append((image_col, image_row, z_mm))

        output_dir = Path(__file__).resolve().parent.parent / "results"
        # np.savetxt(
        #     output_dir / "z_error_samples.csv",
        #     np.asarray(records),
        #     delimiter=",",
        #     fmt=["%d", "%d", "%.3f"],
        #     header="u_px,v_px,world_z_mm",
        #     comments="",
        # )

        fig, ax = plt.subplots(figsize=(12, 7))
        image = ax.imshow(
            np.ma.masked_invalid(z_map),
            cmap="viridis",
            vmin=-100,   # mm: same lower limit for every capture
            vmax=300,    # mm: same upper limit for every capture
            interpolation="nearest",
            extent=(0, columns, rows, 0),
        )
        fig.colorbar(image, ax=ax, label="World z (mm)")
        ax.set_title("z error heatmap")
        ax.set_xlabel("Image u (pixels)")
        ax.set_ylabel("Image v (pixels)")
        fig.savefig(output_dir/ "z_error_heatmap.png")
        plt.close(fig)

        return True, "z error heatmap successfully saved"
        
        # 2.8c
    def check_3d_point(self, ee_pose):
        if not self.camera_calibrated:
            return False, "Calibrate the camera first."
        if FIXTURE_OFFSET_MM is None:
            return False, "Fixture offset is unknown; cannot record 3D error."
        
        index = len(self.check_3d_records)
        if index >= len(CHECK_3D_TARGETS_MM):
            return False, "All 12 target measurements are recorded."

        if ee_pose is None:
            return False, "No arm pose available."

        ee_pose = np.asarray(ee_pose, dtype=float)
        if ee_pose.shape != (6,) or not np.all(np.isfinite(ee_pose)):
            return False, "Invalid arm pose."

        # get_ee_pose_sdk currently returns six zeros when SDK FK fails.
        if np.allclose(ee_pose, 0.0):
            return False, "Arm pose unavailable; SDK FK returned zeros."

        fixture = self.find_fixture_detection()
        if fixture is None:
            return False, "Fixture not uniquely detected; keep board tag 1 and the fixture visible."
        
        u, v = np.round(fixture.center).astype(int)
        height, width = self.depth_frame_raw.shape
        if not (0 <= u < width and 0 <= v < height):
            return False, "Fixture tag center is outside the depth image."

        raw_depth = self.depth_frame_raw[v, u]
        if raw_depth == 0:
            return False, "No depth at the fixture tag center."

        # Target describes the intended tag-center position.
        target = CHECK_3D_TARGETS_MM[index]

        # Truth comes from the arm's ACTUAL pose, not the intended target.
        ee_rotation = R.from_euler("xyz", ee_pose[3:]).as_matrix()
        truth = ee_pose[:3] + ee_rotation @ FIXTURE_OFFSET_MM

        # Camera estimate comes from the detected tag center and aligned depth.
        world_point = self.image_to_world(u, v)
        if world_point is None:
            return False, "Could not convert fixture pixel to world coordinates."
        estimate = np.asarray(world_point[:3], dtype=float)

        error = estimate - truth
        distance_error = float(np.linalg.norm(error))
        depth_mm = float(raw_depth * self.depth_scale_mm)

        record = np.concatenate((
            [index + 1],
            target,
            ee_pose,
            truth,
            estimate,
            error,
            [distance_error, u, v, depth_mm],
        ))
        self.check_3d_records.append(record)

        CHECK_3D_CSV.parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(
            CHECK_3D_CSV,
            np.asarray(self.check_3d_records),
            delimiter=",",
            fmt="%.3f",
            header=(
                "sample,target_x,target_y,target_z,"
                "ee_x,ee_y,ee_z,ee_roll,ee_pitch,ee_yaw,"
                "truth_x,truth_y,truth_z,"
                "est_x,est_y,est_z,"
                "err_x,err_y,err_z,err_dist,"
                "u_px,v_px,depth_mm"
            ),
            comments="",
        )

        message = (
            f"Recorded {index + 1}/12 for target {target.astype(int)} mm: "
            f"error=({error[0]:+.1f}, {error[1]:+.1f}, "
            f"{error[2]:+.1f}) mm; distance={distance_error:.1f} mm"
        )
        print(message)
        return True, message

    def draw_tags_in_rgb_image(self):
        # TODO: student lab
        self.tag_image_frame = self.video_frame.copy()

        # uncomment this for task 2.5
        # self.estimate_extrinsics_from_tags()

        if not self.camera_calibrated:
            return
        fixture = self.find_fixture_detection()
        for detection_obj in self.tag_detections:

            # breakpoint()
            cv2.drawMarker(self.tag_image_frame, [int(i) for i in detection_obj.center], [0, 255, 0], markerSize=30, thickness=2)
            # cv2.line(self.tag_image_frame, test[0], test[1], color=[0, 255, 0], thickness=2)
            # breakpoint()
            cv2.polylines(self.tag_image_frame, [detection_obj.corners.astype(int)], isClosed=True, color=[0, 255, 0], thickness=2)
            label = "5*" if detection_obj is fixture else str(detection_obj.tag_id)
            # cv2.putText(self.tag_image_frame, str(detection_obj.tag_id), detection_obj.corners[0].astype(int), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1.0 ,color=[255, 0, 0], thickness=2)
            cv2.putText(
                self.tag_image_frame, label,
                detection_obj.corners[0].astype(int),
                cv2.FONT_HERSHEY_PLAIN, 1.0, [255, 0, 0], 2
            )
        # breakpoint()

    def estimate_extrinsics_from_tags(self):
        """
        Uses AprilTag centers and known world-frame tag locations to estimate a
        camera-to-world transform. The result is stored in memory only and also
        printed to the terminal every time the user clicks Calibrate.
        """
        if not self.camera_connected:
            return False, "Camera offline - cannot calibrate."

        # TODO: student lab

        tag_ids = (1, 2, 3, 4)
        detections = tuple(self.tag_detections)
        board = {
            tag_id: [d for d in detections if int(d.tag_id) == tag_id]
            for tag_id in tag_ids
        }
        if any(len(board[tag_id]) != 1 for tag_id in tag_ids):
            return False, "Calibrate with exactly one of each board tag 1–4; move the fixture out of view."

        centers = {
            tag_id: np.asarray(board[tag_id][0].center, dtype=float).copy()
            for tag_id in tag_ids
        }

        if any(tag_id not in centers for tag_id in tag_ids):
            return False, "Board tags 1–4 must be visible."

        world_points = np.array([self.tag_world_points[tag_id] for tag_id in tag_ids], dtype=np.float64)

        image_points = np.array([centers[tag_id] for tag_id in tag_ids], dtype=np.float64)
        workspace_transform = self._calculate_workspace_transform(image_points)
        if workspace_transform is None:
            return False, "Board tags 1–4 do not form a valid workspace rectangle."
        # breakpoint()

        ok, rot_vec, trans_vec = cv2.solvePnP(world_points, image_points, self.intrinsic_matrix, distCoeffs=None)
        if not ok:
            return False, "solvePnP could not estimate the camera pose."
        rot_matrix, _ = cv2.Rodrigues(rot_vec)

        # breakpoint()

        extr_matrix = np.eye(4)

        for i in range(len(rot_matrix)):
            extr_matrix[i] = np.append(rot_matrix[i], trans_vec[i])

        self.extrinsic_matrix = np.linalg.inv(extr_matrix)
        self.extrinsic_matrix_inv = extr_matrix
        self.workspace_homography, self.workspace_homography_inv = workspace_transform
        self.board_tag_centers_px = centers
        self.camera_calibrated = True


        # breakpoint()

        return True, "Camera successfully calibrated"




    def depth_to_camera_point(self, x, y, depth_raw):
        """Convert an image pixel + raw depth unit to a camera-frame 3D point (mm)."""
        fx = self.intrinsic_matrix[0][0]
        fy = self.intrinsic_matrix[1][1]
        cx = self.intrinsic_matrix[0][2]
        cy = self.intrinsic_matrix[1][2]
        Z = depth_raw[y][x] * self.depth_scale_mm
        X = (x - cx) * Z / fx
        Y = (y- cy) * Z / fy
        # breakpoint()
        #print(f"X: {X}, Y: {Y}, Z: {Z}")
        
        return np.array([X, Y, Z])

    def camera_to_world(self, camera_point):
        """Convert a camera-frame 3D point to a world-frame 3D point."""
        if not self.camera_calibrated:
            return None
        
        one = np.array([1])
        # breakpoint()
        # 
        # self.estimate_extrinsics_from_tags()
        # print(self.camera_calibrated)
        return self.extrinsic_matrix @ np.concatenate((camera_point, one))

    def image_to_world(self, x, y):
        """Pixel (x, y) -> world-frame (X, Y, Z) in mm using live depth."""
        return self.camera_to_world(self.depth_to_camera_point(x, y, self.depth_frame_raw))

    def workspace_pixel_to_image(self, x_px, y_px):
        """Map a workspace-view pixel back to the original image pixel."""
        if self.workspace_homography_inv is None:
            return None
        # self._update_workspace_transform()
        
        transformed_point = cv2.perspectiveTransform(np.array([[[x_px,y_px]]], dtype=np.float32), self.workspace_homography_inv)
        return transformed_point[0][0]

    # def _update_workspace_transform(self):
    #     """Build the homography used for the top-down workspace view."""
    #     self.workspace_homography = None
    #     self.workspace_homography_inv = None

    #     tag_ids = [1,2,3,4]
    #     source_readings = {}
    #     for detection_obj in self.tag_detections:
    #         if detection_obj.tag_id in tag_ids:
    #             source_readings[detection_obj.tag_id] = detection_obj.center

    #     if any(tag_id not in source_readings for tag_id in tag_ids):
    #         return False

    #     source_readings = np.array([source_readings[tag_id] for tag_id in tag_ids],  dtype=np.float64)

    #     destinations = np.array([[380, 475], 
    #                     [950, 475],
    #                     [950, 190],
    #                     [380, 190] ])

    #     #breakpoint()

    #     H, _ = cv2.findHomography(source_readings, destinations)

    #     self.workspace_homography = H
    #     self.workspace_homography_inv = np.linalg.inv(H)

    #     return True

    def _calculate_workspace_transform(self, image_points):
        """Calculate a transform from the four fixed board tags."""
        source = np.asarray(image_points, dtype=np.float32)

        if source.shape != (4, 2) or not np.all(np.isfinite(source)):
            return None

        contour = source.reshape(4, 1, 2)
        if not cv2.isContourConvex(contour):
            return None
        if cv2.contourArea(contour) < 1000.0:
            return None

        destinations = np.array([
            [380, 475],  # tag 1
            [950, 475],  # tag 2
            [950, 190],  # tag 3
            [380, 190],  # tag 4
        ], dtype=np.float32)

        H = cv2.getPerspectiveTransform(source, destinations)
        if not np.all(np.isfinite(H)):
            return None

        try:
            H_inv = np.linalg.inv(H)
        except np.linalg.LinAlgError:
            return None

        if not np.all(np.isfinite(H_inv)):
            return None

        return H, H_inv
        

    # def update_workspace_frame(self):
    #     """Warp the RGB frame into the top-down workspace view when tags are visible."""
    #     self.workspace_frame = self.video_frame.copy()
    #     if self._update_workspace_transform():
    #         height, width = self.workspace_frame.shape[:2]
    #         self.workspace_frame = cv2.warpPerspective(self.workspace_frame, self.workspace_homography, (width, height))

    def update_workspace_frame(self):
        """Display the workspace using the transform saved at calibration."""
        self.workspace_frame = self.video_frame.copy()

        if self.workspace_homography is not None:
            height, width = self.workspace_frame.shape[:2]
            self.workspace_frame = cv2.warpPerspective(
                self.workspace_frame,
                self.workspace_homography,
                (width, height),
            )

    def _to_qimage(self, frame):
        """Convert a NumPy RGB image into a Qt image for display."""
        try:
            h, w = frame.shape[:2]
            return QImage(frame.data, w, h, frame.strides[0], QImage.Format_RGB888).copy()
        except Exception:
            return None

    def find_fixture_detection(self, tolerance_px=30.0):
        if self.board_tag_centers_px is None:
            return None

        detections = tuple(self.tag_detections)
        board_detections = []

        for tag_id, board_center in self.board_tag_centers_px.items():
            matches = [
                d for d in detections
                if int(d.tag_id) == tag_id
                and np.linalg.norm(np.asarray(d.center) - board_center) < tolerance_px
            ]
            if len(matches) > 1:
                return None  # Ambiguous
            board_detections.extend(matches)

        # Since the fixture also decodes as ID 1, require the real board ID 1
        # to be visible so we can distinguish the two safely.
        if not any(int(d.tag_id) == 1 for d in board_detections):
            return None

        extras = [
            d for d in detections
            if not any(d is board_d for board_d in board_detections)
        ]
        return extras[0] if len(extras) == 1 else None

    def warp_depth_image(self):
        self.depth_frame_topdown_raw = self.depth_frame_raw.copy()
        
        if self.workspace_homography is not None:
            height, width = self.depth_frame_topdown_raw.shape[:2]
            self.depth_frame_topdown_raw = cv2.warpPerspective(
                self.depth_frame_topdown_raw,
                self.workspace_homography,
                (width, height),
                flags=cv2.INTER_NEAREST
            )

        self.flatten_depth_topdown()

    def colorize_depth_frame_topdown(self, depth_lo_mm=300.0, depth_hi_mm=1200.0):
            # Jet colormap (blue=near, red=far); pixels outside range masked black.
            depth_mm = self.depth_frame_topdown_raw.astype(np.float32) * self.depth_scale_mm
            in_range = (depth_mm >= depth_lo_mm) & (depth_mm <= depth_hi_mm)
    
            normalized = np.zeros(depth_mm.shape, dtype=np.uint8)
            normalized[in_range] = np.clip(
                (depth_mm[in_range] - depth_lo_mm) / (depth_hi_mm - depth_lo_mm) * 255.0,
                0, 255,
            ).astype(np.uint8)
    
            colorized = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
            colorized = cv2.cvtColor(colorized, cv2.COLOR_BGR2RGB)
            colorized[~in_range] = 0
    
            self.depth_frame_topdown_rgb = colorized
    

    def flatten_depth_topdown(self):

        TABLE_SAMPLE_PIXEL = np.array([
            [380, 475],  # tag 1
            [950, 475],  # tag 2
            [950, 190],  # tag 3
            [380, 190],  # tag 4
        ])

        sample_us = []
        sample_vs = []
        sample_depths = []
        for (u, v) in TABLE_SAMPLE_PIXEL:
            depth = self.depth_frame_topdown_raw[v, u]
            sample_us.append(u)
            sample_vs.append(v)
            sample_depths.append(depth)

        A = np.column_stack((sample_us, sample_vs, np.ones(4)))

        (a, b, c), *_ = np.linalg.lstsq(A, np.array(sample_depths))

        H, W = self.depth_frame_topdown_raw.shape
        U, V = np.meshgrid(np.arange(W), np.arange(H))
        plane = a * U + b * V + c

        # self.height = np.where(self.depth_frame_topdown_raw > 0, plane - self.depth_frame_topdown_raw, 0.0).astype(np.float32)
        self.table_depth_raw = np.mean(sample_depths)
        self.depth_frame_topdown_raw = np.where(self.depth_frame_topdown_raw > 0, self.depth_frame_topdown_raw - plane + self.table_depth_raw, 0.0)
        self.colorize_depth_frame_topdown()


    def find_contours(self, h_lo_mm, h_hi_mm):

        height_mm = np.where(self.depth_frame_raw > 0, (self.table_depth_raw - self.depth_frame_topdown_raw) * self.depth_scale_mm, 0.0)

        mask = ((height_mm > h_lo_mm) & (height_mm < h_hi_mm)).astype(np.uint8) * 255

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        self.block_contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        cv2.drawContours(self.workspace_frame, self.block_contours, -1, (0, 255, 0), 2)
        
        


class VideoThread(QThread):
    """Polls the RealSense pipeline at ~30 Hz and emits Qt signals for the GUI."""

    updateFrame = pyqtSignal(QImage, QImage, QImage, QImage)

    def __init__(self, camera, parent=None):
        QThread.__init__(self, parent=parent)
        self.camera = camera
        self._running = True
        self._frame_counter = 0

    def stop(self):
        self._running = False

    def run(self):
        if not self.camera.camera_connected:
            while self._running:
                time.sleep(0.5)
            return

        while self._running:
            start = time.time()

            try:
                frames = self.camera.pipeline.wait_for_frames(timeout_ms=100)
            except RuntimeError:
                if not self._running:
                    break
                continue

            aligned = self.camera.align.process(frames)
            color_frame = aligned.get_color_frame()
            depth_frame = aligned.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            depth_frame = self.camera.spatial_filter.process(depth_frame)
            depth_frame = self.camera.temporal_filter.process(depth_frame)
            depth_frame = self.camera.hole_filling.process(depth_frame)

            bgr = np.asanyarray(color_frame.get_data())
            depth = np.asanyarray(depth_frame.get_data())

            self.camera.video_frame = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            self.camera.depth_frame_raw = depth
            self.camera.colorize_depth_frame()

            if self._frame_counter % self.camera.tag_detect_interval == 0:
                gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
                self.camera.tag_detections = self.camera.tag_detector.detect(gray)
            self._frame_counter += 1

            
            self.camera.draw_tags_in_rgb_image()
            self.camera.update_workspace_frame()
            self.camera.draw_gridlines_in_rgb_image()
            # self.camera.check_board_plane()

            self.camera.warp_depth_image()
            self.camera.find_contours()
            

            if self.camera.board_check_requested:
                self.camera.board_check_requested = False
                _, message = self.camera.check_board_plane()
                print(message)

            rgb_qt   = self.camera._to_qimage(self.camera.video_frame)
            depth_qt = self.camera._to_qimage(self.camera.depth_frame_topdown_rgb)
            tag_qt   = self.camera._to_qimage(self.camera.tag_image_frame)
            workspace_qt = self.camera._to_qimage(self.camera.workspace_frame)
            heatmap_qt = self.camera._to_qimage(self.camera.heatmap)

            if rgb_qt and depth_qt and tag_qt and workspace_qt and heatmap_qt:
                self.updateFrame.emit(rgb_qt, depth_qt, tag_qt, workspace_qt)
                # self.updateFrame.emit(rgb_qt, depth_qt, heatmap_qt, workspace_qt)

            elapsed = time.time() - start
            time.sleep(max(0.0, 1 / 30 - elapsed))


if __name__ == '__main__':
    import sys

    from PyQt5.QtCore import Qt, pyqtSlot
    from PyQt5.QtGui import QPixmap
    from PyQt5.QtWidgets import QApplication, QLabel

    app = QApplication(sys.argv)
    camera = Camera()
    print(f"Intrinsic matrix:\n{camera.intrinsic_matrix}")

    # camera.estimate_extrinsics_from_tags()

    wins = {}
    for title in ("RGB", "Depth", "Tags", "Workspace"):
        label = QLabel(title)
        label.setWindowTitle(title)
        label.setAlignment(Qt.AlignCenter)
        label.resize(640, 360)
        label.show()
        wins[title] = label

    @pyqtSlot(QImage, QImage, QImage, QImage)
    def on_frame(rgb, depth, tags, workspace):
        for label, image in zip(wins.values(), (rgb, depth, tags, workspace)):
            label.setPixmap(
                QPixmap.fromImage(image).scaled(
                    label.size(),
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation,
                )
            )

    thread = VideoThread(camera)
    thread.updateFrame.connect(on_frame)
    thread.start()

    result = app.exec_()
    thread.stop()
    thread.wait(500)
    camera.stop()
    sys.exit(result)
