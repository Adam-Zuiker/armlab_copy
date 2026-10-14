"""Test measurement math/report outputs using explicitly synthetic observations."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from detector_verify import (COLORS, PREDICTED_LABELS, match_detections, record_measurement,
                             load_measurements, calculate_position_errors,
                             build_color_confusion_matrix, summarize_measurements, generate_report,
                             GROUND_TRUTHS, validate_ground_truths, record_ground_truth_scene)


class DetectorMeasurementTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.csv = Path(self.directory.name) / "measurements.csv"

    def test_matching_does_not_use_true_color_and_is_one_to_one(self):
        truths = [{"world_position_mm": (0, 0), "color": "red"},
                  {"world_position_mm": (8, 0), "color": "blue"}]
        predictions = [{"world_position_mm": (7, 0), "color": "red"},
                       {"world_position_mm": (12, 0), "color": "red"}]
        matches, misses, extras = match_detections(truths, predictions, 10)
        self.assertEqual(matches, [(0, 0), (1, 1)])
        self.assertEqual(misses, [])
        self.assertEqual(extras, [])
        far_correct_color = [{"world_position_mm": (200, 0), "color": "red"},
                             {"world_position_mm": (0, 0), "color": "blue"}]
        self.assertEqual(match_detections(truths[:1], far_correct_color, 10)[0], [(0, 1)])

    def test_csv_preserves_misses_color_confusion_and_signed_position_errors(self):
        truths = [{"world_position_mm": (250, 0, 38), "color": "red", "size_mm": 38},
                  {"world_position_mm": (450, 0, 25), "color": "green", "size_mm": 25}]
        predictions = [{"world_position_mm": (253, 4, 36), "color": "blue", "size_mm": 38}]
        record_measurement(self.csv, truths, predictions, settings={"test_only": True})
        rows = load_measurements(self.csv)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[1]["predicted_color"], "missed")
        errors = calculate_position_errors(rows)
        self.assertEqual(errors[0]["error_x_mm"], 3)
        self.assertEqual(errors[0]["error_y_mm"], 4)
        self.assertEqual(errors[0]["error_xy_mm"], 5)
        self.assertEqual(errors[0]["error_z_mm"], -2)
        self.assertAlmostEqual(errors[0]["error_xyz_mm"], 29**0.5)
        self.assertEqual(errors[1]["error_xy_mm"], "")
        self.assertEqual(errors[1]["radius_from_center_mm"], 200)
        matrix = build_color_confusion_matrix(rows)
        self.assertEqual(matrix[COLORS.index("red"), PREDICTED_LABELS.index("blue")], 1)
        self.assertEqual(matrix[COLORS.index("green"), PREDICTED_LABELS.index("missed")], 1)
        self.assertEqual(json.loads(rows[0]["settings_json"]), {"test_only": True})
        summary, _, _ = summarize_measurements(rows)
        self.assertEqual(summary["position_z"]["observations"], 1)
        self.assertEqual(summary["position_z"]["bias_mm"], -2)
        self.assertEqual(summary["position_z"]["rmse_mm"], 2)
        self.assertAlmostEqual(summary["position_xyz"]["rmse_mm"], 29**0.5)

    def test_summary_includes_misses_and_counts_extras_once_per_trial(self):
        truth = [{"world_position_mm": (250, 0), "color": "red"},
                 {"world_position_mm": (450, 0), "color": "green"}]
        predicted = [{"world_position_mm": (253, 4, 38), "color": "red"},
                     {"world_position_mm": (1000, 0, 25), "color": "purple"}]
        record_measurement(self.csv, truth, predicted)
        summary, bins, errors = summarize_measurements(load_measurements(self.csv))
        self.assertEqual(summary["observations"], 2)
        self.assertEqual(summary["extra_detections"], 1)
        self.assertEqual(summary["detection_rate"], 0.5)
        self.assertEqual(summary["color_accuracy_all_observations"], 0.5)
        self.assertEqual(summary["color_accuracy_matched_only"], 1)
        self.assertEqual(summary["position_xy"]["rmse_mm"], 5)
        self.assertEqual(bins[0]["matched"], 1)
        self.assertEqual(bins[1]["missed"], 1)
        self.assertIsNone(bins[1]["mean_xy_error_mm"])
        self.assertEqual(errors[0]["error_z_mm"], "")  # optional true top Z omitted

    def test_append_and_coverage_report(self):
        for x in (250, 350):
            record_measurement(self.csv, [{"world_position_mm": (x, 0), "color": "orange"}],
                               [{"world_position_mm": (x+1, 0, 25), "color": "orange"}])
        rows = load_measurements(self.csv)
        summary, _, _ = summarize_measurements(rows)
        self.assertEqual(summary["color_coverage"]["orange"], {"observations": 2, "distinct_positions": 2})
        self.assertEqual(summary["trials"], 2)

    def test_report_handles_all_missed_and_writes_real_artifact_formats(self):
        record_measurement(self.csv, [{"world_position_mm": (250, 0), "color": "yellow"}], [])
        output = Path(self.directory.name) / "report"
        summary = generate_report(self.csv, output)
        self.assertIsNone(summary["position_xy"]["mean_mm"])
        self.assertEqual(summary["color_accuracy_all_observations"], 0)
        expected = {"summary.json", "position_errors.csv", "position_by_radius.csv",
                    "color_confusion.csv", "position_error.png", "color_confusion.png"}
        self.assertEqual({path.name for path in output.iterdir()}, expected)
        self.assertEqual(json.loads((output / "summary.json").read_text())["missed"], 1)
        self.assertTrue((output / "color_confusion.png").read_bytes().startswith(b"\x89PNG"))

    def test_reject_empty_truth_invalid_gate_and_incompatible_csv(self):
        with self.assertRaises(ValueError):
            record_measurement(self.csv, [], [])
        with self.assertRaises(ValueError):
            match_detections([], [], -1)
        with self.assertRaises(ValueError):
            summarize_measurements([])
        self.csv.write_text("other,header\n")
        with self.assertRaises(ValueError):
            record_measurement(self.csv, [{"world_position_mm": (0, 0), "color": "red"}], [])

    def test_saved_thirty_block_scene_generates_xy_graphs_and_confusion_with_misses(self):
        truths = validate_ground_truths(GROUND_TRUTHS)
        self.assertEqual(len(truths), 30)
        predictions = [{"world_position_mm": (t["world_position_mm"][0]+3,
                                                t["world_position_mm"][1]+4, 38),
                        "color": t["color"]} for t in truths[:-1]]
        predictions[0]["color"] = "blue"  # deliberately wrong color, not used for matching
        output = Path(self.directory.name)/"scene_report"
        rows, summary = record_ground_truth_scene(self.csv, predictions, output_dir=output)
        self.assertEqual(len(rows), 30)
        self.assertEqual(summary["matched"], 29)
        self.assertEqual(summary["missed"], 1)
        self.assertEqual(summary["color_accuracy_all_observations"], 28/30)
        self.assertEqual(summary["position_xy"]["mean_mm"], 5)
        self.assertEqual(summary["position_z"]["observations"], 0)
        self.assertTrue(all(row["true_z_mm"] == "" for row in rows))
        matrix = build_color_confusion_matrix(rows)
        self.assertEqual(int(matrix.sum()), 30)
        self.assertEqual(matrix[COLORS.index("yellow"), COLORS.index("blue")], 1)
        self.assertTrue((output/"position_error.png").exists())
        self.assertTrue((output/"color_confusion.png").exists())

    def test_saved_scene_rejects_duplicate_positions_before_writing_data(self):
        duplicate = [{"world_position_mm": (300, -400), "color": "red"},
                     {"world_position_mm": (300, -400), "color": "orange"}]
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            record_ground_truth_scene(self.csv, [], ground_truth=duplicate)
        self.assertFalse(self.csv.exists())


if __name__ == "__main__":
    unittest.main()
