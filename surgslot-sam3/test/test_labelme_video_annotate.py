"""Contract tests for assisted annotation; no checkpoint/GPU needed."""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np
from PIL import Image

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/labelme_video_annotate.py"
spec = importlib.util.spec_from_file_location("labelme_video_annotate", SCRIPT)
app = importlib.util.module_from_spec(spec)
spec.loader.exec_module(app)


def polygon(label, x=3, y=3, flags=None):
    return {
        "label": label,
        "points": [[x, y], [x + 12, y], [x + 12, y + 12], [x, y + 12]],
        "shape_type": "polygon",
        "group_id": 7,
        "flags": flags or {},
        "description": "doctor",
    }


class AnnotationTests(unittest.TestCase):
    def test_middle_anchor_reaches_both_ends(self):
        docs = [None] * 7
        docs[3] = {"shapes": [polygon("late")]}
        tasks = app.make_tasks({"late": [3]}, docs, app.parse_args([]))
        self.assertEqual(
            tasks[0]["spans"],
            [{"reverse": True, "end": 0}, {"reverse": False, "end": 6}],
        )

    def test_intervals_use_adjacent_same_class_anchors(self):
        docs = [None] * 9
        for i in (1, 4, 7):
            docs[i] = {"shapes": [polygon("A")]}
        tasks = app.make_tasks({"A": [1, 4, 7]}, docs, app.parse_args([]))
        contributors = {}
        for task in tasks:
            for span in task["spans"]:
                step = -1 if span["reverse"] else 1
                for i in range(task["anchor"] + step, span["end"] + step, step):
                    contributors.setdefault(i, []).append(task["anchor"])
        self.assertEqual(
            contributors, {0: [1], 2: [1, 4], 3: [1, 4], 5: [4, 7], 6: [4, 7], 8: [7]}
        )

    def test_disconnected_regions_and_circle(self):
        a = app.shape_mask(polygon("A"), 64, 64)
        b = app.shape_mask(polygon("A", 32, 32), 64, 64)
        self.assertEqual(len(list(app.components(a | b))), 2)
        shapes, holes = app.mask_shapes(a | b, "A", epsilon=0)
        self.assertEqual(len(shapes), 2)
        self.assertEqual(holes, 0)
        np.testing.assert_array_equal(app.class_masks(shapes, 64, 64)["A"], a | b)
        circle = {"points": [[20, 20], [23, 24]], "shape_type": "circle"}
        self.assertTrue(app.shape_mask(circle, 64, 64)[20, 20])

    def test_empty_predictions_and_distance_fusion(self):
        empty = np.zeros((8, 8), np.uint8)
        full = np.full((8, 8), 255, np.uint8)
        mask, iou = app.fuse_predictions([(empty, 1), (full, 3)])
        self.assertFalse(mask.any())
        self.assertEqual(iou, 0)
        mask, iou = app.fuse_predictions([(empty, 2), (empty, 2)])
        self.assertFalse(mask.any())
        self.assertEqual(iou, 1)
        mask, _ = app.fuse_predictions([(full, 1), (empty, 3)])
        self.assertTrue(mask.all())

    def test_refinement_is_default_with_filters_and_preserve_override(self):
        args = app.parse_args(
            ["--refine", "--refine-label", "A", "--refine-frame", "3"]
        )
        self.assertTrue(app.selected_for_refinement(polygon("A"), 2, args))
        self.assertFalse(app.selected_for_refinement(polygon("A"), 1, args))
        self.assertFalse(app.selected_for_refinement(polygon("B"), 2, args))
        self.assertFalse(
            app.selected_for_refinement(polygon("B", flags={"rough": True}), 2, args)
        )
        self.assertTrue(
            app.selected_for_refinement(polygon("B"), 2, app.parse_args([]))
        )
        self.assertFalse(
            app.selected_for_refinement(
                polygon("B", flags={"preserve": True}), 2, app.parse_args([])
            )
        )
        self.assertFalse(
            app.selected_for_refinement(
                polygon("B"), 2, app.parse_args(["--no-refine"])
            )
        )

    def test_temporal_postprocessing_respects_selected_frame_and_label(self):
        args = app.parse_args(["--refine-label", "B", "--refine-frame", "3"])
        args.height = args.width = 64
        original = [polygon("A"), polygon("B", 30, 30)]
        seeds = app.class_masks(original, 64, 64)
        temporal = {k: ([(np.full((64, 64), 255, np.uint8), 1)], [0]) for k in seeds}
        for frame_index in (1, 2):
            doc = {"shapes": list(original), "surgslot_original_shapes": original}
            masks, evidence, _, _ = app.improve_frame_masks(
                doc, seeds, temporal, args, frame_index
            )
            self.assertEqual(doc["shapes"][0], original[0])
            np.testing.assert_array_equal(masks["A"], seeds["A"])
            if frame_index == 1:
                self.assertEqual(doc["shapes"], original)
                self.assertFalse(evidence)
            else:
                self.assertIn("B", evidence)

    def test_overlap_keeps_both_labels(self):
        masks = app.class_masks([polygon("A"), polygon("B", 8, 8)], 64, 64)
        self.assertTrue((masks["A"] & masks["B"]).any())
        conflicts, manual = app.overlap_pairs(masks, set())
        self.assertFalse(conflicts)
        self.assertEqual(len(manual), 1)
        conflicts, manual = app.overlap_pairs(masks, {"B"})
        self.assertEqual(len(conflicts), 1)
        self.assertFalse(manual)
        # Colour blending must not depend on class overwrite order.
        frame = np.zeros((64, 64, 3), np.uint8)
        colors = {"A": (255, 0, 0), "B": (0, 255, 0)}
        forward = app.render_overlay(frame, masks, colors, [], 0.5)
        backward = app.render_overlay(
            frame, dict(reversed(list(masks.items()))), colors, [], 0.5
        )
        # Contour draw order can affect edge pixels; interior must be identical.
        np.testing.assert_array_equal(forward[10, 10], backward[10, 10])

    def test_refinement_guard_rejects_displaced_empty_and_oversized_masks(self):
        args = app.parse_args([])
        seed = app.shape_mask(polygon("A", 20, 20), 64, 64)
        far = app.shape_mask(polygon("A", 45, 45), 64, 64)
        for prediction, reason in [
            (far, "displaced"),
            (np.zeros_like(seed), "empty"),
            (np.ones_like(seed), "excessive_area"),
        ]:
            out, accepted, actual_reason, metrics = app.check_refinement(
                seed, prediction, args
            )
            self.assertFalse(accepted)
            self.assertEqual(actual_reason, reason)
            np.testing.assert_array_equal(out, seed)
        expanded = cv2.dilate(seed, np.ones((5, 5), np.uint8))
        out, accepted, reason, metrics = app.check_refinement(seed, expanded, args)
        self.assertTrue(accepted)
        self.assertEqual(metrics["seed_coverage"], 1)
        self.assertGreater(out.sum(), seed.sum())

    def test_positive_prompts_are_inside_seed_and_box_is_expanded(self):
        seed = app.shape_mask(polygon("A", 20, 20), 64, 64)
        points, labels = app.refinement_points(seed, 0.1)
        self.assertTrue((labels[:2] == [2, 3]).all())
        self.assertEqual(int((labels == 1).sum()), 3)
        for x, y in points[2:] * [64, 64]:
            self.assertTrue(seed[int(round(y)), int(round(x))])
        self.assertLess(points[0, 0] * 64, 20)
        self.assertGreater(points[1, 0] * 64, 32)

    def test_actual_candidate_document_replaces_accepted_geometry(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = app.parse_args([])
            seed = app.shape_mask(polygon("A"), 64, 64)
            refined = cv2.dilate(seed, np.ones((5, 5), np.uint8))
            doc = {"shapes": [polygon("A"), polygon("B", 30, 30)]}
            records = [
                {
                    "frame": 0,
                    "shape_index": 0,
                    "accepted": True,
                    "changed_pixels": int((seed != refined).sum()),
                    "trials": [{"clipped_pixels": 0}],
                }
            ]
            np.savez_compressed(Path(tmp) / "refined_00000_00000.npz", mask=refined)
            proposal = app.apply_refinements(doc, 0, records, tmp, args)
            self.assertNotEqual(
                proposal["shapes"][0]["points"], doc["shapes"][0]["points"]
            )
            self.assertEqual(proposal["shapes"][1], doc["shapes"][1])
            self.assertEqual(proposal["surgslot_original_shapes"], doc["shapes"])
            self.assertTrue(proposal["shapes"][0]["flags"]["refined"])
            self.assertEqual(proposal["shapes"][0]["group_id"], 7)
            self.assertEqual(proposal["shapes"][0]["source_shape_index"], 0)

    def test_baseline_manual_overlap_is_not_a_new_conflict(self):
        masks = app.class_masks([polygon("A"), polygon("B", 8, 8)], 64, 64)
        conflicts, _ = app.overlap_pairs(masks, {"A", "B"}, masks)
        self.assertEqual(conflicts, [])
        changed = dict(masks)
        changed["A"] = cv2.dilate(masks["A"], np.ones((7, 7), np.uint8))
        conflicts, _ = app.overlap_pairs(changed, {"A"}, masks)
        self.assertGreater(conflicts[0]["pixels"], 0)

    def test_primary_json_uses_refined_anchor_and_rejected_shape_falls_back(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            output, visual, work = [tmp / n for n in ("out", "visual", "work")]
            for directory in (output, visual, work):
                directory.mkdir()
            frame = tmp / "frame.jpg"
            Image.new("RGB", (64, 64), (40, 50, 60)).save(frame)
            doc = {
                "version": "7.5.0",
                "flags": {},
                "shapes": [polygon("A"), polygon("B", 32, 32)],
            }
            seed = app.shape_mask(doc["shapes"][0], 64, 64)
            refined = cv2.dilate(seed, np.ones((5, 5), np.uint8))
            records = [
                {
                    "frame": 0,
                    "shape_index": 0,
                    "label": "A",
                    "accepted": True,
                    "changed_pixels": int((seed != refined).sum()),
                    "trials": [{"clipped_pixels": 0, "reason": "accepted"}],
                },
                {
                    "frame": 0,
                    "shape_index": 1,
                    "label": "B",
                    "accepted": False,
                    "changed_pixels": 0,
                    "trials": [{"clipped_pixels": 0, "reason": "displaced"}],
                },
            ]
            np.savez_compressed(work / "refined_00000_00000.npz", mask=refined)
            args = app.parse_args([])
            args.height = args.width = 64
            summary = app.export_results(
                [frame],
                [doc],
                {"A": [0], "B": [0]},
                [],
                work,
                output,
                visual,
                args,
                records,
            )
            actual = json.loads((output / "frame.json").read_text())
            self.assertNotEqual(
                next(x for x in actual["shapes"] if x["label"] == "A")["points"],
                doc["shapes"][0]["points"],
            )
            self.assertEqual(
                next(x for x in actual["shapes"] if x["label"] == "B"), doc["shapes"][1]
            )
            self.assertEqual(actual["surgslot_original_shapes"], doc["shapes"])
            self.assertEqual(
                actual["surgslot"]["boundary_policy"], "refine_with_fallback"
            )
            self.assertEqual(summary["refined_source_shapes"], 1)
            self.assertEqual(summary["original_shapes_preserved"], 1)

    def test_source_artifacts_removed_but_small_targets_and_explicit_keep_survive(self):
        large = {
            "label": "bone",
            "points": [[0, 0], [150, 0], [150, 150], [0, 150]],
            "shape_type": "polygon",
            "flags": {},
        }
        tiny = polygon("bone", 170, 170)
        target = polygon("bubble", 170, 190)
        kept = polygon("bone", 190, 170, {"keep_small": True})
        doc = {"shapes": [large, tiny, target, kept]}
        original = json.loads(json.dumps(doc["shapes"]))
        removed = app.filter_source_shapes(doc, 256, 256, app.parse_args([]))
        self.assertEqual([r["shape_index"] for r in removed], [1])
        self.assertEqual([i for i, _ in app.valid_source_shapes(doc)], [0, 2, 3])
        proposal = app.apply_refinements(doc, 0, [], "/tmp", app.parse_args([]))
        self.assertEqual(proposal["shapes"], [large, target, kept])
        self.assertEqual(proposal["surgslot_original_shapes"], original)

    def test_positive_core_prevents_artificial_horizontal_split(self):
        seed = np.zeros((128, 128), np.uint8)
        seed[10:110, 10:110] = 1
        prediction = seed.copy()
        prediction[54:66] = 0
        result, accepted, _, metrics = app.check_refinement(
            seed, prediction, app.parse_args([])
        )
        self.assertTrue(accepted)
        self.assertTrue(result[54:66, 20:100].all())
        self.assertEqual(len(list(app.components(result))), 1)
        self.assertGreater(metrics["restored_core_pixels"], 0)

    def test_hole_preserving_polygon_roundtrip(self):
        mask = np.zeros((64, 64), np.uint8)
        mask[5:55, 5:55] = 1
        mask[20:35, 20:35] = 0
        polygons, holes = app.mask_shapes(mask, "A", min_area=1, epsilon=0)
        self.assertGreaterEqual(holes, 1)
        exported = app.class_masks(polygons, 64, 64)["A"]
        np.testing.assert_array_equal(exported, mask)

    def test_temporal_prediction_stabilizes_boundary_without_erasing_positive_seed(
        self,
    ):
        target = np.zeros((64, 64), np.uint8)
        target[10:45, 10:45] = 1
        local = target.copy()
        local[15:30, 45:49] = 1
        seed = np.zeros_like(local)
        seed[15:40, 15:40] = 1
        result, _, info = app.temporal_consensus(
            local, seed, [(target * 255, 2), (target * 255, 3)], app.parse_args([])
        )
        self.assertTrue(info["used"])
        self.assertLess(
            np.count_nonzero(result != target), np.count_nonzero(local != target)
        )
        self.assertTrue(result[20:35, 20:35].all())
        result, _, info = app.temporal_consensus(
            local, seed, [(np.zeros_like(local), 2)], app.parse_args([])
        )
        self.assertFalse(info["used"])
        np.testing.assert_array_equal(result, local)

    def test_temporal_tracks_include_already_annotated_frames(self):
        docs = [{"shapes": [polygon("A")]} for _ in range(17)]
        tasks = app.make_tasks({"A": list(range(17))}, docs, app.parse_args([]))
        contributors = []
        for task in tasks:
            for span in task.get("temporal_spans", []):
                if (
                    min(task["anchor"], span["end"])
                    <= 4
                    <= max(task["anchor"], span["end"])
                ):
                    contributors.append(task["anchor"])
        self.assertEqual(sorted(contributors), [0, 8])
        self.assertTrue(all(not t["spans"] for t in tasks))

    def test_boundary_coordination_is_order_independent_and_preserves_manual_overlap(
        self,
    ):
        args = app.parse_args([])
        left = np.zeros((64, 64), np.uint8)
        left[10:45, 5:21] = 1
        right = np.zeros_like(left)
        right[10:45, 25:41] = 1
        sources = {"A": left, "B": right}
        masks = {
            k: cv2.dilate(v, np.ones((9, 9), np.uint8)) for k, v in sources.items()
        }
        result, info = app.coordinate_boundaries(masks, sources, {})
        reversed_result, _ = app.coordinate_boundaries(
            dict(reversed(list(masks.items()))), sources, {}
        )
        self.assertGreater(info["resolved_pixels"], 0)
        for label in sources:
            np.testing.assert_array_equal(result[label], reversed_result[label])
        same_source = {"A": left, "B": left.copy()}
        result, _ = app.coordinate_boundaries(same_source, same_source, {})
        self.assertTrue((result["A"] & result["B"])[15:40, 10:20].all())

    def test_holes_are_reported(self):
        mask = np.zeros((64, 64), np.uint8)
        mask[5:50, 5:50] = 1
        mask[20:30, 20:30] = 0
        _, holes = app.mask_shapes(mask, "A")
        self.assertEqual(holes, 1)

    def test_export_preserves_shapes_images_overlap_and_video(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            source, output, visual, work = [
                tmp / s for s in ("source", "out", "visual", "work")
            ]
            for p in (source, output, visual, work):
                p.mkdir()
            frames = []
            for i in range(3):
                p = source / f"{i:03d}.jpg"
                Image.new("RGB", (64, 64), (50, 60, 70)).save(p)
                frames.append(p)
            docs = [
                {"version": "7.5.0", "flags": {}, "shapes": [polygon("A")]},
                {
                    "version": "7.5.0",
                    "flags": {},
                    "shapes": [polygon("A"), polygon("B", 8, 8)],
                },
                None,
            ]
            anchors = {"A": [0, 1], "B": [1]}
            args = app.parse_args(["--no-refine"])
            args.width = args.height = 64
            tasks = app.make_tasks(anchors, docs, args)
            for task in tasks:
                for span in task["spans"]:
                    step = -1 if span["reverse"] else 1
                    mask = app.shape_mask(polygon(task["label"], 8, 8), 64, 64)
                    for i in range(task["anchor"] + step, span["end"] + step, step):
                        np.savez_compressed(
                            app.prediction_path(work, task, span["reverse"], i),
                            probability=mask * 255,
                        )
            summary = app.export_results(
                frames, docs, anchors, tasks, work, output, visual, args
            )
            self.assertEqual(summary["original_shapes_preserved"], 3)
            self.assertEqual(summary["generated_polygons"], 3)
            for i, frame in enumerate(frames):
                doc = json.loads((output / frame.with_suffix(".json").name).read_text())
                original = docs[i]["shapes"] if docs[i] else []
                self.assertEqual(doc["shapes"][: len(original)], original)
                self.assertEqual((output / frame.name).read_bytes(), frame.read_bytes())
                self.assertIsNone(doc["imageData"])
                self.assertEqual(doc["imagePath"], frame.name)
                self.assertTrue((visual / frame.name).is_file())
            first = json.loads((output / "000.json").read_text())
            self.assertEqual({s["label"] for s in first["shapes"]}, {"A", "B"})
            self.assertTrue(first["shapes"][-1]["flags"]["needs_review"])
            video = cv2.VideoCapture(str(visual / "preview.mp4"))
            self.assertTrue(video.isOpened())
            self.assertEqual(int(video.get(cv2.CAP_PROP_FRAME_COUNT)), 3)
            self.assertEqual(video.get(cv2.CAP_PROP_FPS), 10)
            video.release()

    def test_component_tracking_resets_state_and_uses_reverse(self):
        import torch

        class FakeTracker:
            def __init__(self):
                self.calls, self.resets = [], 0

            def clear_all_points_in_video(self, state):
                self.resets += 1

            def add_new_mask(self, **kwargs):
                self.mask = kwargs["mask"]

            def propagate_in_video(self, **kwargs):
                self.calls.append(kwargs["reverse"])
                step = -1 if kwargs["reverse"] else 1
                for offset in range(kwargs["max_frame_num_to_track"] + 1):
                    index = kwargs["start_frame_idx"] + step * offset
                    logits = self.mask[None, None] * 20 - 10
                    # Simulate occlusion at the leading edge.
                    original_index = kwargs["inference_state"]["images"][index]
                    presence = torch.tensor([-10.0 if original_index == 0 else 10.0])
                    yield index, [1], None, logits, presence

        with tempfile.TemporaryDirectory() as tmp:
            docs = [None, {"shapes": [polygon("A"), polygon("A", 32, 32)]}, None]
            task = app.make_tasks({"A": [1]}, docs, app.parse_args(["--no-refine"]))[0]
            tracker = FakeTracker()
            result = app.run_task(
                tracker,
                {
                    "video_height": 64,
                    "video_width": 64,
                    "num_frames": 3,
                    "images": [0, 1, 2],
                    "cached_features": {},
                },
                task,
                docs,
                tmp,
                app.parse_args(["--no-refine"]),
            )
            self.assertEqual(result["components"], 2)
            self.assertEqual(tracker.calls, [False, False, False, False])
            with np.load(app.prediction_path(tmp, task, True, 0)) as data:
                self.assertFalse(data["probability"].any())
            with np.load(app.prediction_path(tmp, task, False, 2)) as data:
                self.assertEqual(
                    len(list(app.components(data["probability"] > 127))), 2
                )


if __name__ == "__main__":
    unittest.main()
