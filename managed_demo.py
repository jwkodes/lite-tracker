"""External track lifecycle controller; deliberately no poses or SLAM landmarks."""

import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import argparse
import json
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
import torch

from src.lite_tracker import LiteTracker
from src.managed_lite_tracker import ManagedLiteTracker
from src.model_utils import get_points_on_a_grid


class TrackingController:
    """Stand-in for VSLAM keyframe policy, including patience and replenishment.

    The tracker wrapper only executes the requested session handovers.
    """

    def __init__(
        self,
        target_points=100,
        grid_size=10,
        min_distance=15,
        min_keyframe_gap=15,
        max_keyframe_gap=60,
        survivor_ratio=0.75,
        coverage_ratio=0.75,
        patience=3,
    ):
        if min(target_points, grid_size, min_distance, min_keyframe_gap, patience) < 1:
            raise ValueError("Counts, spacing and minimum interval must be positive.")
        if max_keyframe_gap < min_keyframe_gap:
            raise ValueError("Maximum keyframe gap must be at least the minimum gap.")
        if not 0 < survivor_ratio <= 1 or not 0 < coverage_ratio <= 1:
            raise ValueError("Keyframe ratios must be in (0, 1].")
        if grid_size**2 > target_points:
            raise ValueError("Initial grid cannot exceed target_points.")
        self.target_points = target_points
        self.grid_size = grid_size
        self.min_distance = min_distance
        self.min_keyframe_gap = min_keyframe_gap
        self.max_keyframe_gap = max_keyframe_gap
        self.survivor_ratio = survivor_ratio
        self.coverage_ratio = coverage_ratio
        self.patience = patience
        self.reference_ids = set()
        self.reference_coverage = 0
        self.last_keyframe = None
        self.last_reason = None
        self.last_dropped = 0
        self.keyframe_id = -1
        self.low_survivor_frames = 0
        self.low_coverage_frames = 0
        self.event = {}
        self.metrics = {}

    def _measure(self, tracker, outputs):
        h, w = self.image_shape
        coords, visible, _ = outputs
        positions = {
            tid: xy
            for tid, xy, seen in zip(
                tracker.track_ids,
                coords[0, 0].cpu().numpy(),
                visible[0, 0].cpu().numpy(),
            )
            if seen and np.isfinite(xy).all() and 0 <= xy[0] < w and 0 <= xy[1] < h
        }
        occupied = {
            (min(3, int(x * 4 / w)), min(3, int(y * 4 / h)))
            for x, y in positions.values()
        }
        return positions, len(occupied)

    def _grid_top_up(self, positions, budget, device):
        grid = get_points_on_a_grid(self.grid_size, self.image_shape)
        candidates = grid[0].numpy()
        existing = np.asarray(list(positions.values()), dtype=np.float32).reshape(-1, 2)
        distances = (
            np.square(candidates[:, None] - existing[None]).sum(axis=2).min(axis=1)
            if len(existing)
            else np.full(len(candidates), np.inf)
        )
        chosen = []
        available = np.ones(len(candidates), dtype=bool)
        for _ in range(budget):
            eligible = available & (distances > self.min_distance**2)
            if not eligible.any():
                break
            index = int(np.argmax(np.where(eligible, distances, -np.inf)))
            chosen.append(index)
            available[index] = False
            distances = np.minimum(
                distances, np.square(candidates - candidates[index]).sum(axis=1)
            )
        return grid[:, chosen].to(device)

    def process_frame(self, tracker, frame):
        """Track once; if triggered, re-prime on this same boundary image."""
        self.image_shape = tuple(frame.shape[-2:])
        index = tracker.frame_index
        initial = self.last_keyframe is None
        if initial and (index != 0 or tracker.track_ids):
            raise ValueError("Start this controller with a fresh, empty tracker.")
        if initial and self.grid_size**2 > tracker.max_points:
            raise ValueError("Initial grid exceeds tracker capacity.")
        self.event = {
            "keyframe": False,
            "keyframe_id": self.keyframe_id,
            "keyframe_reason": None,
            "requested_additions": 0,
            "capacity_blocked": 0,
            "added": [],
            "carried": [],
            "dropped": [],
        }
        if initial:
            grid = get_points_on_a_grid(self.grid_size, self.image_shape).to(
                frame.device
            )
            queries = torch.cat((torch.zeros_like(grid[..., :1]), grid), dim=2)
            self.event["added"] = tracker.add_queries(queries)
            self.event["requested_additions"] = grid.shape[1]
        outputs = tracker(frame)
        positions, occupied = self._measure(tracker, outputs)
        survivors = len(set(positions) & self.reference_ids) / max(
            1, len(self.reference_ids)
        )
        coverage = occupied / max(1, self.reference_coverage)
        reason = "initial" if initial else None
        if not initial:
            self.low_survivor_frames = (
                self.low_survivor_frames + 1 if survivors < self.survivor_ratio else 0
            )
            self.low_coverage_frames = (
                self.low_coverage_frames + 1 if coverage < self.coverage_ratio else 0
            )
            gap = index - self.last_keyframe
            if gap >= self.min_keyframe_gap:
                reasons = []
                if self.low_survivor_frames >= self.patience:
                    reasons.append("low_survivors")
                if self.low_coverage_frames >= self.patience:
                    reasons.append("low_coverage")
                if gap >= self.max_keyframe_gap:
                    reasons.append("interval")
                reason = "+".join(reasons) or None
        # Preserve the old session's observations for auditing the handover.
        self.event["observed_usable_ids"] = list(positions)
        self.event["observed_usable_xy"] = [xy.tolist() for xy in positions.values()]
        self.event["observed_survivor_fraction"] = survivors
        self.event["observed_coverage_fraction"] = coverage
        if reason is not None:
            if not initial:
                old_ids = tracker.track_ids
                carry = list(positions)
                requested = max(0, self.target_points - len(carry))
                free = tracker.max_points - len(carry)
                self.event["requested_additions"] = requested
                self.event["capacity_blocked"] = max(0, requested - free)
                points = self._grid_top_up(
                    positions, min(requested, free), frame.device
                )
                outputs = tracker.start_keyframe(carry, points)
                self.event["carried"] = carry
                self.event["dropped"] = [tid for tid in old_ids if tid not in positions]
                self.event["added"] = tracker.track_ids[len(carry) :]
                positions, occupied = self._measure(tracker, outputs)
            self.keyframe_id += 1
            self.last_keyframe = index
            self.last_reason = reason
            self.last_dropped = len(self.event["dropped"])
            self.reference_ids = set(positions)
            self.reference_coverage = occupied
            self.low_survivor_frames = self.low_coverage_frames = 0
            self.event.update(
                keyframe=True,
                keyframe_id=self.keyframe_id,
                keyframe_reason=reason,
            )
        self.event.update(
            session_index=tracker.session_index,
            last_keyframe_frame=self.last_keyframe,
            last_keyframe_reason=self.last_reason,
            last_keyframe_dropped=self.last_dropped,
        )
        self.metrics = {"usable_visible": len(positions), "occupied_cells": occupied}
        return outputs


def draw_overlay(bgr, record):
    """Render filled, outlined markers; also usable to replay saved JSONL logs."""
    h, w = bgr.shape[:2]
    added = set(record["added"])
    for tid, (x, y), seen in zip(record["track_ids"], record["xy"], record["visible"]):
        if not np.isfinite([x, y]).all() or not (0 <= x < w and 0 <= y < h):
            continue
        position = (round(x), round(y))
        color = (255, 170, 0) if seen else (0, 165, 255)  # blue / orange, BGR
        cv2.circle(bgr, position, 8, (0, 0, 0), -1, cv2.LINE_AA)
        cv2.circle(bgr, position, 6, color, -1, cv2.LINE_AA)
        if tid in added:
            cv2.circle(bgr, position, 11, (255, 255, 255), 2, cv2.LINE_AA)
        label_position = (position[0] + 10, position[1] - 8)
        cv2.putText(
            bgr,
            str(tid),
            label_position,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            bgr,
            str(tid),
            label_position,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    recent = record["frame"] - record.get(
        "last_keyframe_frame", record["frame"]
    ) < record.get("fps", 25)
    kind = "NEW SESSION | Frame" if recent else "Frame"
    label = (
        f"{kind} {record['frame']} | KF {record.get('keyframe_id', '-')} "
        f"| retained {len(record['track_ids'])} | visible {record.get('usable_visible', sum(record['visible']))} "
        f"| +{len(added)}"
    )
    cv2.rectangle(bgr, (0, 0), (w, 100), (0, 0, 0), -1)
    cv2.putText(
        bgr,
        label,
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    legend = "Blue: visible | Orange: invisible (retained) | White ring: new"
    cv2.putText(
        bgr,
        legend,
        (10, 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    detail = (
        f"Last KF {record.get('last_keyframe_frame', 0)}: "
        f"{record.get('last_keyframe_reason', 'initial')} | "
        f"left behind {record.get('last_keyframe_dropped', 0)} points"
    )
    if record.get("capacity_blocked"):
        detail += f" | capacity blocks {record['capacity_blocked']} requested additions"
    cv2.putText(
        bgr,
        detail,
        (10, 84),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return bgr


def synchronize(device):
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--checkpoint", default="weights/scaled_online.pth")
    source.add_argument(
        "--onnx-dir", help="Use exported ONNX components with CPU orchestration."
    )
    parser.add_argument("--output", default="results/managed")
    parser.add_argument(
        "--device", choices=["auto", "cpu", "mps", "cuda"], default="auto"
    )
    parser.add_argument("--max-frames", type=int, default=300)
    parser.add_argument(
        "--max-points",
        type=int,
        default=512,
        help="Per-session query cap; invisible queries persist until the next keyframe.",
    )
    parser.add_argument(
        "--target-points",
        type=int,
        default=100,
        help="Desired usable visible count after a keyframe top-up.",
    )
    parser.add_argument("--grid-size", type=int, default=10)
    parser.add_argument("--min-distance", type=int, default=15)
    parser.add_argument("--kf-min-gap", type=int, default=15)
    parser.add_argument("--kf-max-gap", type=int, default=60)
    parser.add_argument("--kf-survivor-ratio", type=float, default=0.75)
    parser.add_argument("--kf-coverage-ratio", type=float, default=0.75)
    parser.add_argument("--kf-patience", type=int, default=3)
    args = parser.parse_args()
    if args.max_frames < 1 or args.max_points < args.grid_size**2:
        parser.error(
            "Require positive max-frames and enough capacity for the initial grid."
        )
    controller = TrackingController(
        target_points=args.target_points,
        grid_size=args.grid_size,
        min_distance=args.min_distance,
        min_keyframe_gap=args.kf_min_gap,
        max_keyframe_gap=args.kf_max_gap,
        survivor_ratio=args.kf_survivor_ratio,
        coverage_ratio=args.kf_coverage_ratio,
        patience=args.kf_patience,
    )
    device = args.device
    if args.onnx_dir:
        if device not in ("auto", "cpu"):
            parser.error("The hybrid ONNX demo requires --device cpu (or auto).")
        device = "cpu"
    if device == "auto":
        device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
    torch.set_num_threads(2)
    if args.onnx_dir:
        from onnx_example import OnnxLiteTracker

        model = OnnxLiteTracker(args.onnx_dir, providers=["CPUExecutionProvider"])
    else:
        model = LiteTracker()
        weights = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(weights.get("model", weights))
        model = model.to(device).eval()
    tracker = ManagedLiteTracker(model, max_points=args.max_points)
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {args.video}")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps if np.isfinite(fps) and fps > 0 else 30.0
    writer = None
    timings = []
    totals = {
        "added": 0,
        "dropped_at_keyframes": 0,
        "keyframes": 0,
        "max_retained": 0,
        "capacity_limited_keyframes": 0,
    }
    try:
        with (output / "frames.jsonl").open("w") as log:
            for index in range(args.max_frames):
                ok, bgr = cap.read()
                if not ok:
                    break
                if writer is None:
                    h, w = bgr.shape[:2]
                    writer = cv2.VideoWriter(
                        str(output / "tracks.mp4"),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        fps,
                        (w, h),
                    )
                    if not writer.isOpened():
                        raise RuntimeError("Cannot open video encoder.")
                frame = (
                    torch.from_numpy(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
                    .permute(2, 0, 1)[None]
                    .to(device)
                    .float()
                )
                synchronize(device)
                start = perf_counter()
                coords, visibility, confidence = controller.process_frame(
                    tracker, frame
                )
                synchronize(device)
                elapsed = (perf_counter() - start) * 1000
                timings.append(elapsed)
                ids = tracker.track_ids
                xy = coords[0, 0].cpu().numpy()
                visible = visibility[0, 0].cpu().numpy()
                conf = confidence[0, 0].cpu().numpy()
                metrics = controller.metrics
                added = controller.event["added"]
                totals["dropped_at_keyframes"] += len(controller.event["dropped"])
                totals["added"] += len(added)
                totals["keyframes"] += int(controller.event["keyframe"])
                totals["capacity_limited_keyframes"] += int(
                    controller.event["capacity_blocked"] > 0
                )
                totals["max_retained"] = max(totals["max_retained"], len(ids))
                record = {
                    "frame": index,
                    "time_s": index / fps,
                    "fps": fps,
                    "track_ids": ids,
                    "xy": xy.tolist(),
                    "visible": visible.tolist(),
                    "confidence": conf.tolist(),
                    **controller.event,
                    **metrics,
                    "processing_ms": elapsed,
                    "backend_frame_index": model.online_ind,
                }
                log.write(json.dumps(record) + "\n")
                draw_overlay(bgr, record)
                writer.write(bgr)
                if (index + 1) % 20 == 0:
                    print(f"Processed {index + 1} frames; {totals}", flush=True)
    finally:
        cap.release()
        if writer is not None:
            writer.release()
    if not timings:
        raise RuntimeError("No frames decoded.")
    summary = dict(
        video=str(Path(args.video).resolve()),
        device=device,
        backend="onnxruntime" if args.onnx_dir else "pytorch",
        frames=len(timings),
        model_resolution_hw=list(model.model_resolution),
        input_resolution_hw=[h, w],
        max_points=args.max_points,
        target_points=args.target_points,
        retirement="keyframe_session_boundary_only",
        keyframe_rules={
            "min_gap": args.kf_min_gap,
            "max_gap": args.kf_max_gap,
            "survivor_ratio": args.kf_survivor_ratio,
            "coverage_ratio": args.kf_coverage_ratio,
            "patience": args.kf_patience,
        },
        initial_points="grid",
        replenishment="grid",
        grid_size=args.grid_size,
        processing_median_ms=float(np.median(timings)),
        **totals,
        note="Simulated keyframes only; no pose, landmark, stereo or ground-truth accuracy evaluation. "
        "Timing includes policy, tracking, boundary reinitialization and device synchronization; excludes decoding and rendering.",
    )
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"Video and logs: {output.resolve()}")


if __name__ == "__main__":
    main()
