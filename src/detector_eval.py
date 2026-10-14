"""
Task 3.5 - measure the block detector.

You list every block you will put on the board (BLOCKS below), place them all
at once, and press Enter. The script captures FRAMES frames, runs the same
detector the GUI runs (find_contours over the height bands, then find_blocks),
matches each detection to the nearest true block, logs everything to CSV, and
writes:

    position_error_map.png    error at every true position, over the board
    position_error_vs_radius.png
    detection_rate_map.png    how often each block was found + false positives
    color_confusion*.png      true color vs detected color (one per lighting)
    angle_error.png           true vs detected orientation
    summary.txt               all the numbers for the report

Usage (from src/, with the control station CLOSED - only one process can own the camera):
    python3 detector_eval.py                       # live: capture and evaluate
    python3 detector_eval.py --analyze <run_dir>   # re-plot an earlier run, no camera needed

Each block is written as [color, size, x, y, angle] - see BLOCKS below.
"""
import argparse
import csv
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# --------------------------------------------------------------------------
# Settings
# --------------------------------------------------------------------------
FRAMES = 30                # detection frames logged
SETTLE_FRAMES = 40         # frames discarded after you press Enter (the temporal filter needs time)
WARMUP_FRAMES = 30         # frames discarded before calibrating
MATCH_RADIUS_MM = 40.0     # a detection farther than this from every true block is a false positive
ANGLE_SIGN = +1            # flip to -1 if the summary says the angle convention looks mirrored

# Same height bands VideoThread.run() uses.
HEIGHT_BANDS = [(15, 57), (58, 95), (96, 133), (134, 171), (172, 209)]

COLORS = ["red", "orange", "yellow", "green", "blue", "purple"]

# Copied from camera.py so --analyze works on a machine without the camera libraries.
TAG_WORLD_XY = [(100.0, 300.0), (100.0, -300.0), (400.0, -300.0), (400.0, 300.0)]
TABLE_EDGE_XY = [(-100, 500), (-100, -500), (500, -500), (500, 500)]
BOARD_CENTER_XY = (250.0, 0.0)   # center of the tag rectangle; "distance from center" is measured from here

OUT_ROOT = Path(__file__).resolve().parent.parent / "results" / "detector_eval"


# --------------------------------------------------------------------------
# Every block you will put on the board, one entry per block:
#     [color, size, x, y, angle]
#       color  "red" "orange" "yellow" "green" "blue" "purple"
#       size   "big" (~38 mm) or "small" (~25 mm)
#       x, y   world mm of the block's center (the block sits on the table)
#       angle  degrees, counterclockwise seen from above, from world +X (only mod 90 matters)
# Keep blocks > 2 block widths apart so they never merge into one contour, and
# more than MATCH_RADIUS_MM apart so a detection can't be matched to the wrong block.
# --------------------------------------------------------------------------
LIGHTING = "normal"        # label for this run; change it and run again under other lighting

BLOCKS = [
    # color     size     x     y   angle
    ["red",    "big",   150,  200,  0],
    ["orange", "big",   150,    0,  0],
    ["yellow", "big",   150, -200,  0],
    ["green",  "big",   350,  200,  0],
    ["blue",   "big",   350,    0, 30],
    ["purple", "big",   350, -200, 45],
    ["red",    "small", 250,  100, 15],
    ["blue",   "small", 250, -100, 60],
]


def parse_block(entry):
    """[color, size, x, y, angle] -> dict used by the rest of the script."""
    color, size, x, y, angle = entry
    if color not in COLORS:
        raise ValueError(f"unknown color {color!r} in {entry}; use one of {COLORS}")
    if size not in ("big", "small"):
        raise ValueError(f"size must be 'big' or 'small', got {size!r} in {entry}")
    return {"color": color, "size": size, "x": float(x), "y": float(y), "angle": float(angle)}


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------
def angle_diff_mod90(a, b):
    """Signed difference a - b for square blocks, wrapped into [-45, 45)."""
    return (a - b + 45.0) % 90.0 - 45.0


def mean_angle_mod90(angles_deg):
    """Circular mean for a 90-deg-periodic angle (so 1 and 89 average to 0, not 45)."""
    t = np.radians(np.asarray(angles_deg) * 4.0)
    return np.degrees(np.arctan2(np.sin(t).mean(), np.cos(t).mean())) / 4.0 % 90.0


def match(gt_blocks, dets):
    """Greedy one-to-one matching by XY distance. Returns {gt_index: det_index}."""
    pairs = []
    for gi, g in enumerate(gt_blocks):
        for di, d in enumerate(dets):
            dist = np.hypot(d["x"] - g["x"], d["y"] - g["y"])
            if dist <= MATCH_RADIUS_MM:
                pairs.append((dist, gi, di))
    pairs.sort()
    gt_used, det_used, result = set(), set(), {}
    for _, gi, di in pairs:
        if gi not in gt_used and di not in det_used:
            result[gi] = di
            gt_used.add(gi)
            det_used.add(di)
    return result


def write_csv(path, rows, fields):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


GT_FIELDS = ["trial", "lighting", "frame", "block", "gt_x", "gt_y", "gt_color", "gt_size", "gt_angle",
             "found", "det_x", "det_y", "det_z", "det_color", "det_size", "det_side_mm", "det_angle"]
FP_FIELDS = ["trial", "lighting", "frame", "det_x", "det_y", "det_z", "det_color", "det_size"]
FRAME_FIELDS = ["trial", "lighting", "frame", "n_true", "n_detected"]


# --------------------------------------------------------------------------
# Live capture (needs the camera)
# --------------------------------------------------------------------------
def grab(camera, cv2, detect_tags=False):
    """One frame through the same pipeline as VideoThread.run(). Returns False on timeout."""
    try:
        frames = camera.pipeline.wait_for_frames(timeout_ms=2000)
    except RuntimeError:
        return False
    aligned = camera.align.process(frames)
    color_frame = aligned.get_color_frame()
    depth_frame = aligned.get_depth_frame()
    if not color_frame or not depth_frame:
        return False
    depth_frame = camera.spatial_filter.process(depth_frame)
    depth_frame = camera.temporal_filter.process(depth_frame)
    depth_frame = camera.hole_filling.process(depth_frame)

    bgr = np.asanyarray(color_frame.get_data()).copy()
    camera.video_frame = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    camera.depth_frame_raw = np.asanyarray(depth_frame.get_data()).copy()
    if detect_tags:
        camera.tag_detections = camera.tag_detector.detect(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY))
    return True


def detect(camera):
    """Run the detector exactly as VideoThread does and return plain dicts."""
    camera.update_workspace_frame()
    camera.block_contours.clear()
    camera.warp_depth_image()
    for lo, hi in HEIGHT_BANDS:
        camera.find_contours(lo, hi)
    camera.find_blocks()
    out = []
    for d in camera.block_detections:
        X, Y, Z = (float(v) for v in d["world"][:3])
        out.append({"x": X, "y": Y, "z": Z, "color": d["color"], "size": d["size"],
                    "side_mm": float(d["side_mm"]), "angle": float(d["angle_degree"])})
    return out


def calibrate(camera, cv2):
    print("Warming up the camera...")
    for _ in range(WARMUP_FRAMES):
        grab(camera, cv2)
    print("Clear the board (tags 1-4 visible, fixture out of view) and press Enter to calibrate.")
    input()
    msg = "no frames"
    for _ in range(60):
        if not grab(camera, cv2, detect_tags=True):
            continue
        ok, msg = camera.estimate_extrinsics_from_tags()
        if ok:
            print("Calibrated.")
            return
    sys.exit(f"Calibration failed: {msg}")


def run_live(out_dir):
    blocks = [parse_block(e) for e in BLOCKS]       # catch typos before calibrating

    import cv2
    from camera import Camera

    camera = Camera()
    if not camera.camera_connected:
        sys.exit("Camera not connected (is the control station still running?).")

    name, lighting = "run", LIGHTING
    gt_rows, fp_rows, frame_rows = [], [], []
    try:
        calibrate(camera, cv2)

        print(f"\nPlace these {len(blocks)} blocks (lighting: {lighting}):")
        for i, b in enumerate(blocks):
            print(f"  [{i}] {b['color']:7s} {b['size']:5s} at ({b['x']:.0f}, {b['y']:.0f})  angle {b['angle']:.0f} deg")
        input("Keep your hands clear, then press Enter: ")

        for _ in range(SETTLE_FRAMES):
            grab(camera, cv2)

        # One raw RGB + 16-bit depth pair (useful for offline debugging).
        cv2.imwrite(str(out_dir / "rgb.png"), cv2.cvtColor(camera.video_frame, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(out_dir / "depth.png"), camera.depth_frame_raw.astype(np.uint16))

        n_logged = 0
        while n_logged < FRAMES:
            if not grab(camera, cv2):
                continue
            dets = detect(camera)
            pairs = match(blocks, dets)
            for gi, g in enumerate(blocks):
                row = {"trial": name, "lighting": lighting, "frame": n_logged, "block": gi,
                       "gt_x": g["x"], "gt_y": g["y"], "gt_color": g["color"],
                       "gt_size": g["size"], "gt_angle": g["angle"], "found": int(gi in pairs)}
                if gi in pairs:
                    d = dets[pairs[gi]]
                    row.update({"det_x": d["x"], "det_y": d["y"], "det_z": d["z"],
                                "det_color": d["color"], "det_size": d["size"],
                                "det_side_mm": d["side_mm"], "det_angle": d["angle"]})
                gt_rows.append(row)
            matched = set(pairs.values())
            for di, d in enumerate(dets):
                if di not in matched:
                    fp_rows.append({"trial": name, "lighting": lighting, "frame": n_logged,
                                    "det_x": d["x"], "det_y": d["y"], "det_z": d["z"],
                                    "det_color": d["color"], "det_size": d["size"]})
            frame_rows.append({"trial": name, "lighting": lighting, "frame": n_logged,
                               "n_true": len(blocks), "n_detected": len(dets)})
            n_logged += 1

        # Annotated view from the last frame, so you can see what the detector saw.
        cv2.imwrite(str(out_dir / "detections.png"), cv2.cvtColor(camera.workspace_frame, cv2.COLOR_RGB2BGR))
        print(f"Last frame: found {len(pairs)}/{len(blocks)}, {len(dets) - len(matched)} false positive(s)")

        write_csv(out_dir / "gt_frames.csv", gt_rows, GT_FIELDS)
        write_csv(out_dir / "false_positives.csv", fp_rows, FP_FIELDS)
        write_csv(out_dir / "frames.csv", frame_rows, FRAME_FIELDS)
    finally:
        camera.stop()


# --------------------------------------------------------------------------
# Analysis (works from the CSVs alone)
# --------------------------------------------------------------------------
def draw_board(ax):
    """Board drawn the way the Workspace view shows it: +X up, +Y to the left."""
    edge = TABLE_EDGE_XY + [TABLE_EDGE_XY[0]]
    ax.plot([p[1] for p in edge], [p[0] for p in edge], "k-", lw=1, label="table edge")
    ax.plot([p[1] for p in TAG_WORLD_XY], [p[0] for p in TAG_WORLD_XY], "ks", ms=7, mfc="none", label="tags")
    ax.plot(0, 0, "k^", ms=10, label="arm base")
    ax.set_xlabel("world Y (mm)")
    ax.set_ylabel("world X (mm)")
    ax.invert_xaxis()
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)


def analyze(out_dir):
    out_dir = Path(out_dir)
    gt = read_csv(out_dir / "gt_frames.csv")
    fps = read_csv(out_dir / "false_positives.csv") if (out_dir / "false_positives.csv").exists() else []
    frames = read_csv(out_dir / "frames.csv") if (out_dir / "frames.csv").exists() else []
    lines = []

    def say(s=""):
        print(s)
        lines.append(s)

    for r in gt:
        r["found"] = int(r["found"])
        for k in ("gt_x", "gt_y", "gt_angle", "det_x", "det_y", "det_z", "det_side_mm", "det_angle"):
            r[k] = float(r[k]) if r.get(k) not in (None, "") else np.nan
    found = [r for r in gt if r["found"]]

    say(f"Detector evaluation: {out_dir}")
    say(f"Blocks: {len({r['block'] for r in gt})}, frames: {len(frames)}, block-frames: {len(gt)}, "
        f"matched: {len(found)}, match radius {MATCH_RADIUS_MM:.0f} mm")

    # ---------------- 1. Position ----------------
    say("\n== 1. Position (detected - true, XY) ==")
    if found:
        ex = np.array([r["det_x"] - r["gt_x"] for r in found])
        ey = np.array([r["det_y"] - r["gt_y"] for r in found])
        e = np.hypot(ex, ey)
        say(f"XY error  mean {e.mean():.1f} mm, median {np.median(e):.1f}, RMS {np.sqrt((e**2).mean()):.1f}, "
            f"max {e.max():.1f}")
        say(f"Mean bias (systematic offset)  dx {ex.mean():+.1f} mm, dy {ey.mean():+.1f} mm")

        # Per true block position: average over frames (and over trials at the same spot).
        by_pos = defaultdict(list)
        for r, dx, dy in zip(found, ex, ey):
            by_pos[(r["gt_x"], r["gt_y"])].append((dx, dy))
        P = np.array(list(by_pos.keys()))
        D = np.array([np.mean(v, axis=0) for v in by_pos.values()])
        E = np.hypot(D[:, 0], D[:, 1])

        fig, ax = plt.subplots(figsize=(10, 7))
        draw_board(ax)
        sc = ax.scatter(P[:, 1], P[:, 0], c=E, cmap="viridis", vmin=0, vmax=max(20.0, E.max()),
                        s=180, edgecolors="k", zorder=3)
        ARROW_SCALE = 5.0
        ax.quiver(P[:, 1], P[:, 0], D[:, 1] * ARROW_SCALE, D[:, 0] * ARROW_SCALE,
                  angles="xy", scale_units="xy", scale=1, color="r", width=0.004, zorder=4)
        for (px, py), err in zip(P, E):
            ax.annotate(f"{err:.0f}", (py, px), textcoords="offset points", xytext=(8, 8), fontsize=8)
        fig.colorbar(sc, ax=ax, label="mean XY error (mm)")
        ax.set_title(f"Detector position error (arrows: error direction x{ARROW_SCALE:.0f})")
        ax.legend(loc="lower left", fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / "position_error_map.png", dpi=150)
        plt.close(fig)

        # Error vs distance from the board center.
        rad = np.hypot(P[:, 0] - BOARD_CENTER_XY[0], P[:, 1] - BOARD_CENTER_XY[1])
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.scatter(rad, E, s=40)
        bins = np.arange(0, rad.max() + 50, 50)
        say("Error vs distance from board center (per position, mean over frames):")
        centers, means = [], []
        for lo, hi in zip(bins[:-1], bins[1:]):
            sel = (rad >= lo) & (rad < hi)
            if sel.any():
                centers.append((lo + hi) / 2)
                means.append(E[sel].mean())
                say(f"  {lo:4.0f}-{hi:4.0f} mm: mean {E[sel].mean():5.1f} mm, max {E[sel].max():5.1f} mm "
                    f"({sel.sum()} positions)")
        ax.plot(centers, means, "r-o", label="bin mean (50 mm bins)")
        ax.set_xlabel(f"distance from board center {BOARD_CENTER_XY} (mm)")
        ax.set_ylabel("XY error (mm)")
        ax.set_title("Position error vs distance from center")
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / "position_error_vs_radius.png", dpi=150)
        plt.close(fig)

        dz = [r["det_z"] for r in found if r["gt_size"] == "big"]
        if dz:
            say(f"Top-face Z of big blocks: mean {np.mean(dz):.1f} mm, std {np.std(dz):.1f} mm "
                f"(should be ~ the block height)")
    else:
        say("No matched detections.")

    # ---------------- 2. Color ----------------
    say("\n== 2. Color confusion (rows = true, columns = detected; counts are block-frames) ==")
    pred_labels = COLORS + ["unknown", "missed"]
    for lighting in sorted({r["lighting"] for r in gt}):
        rows = [r for r in gt if r["lighting"] == lighting]
        true_labels = [c for c in COLORS if any(r["gt_color"] == c for r in rows)]
        if not true_labels:
            continue
        M = np.zeros((len(true_labels), len(pred_labels)), dtype=int)
        for r in rows:
            if r["gt_color"] not in true_labels:
                continue
            p = r["det_color"] if r["found"] else "missed"
            p = p if p in pred_labels else "unknown"
            M[true_labels.index(r["gt_color"]), pred_labels.index(p)] += 1

        correct = sum(M[i, pred_labels.index(c)] for i, c in enumerate(true_labels))
        classified = M[:, :len(COLORS) + 1].sum()        # found blocks only
        say(f"[lighting: {lighting}] accuracy on found blocks: {correct}/{classified} "
            f"= {100 * correct / max(classified, 1):.1f}%")
        say("        " + " ".join(f"{p[:7]:>7s}" for p in pred_labels))
        for i, c in enumerate(true_labels):
            say(f"{c:>7s} " + " ".join(f"{v:7d}" for v in M[i]))
        confusions = [(M[i, j], c, pred_labels[j]) for i, c in enumerate(true_labels)
                      for j in range(len(COLORS)) if pred_labels[j] != c and M[i, j] > 0]
        if confusions:
            n, a, b = max(confusions)
            say(f"Most confused: true {a} -> detected {b} ({n} block-frames)")
        else:
            say("No color confusions.")

        fig, ax = plt.subplots(figsize=(8, 5.5))
        Mn = M / np.maximum(M.sum(axis=1, keepdims=True), 1)
        im = ax.imshow(Mn, cmap="Blues", vmin=0, vmax=1)
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                if M[i, j]:
                    ax.text(j, i, f"{M[i, j]}\n{100 * Mn[i, j]:.0f}%", ha="center", va="center",
                            fontsize=8, color="white" if Mn[i, j] > 0.5 else "black")
        ax.set_xticks(range(len(pred_labels)), pred_labels, rotation=30)
        ax.set_yticks(range(len(true_labels)), true_labels)
        ax.set_xlabel("detected")
        ax.set_ylabel("true")
        ax.set_title(f"Color confusion matrix (lighting: {lighting})")
        fig.colorbar(im, ax=ax, label="fraction of row")
        fig.tight_layout()
        suffix = "" if lighting == "normal" else f"_{lighting}"
        fig.savefig(out_dir / f"color_confusion{suffix}.png", dpi=150)
        plt.close(fig)

    # ---------------- 3. Size and orientation ----------------
    say("\n== 3. Size and orientation ==")
    sizes = sorted({r["gt_size"] for r in gt})
    for s in sizes:
        rs = [r for r in found if r["gt_size"] == s]
        if rs:
            ok = sum(r["det_size"] == s for r in rs)
            side = np.array([r["det_side_mm"] for r in rs])
            say(f"Size {s:5s}: {ok}/{len(rs)} correct ({100 * ok / len(rs):.1f}%), "
                f"measured side {side.mean():.1f} +/- {side.std():.1f} mm")
    if found:
        ta = np.array([r["gt_angle"] for r in found])
        da = np.array([r["det_angle"] for r in found])
        err = angle_diff_mod90(ANGLE_SIGN * da, ta)
        err_flip = angle_diff_mod90(-ANGLE_SIGN * da, ta)
        say(f"Angle error (mod 90): mean |err| {np.abs(err).mean():.1f} deg, max {np.abs(err).max():.1f} deg, "
            f"bias {err.mean():+.1f} deg")
        if np.abs(err_flip).mean() < 0.5 * np.abs(err).mean():
            say(f"  NOTE: with ANGLE_SIGN = {-ANGLE_SIGN} the mean error would be "
                f"{np.abs(err_flip).mean():.1f} deg - the detector's angle runs the other way. "
                f"Flip ANGLE_SIGN and re-run --analyze.")
        for a in sorted(set(ta)):
            sel = ta == a
            say(f"  true {a:5.1f} deg: detected mean {mean_angle_mod90(ANGLE_SIGN * da[sel]):5.1f} deg, "
                f"|err| {np.abs(err[sel]).mean():4.1f} deg ({sel.sum()} frames)")

        fig, ax = plt.subplots(figsize=(5.5, 5))
        ax.scatter(ta % 90, (ANGLE_SIGN * da) % 90, alpha=0.4)
        ax.plot([0, 90], [0, 90], "k--", lw=1, label="perfect")
        ax.set_xlim(-2, 92)
        ax.set_ylim(-2, 92)
        ax.set_xlabel("true angle mod 90 (deg)")
        ax.set_ylabel("detected angle mod 90 (deg)")
        ax.set_title("Orientation")
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / "angle_error.png", dpi=150)
        plt.close(fig)

    # ---------------- 4. Detection rate ----------------
    say("\n== 4. Detection rate ==")
    if gt:
        say(f"Found {len(found)}/{len(gt)} block-frames = {100 * len(found) / len(gt):.1f}%")
        per_block = defaultdict(list)
        for r in gt:
            per_block[(r["trial"], r["block"], r["gt_x"], r["gt_y"], r["gt_color"], r["gt_size"])].append(r["found"])
        missed = [(k, np.mean(v)) for k, v in per_block.items() if np.mean(v) < 1.0]
        for (trial, _, x, y, c, s), rate in sorted(missed, key=lambda t: t[1]):
            say(f"  missed sometimes: {c} {s} at ({x:.0f}, {y:.0f}) in '{trial}': found {100 * rate:.0f}% of frames")
        if not missed:
            say("  Every block was found in every frame.")

        pos_rate = defaultdict(list)
        for (_, _, x, y, _, _), v in per_block.items():
            pos_rate[(x, y)].extend(v)
        P = np.array(list(pos_rate.keys()))
        R = np.array([np.mean(v) for v in pos_rate.values()])
        fig, ax = plt.subplots(figsize=(10, 7))
        draw_board(ax)
        sc = ax.scatter(P[:, 1], P[:, 0], c=R, cmap="RdYlGn", vmin=0, vmax=1, s=180, edgecolors="k", zorder=3)
        if fps:
            ax.scatter([float(f["det_y"]) for f in fps], [float(f["det_x"]) for f in fps],
                       marker="x", c="k", s=40, zorder=4, label="false positives")
        fig.colorbar(sc, ax=ax, label="fraction of frames detected")
        ax.set_title("Detection rate per position (x = false positive)")
        ax.legend(loc="lower left", fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / "detection_rate_map.png", dpi=150)
        plt.close(fig)

    n_frames = len(frames) if frames else len({(r["trial"], r["frame"]) for r in gt})
    say(f"False positives: {len(fps)} over {n_frames} frames "
        f"({len(fps) / max(n_frames, 1):.2f} per frame)")
    for trial, n in Counter(f["trial"] for f in fps).most_common():
        say(f"  {trial}: {n}")

    (out_dir / "summary.txt").write_text("\n".join(lines) + "\n")
    print(f"\nWrote plots and summary.txt to {out_dir}")


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Task 3.5 detector evaluation")
    ap.add_argument("--analyze", metavar="RUN_DIR", help="re-analyze an earlier run (no camera)")
    args = ap.parse_args()

    if args.analyze:
        analyze(args.analyze)
        return

    out_dir = OUT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving to {out_dir}")
    run_live(out_dir)
    if (out_dir / "gt_frames.csv").exists():
        analyze(out_dir)


if __name__ == "__main__":
    main()
