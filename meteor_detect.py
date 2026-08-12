#!/usr/bin/env python3
"""
meteor_detect.py — detect meteors in night-sky video.

Pipeline (validated on 97-min phone night-sky footage: 4 meteors confirmed,
zero false positives after filtering):
  1. Decode video to gray frames via ffmpeg (frame-accurate, any codec).
  2. Frame difference D = |I_t - I_{t-1}| — sensitive to fast moving streaks.
  3. Extract bright components (D > threshold), keep top-2 per frame.
  4. Greedy chain tracking: link components within <chain-radius> px across
     consecutive frames (1-frame gap tolerated).
  5. Chain filters (meteor = fast, straight, transient, bright):
       - line-fit residual <= max-residual px
       - net displacement >= min-displacement px
       - speed within [min-speed, max-speed] px/frame
       - component density >= min-density (no flickering)
       - peak brightness >= min-brightness (rejects diffuse ground light)
       - chain must END before the analysis window end (meteors fade out)
  6. Merge temporally-overlapping chains (one meteor = several fragments).
  7. Output JSON timeline + optional annotated keyframe images.

Usage:
  python3 meteor_detect.py input.mov -o events.json
  python3 meteor_detect.py input.mov --start 0 --end 600 --threshold 25 \
      --keyframes kf/ -o events.json

Dependencies: python3, numpy, opencv-python, ffmpeg (in PATH).

False-positive guide:
  - Airplane/satellite: long chains, slow (<2 px/f), never fade
    -> filtered by speed + fade-out requirements.
  - Ground light (cars etc.): huge diffuse components, low brightness
    -> filtered by threshold + peak-brightness.
  - Hot pixels: static, no displacement -> filtered by min-displacement.
  - Random noise streaks: never form a coherent moving chain
    -> chains are required by construction.
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np
import cv2

DEFAULT = dict(
    width=640,             # processing width (height = width * 9/16)
    threshold=30,          # frame-diff threshold (lower = more sensitive)
    min_area=2,            # min component area (px)
    chain_radius=50,       # max px between consecutive chain links
    min_chain=5,           # min chain length (frames)
    min_displacement=25,   # min net displacement (px)
    min_speed=2.5,         # min speed (px/frame) — excludes satellites/planes
    max_speed=60.0,        # max speed (px/frame) — excludes teleporting noise
    max_residual=8.0,      # max line-fit residual (px)
    min_density=0.6,       # min occupied-frame fraction
    min_brightness=80,     # min peak brightness (8-bit)
)


def probe(path):
    """Return (fps, duration_s) via ffprobe; fall back to (24, None)."""
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=r_frame_rate,duration",
             "-of", "json", path],
            capture_output=True, text=True, timeout=30).stdout
        info = json.loads(out)["streams"][0]
        num, den = info.get("r_frame_rate", "24/1").split("/")
        fps = float(num) / float(den) if float(den) else 24.0
        dur = float(info.get("duration", 0)) if info.get("duration") else None
        return fps, dur
    except Exception:
        return 24.0, None


def load_components(path, p, start_s=0.0, end_s=None):
    """Stream-decode [start_s, end_s) and return per-frame diff components.

    Returns (byframe: {frame_idx: [(frame_idx, cx, cy, area, maxval), ...]},
             first_frame_index, fps)
    """
    fps, dur = probe(path)
    if end_s is None and dur:
        end_s = dur
    cmd = ["ffmpeg", "-loglevel", "error"]
    if start_s > 0:
        cmd += ["-ss", f"{start_s:.3f}"]
    cmd += ["-i", path]
    if end_s is not None:
        cmd += ["-t", f"{end_s - start_s:.3f}"]
    W = int(p["width"])
    H = int(round(W * 9 / 16))
    cmd += ["-vf", f"scale={W}:{H},format=gray", "-an", "-f", "rawvideo",
            "pipe:1"]

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                            stderr=subprocess.DEVNULL)
    byframe = {}
    prev = None
    frame = 0
    n_bytes = W * H
    buf = bytearray()
    while True:
        chunk = proc.stdout.read(1 << 20)
        if not chunk:
            break
        buf += chunk
        while len(buf) >= n_bytes:
            raw = bytes(buf[:n_bytes])
            del buf[:n_bytes]
            g = np.frombuffer(raw, np.uint8).reshape(H, W)
            if prev is not None:
                D = np.abs(g.astype(np.int16) - prev.astype(np.int16))
                m = D > p["threshold"]
                if m.any():
                    nl, lab, stats, cent = cv2.connectedComponentsWithStats(
                        m.astype(np.uint8), 8)
                    comps = []
                    for j in range(1, nl):
                        a = int(stats[j, cv2.CC_STAT_AREA])
                        if a < p["min_area"]:
                            continue
                        cx, cy = cent[j]
                        maxval = float(D[lab == j].max())
                        comps.append((frame, float(cx), float(cy), a, maxval))
                    if comps:
                        comps.sort(key=lambda c: -c[4])
                        byframe[frame] = comps[:2]
            prev = g
            frame += 1
    proc.wait()
    first_frame = int(round(start_s * fps))
    return byframe, first_frame, fps, frame


def build_chains(byframe, p):
    """Greedy chain linking over time; returns list of chains (list of comps).
    Each comp = (frame_idx, cx, cy, area, maxval)."""
    radius = p["chain_radius"]
    allc = sorted([c for cl in byframe.values() for c in cl],
                  key=lambda c: -c[4])
    used = set()

    def extend(chain, direction):
        while True:
            last = chain[-1] if direction > 0 else chain[0]
            best = None
            for gap in (0, 1):  # tolerate 1-frame gap
                fi = last[0] + direction * (gap + 1)
                if fi not in byframe:
                    continue
                for c in byframe[fi]:
                    d = ((c[1] - last[1]) ** 2 + (c[2] - last[2]) ** 2) ** 0.5
                    if d < radius:
                        score = c[4] - 0.5 * d
                        if best is None or score > best[0]:
                            best = (score, c)
                if best is not None:
                    break
            if best is None:
                break
            if direction > 0:
                chain.append(best[1])
            else:
                chain.insert(0, best[1])

    chains = []
    for seed in allc:
        if id(seed) in used:
            continue
        ch = [seed]
        extend(ch, +1)
        extend(ch, -1)
        for c in ch:
            used.add(id(c))
        if len(ch) >= p["min_chain"]:
            chains.append(ch)
    return chains


def analyze_chain(ch, n_frames, p):
    """Return event dict or None if the chain fails meteor criteria."""
    n = len(ch)
    xs = [c[1] for c in ch]
    ys = [c[2] for c in ch]
    A = np.vstack([xs, np.ones(n)]).T
    k, b = np.linalg.lstsq(A, ys, rcond=None)[0]
    resid = float(np.sqrt(np.mean((np.array(ys) - (k * np.array(xs) + b)) ** 2)))
    disp = float(((xs[-1] - xs[0]) ** 2 + (ys[-1] - ys[0]) ** 2) ** 0.5)
    span = ch[-1][0] - ch[0][0] + 1
    speed = disp / max(span - 1, 1)
    density = n / span
    maxv = max(c[4] for c in ch)
    faded = ch[-1][0] < n_frames - 6
    if not (disp >= p["min_displacement"] and resid <= p["max_residual"]
            and p["min_speed"] <= speed <= p["max_speed"]
            and density >= p["min_density"] and maxv >= p["min_brightness"]
            and faded):
        return None
    return dict(n=n, xs=xs, ys=ys, resid=resid, disp=disp, speed=speed,
                maxv=maxv, frames=(ch[0][0], ch[-1][0]))


def merge_events(events, fps):
    """Merge temporally-overlapping events (same meteor split in fragments)."""
    if not events:
        return []
    events.sort(key=lambda e: e["frames"][0])
    merged = []
    for e in events:
        hit = None
        for m in merged:
            if (e["frames"][0] <= m["frames"][1] + 5
                    and e["frames"][1] >= m["frames"][0] - 5):
                hit = m
                break
        if hit is None:
            merged.append(e)
            continue
        hit["frames"] = (min(hit["frames"][0], e["frames"][0]),
                         max(hit["frames"][1], e["frames"][1]))
        if e["maxv"] > hit["maxv"]:
            hit.update(n=e["n"], xs=e["xs"], ys=e["ys"], resid=e["resid"],
                       disp=e["disp"], speed=e["speed"], maxv=e["maxv"])
    return merged


def save_keyframes(video, merged, first_frame, fps, out_dir, p):
    """Annotated max-projection images for each event (source resolution)."""
    os.makedirs(out_dir, exist_ok=True)
    W = int(p["width"])
    H = int(round(W * 9 / 16))
    for i, e in enumerate(merged):
        f0 = e["frames"][0] + first_frame
        f1 = e["frames"][1] + first_frame
        t0 = f0 / fps
        dur = (f1 - f0 + 2) / fps
        cmd = ["ffmpeg", "-loglevel", "error", "-ss", f"{t0:.3f}", "-i",
               video, "-t", f"{dur:.3f}", "-vf",
               f"scale={W}:{H},format=gray", "-an", "-f", "rawvideo", "pipe:1"]
        raw = subprocess.run(cmd, capture_output=True, timeout=60).stdout
        n = len(raw) // (W * H)
        if n == 0:
            continue
        fr = np.frombuffer(raw[:n * W * H], np.uint8).reshape(n, H, W)
        acc = fr.max(axis=0).astype(np.float32)
        lo, hi = np.percentile(acc, 1), np.percentile(acc, 99.9)
        img = np.clip((acc - lo) * 255 / max(hi - lo, 1), 0, 255).astype(np.uint8)
        p1 = (int(e["xs"][0]), int(e["ys"][0]))
        p2 = (int(e["xs"][-1]), int(e["ys"][-1]))
        cv2.line(img, p1, p2, 255, 2)
        cv2.circle(img, p1, 8, 255, 2)
        cv2.circle(img, p2, 8, 255, 2)
        cv2.putText(img, f"meteor {t0:.2f}s", (12, H - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, 255, 2)
        fn = os.path.join(out_dir, f"meteor_{i + 1:02d}_{t0:07.2f}s.jpg")
        cv2.imwrite(fn, img, [cv2.IMWRITE_JPEG_QUALITY, 90])


def fmt_ts(s):
    m = int(s // 60)
    return f"{m:02d}:{s - m * 60:05.2f}"


def main():
    ap = argparse.ArgumentParser(description="Detect meteors in night-sky video")
    ap.add_argument("video", help="input video path")
    ap.add_argument("-o", "--output", help="output JSON path (default: stdout)")
    ap.add_argument("--start", type=float, default=0.0, help="start time (s)")
    ap.add_argument("--end", type=float, default=None, help="end time (s)")
    ap.add_argument("--keyframes", help="dir for annotated keyframe images")
    for k, v in DEFAULT.items():
        ap.add_argument(f"--{k.replace('_', '-')}", type=type(v), default=v,
                        help=f"default {v}")
    args = ap.parse_args()
    p = {k: getattr(args, k) for k in DEFAULT}
    p["width"] = int(p["width"])

    byframe, first_frame, fps, total_frames = load_components(
        args.video, p, args.start, args.end)
    n_frames = total_frames
    print(f"[meteor-detect] {n_frames} frames analyzed "
          f"({first_frame / fps:.1f}s -> "
          f"{(first_frame + n_frames) / fps:.1f}s), "
          f"{len(byframe)} frames with activations", file=sys.stderr)

    chains = build_chains(byframe, p)
    print(f"[meteor-detect] {len(chains)} raw chains", file=sys.stderr)

    events = []
    for ch in chains:
        ev = analyze_chain(ch, n_frames, p)
        if ev:
            ev["abs_frame_start"] = first_frame + ev["frames"][0]
            ev["abs_frame_end"] = first_frame + ev["frames"][1]
            ev["t_start_abs"] = ev["abs_frame_start"] / fps
            ev["t_end_abs"] = ev["abs_frame_end"] / fps
            events.append(ev)

    merged = merge_events(events, fps)
    merged.sort(key=lambda e: e["t_start_abs"])

    if args.keyframes:
        save_keyframes(args.video, merged, first_frame, fps,
                       args.keyframes, p)

    for e in merged:
        e["time_start"] = fmt_ts(e["t_start_abs"])
        e["time_end"] = fmt_ts(e["t_end_abs"])
        e["duration_s"] = round(e["t_end_abs"] - e["t_start_abs"], 2)
        e["speed_px_per_frame"] = round(e["speed"], 2)
        e["max_brightness"] = round(e["maxv"], 1)
        e["residual_px"] = round(e["resid"], 2)
        e["n_points"] = e.pop("n", len(e["frames"]))
        e.pop("xs", None)
        e.pop("ys", None)
        e.pop("maxv", None)
        e.pop("resid", None)
        e.pop("speed", None)
        e.pop("disp", None)

    out = {
        "video": args.video,
        "fps": fps,
        "params": p,
        "n_events": len(merged),
        "events": merged,
    }

    if args.output:
        with open(args.output, "w") as f:
            json.dump(out, f, indent=1)
        print(f"[meteor-detect] wrote {len(merged)} events -> {args.output}",
              file=sys.stderr)
    else:
        print(json.dumps(out, indent=1))

    for e in merged:
        print(f"  meteor {e['time_start']} - {e['time_end']} "
              f"({e['duration_s']}s) n={e['n_points']} "
              f"speed={e['speed_px_per_frame']}px/f "
              f"brightness={e['max_brightness']} "
              f"resid={e['residual_px']}px", file=sys.stderr)


if __name__ == "__main__":
    main()
