#!/usr/bin/env python3
"""motion_audit.py - OFFLINE descriptive measurement + visual audit of stored keypoints.

Usage: python motion_audit.py --config PATH --out NEW_DIRECTORY

All values describe stored keypoint observations only. They are NOT ground truth,
do not establish movement direction/sign correctness, and make no classification
claims. No smoothing, interpolation, cycle counting or thresholds are applied.
Inputs (config, npz observations, source videos) are only opened for reading.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import re
import sys
from fractions import Fraction

import numpy as np

SCHEMA = "motion_audit/v1"
N_POINTS = 75
NOSE, SH_L, SH_R = 0, 11, 12
HAND_BASE = {"L": 33, "R": 54}
PALM_LOCAL = (0, 5, 9, 13, 17)  # wrist + 4 MCPs, hand-local indices; all 5 required
HAND_EDGES = ((0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
              (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15),
              (15, 16), (13, 17), (0, 17), (17, 18), (18, 19), (19, 20))
POSE_ARM_EDGES = ((11, 12), (11, 13), (13, 15), (12, 14), (14, 16))
POSE_ARM_POINTS = (11, 12, 13, 14, 15, 16)
DENSE_PAD = 5
CROP = (0.15, 0.05, 0.85, 0.95)  # fixed fractions x0, y0, x1, y1; never adaptive
TILE_W = 360
LABEL_H = 32
HEADER_H = 34
GAP = 4
TILES_PER_PAGE = 8  # 4 columns x 2 rows
PANELS_PER_PAGE = 16  # 4 columns x 4 rows
RGB = {"L": (0, 255, 255), "R": (255, 165, 0), "pose": (0, 200, 0)}
DISCLAIMER = ("Descriptive measurements of stored keypoints only; not ground truth, "
              "no sign/direction validation, no classification claims.")


# ----------------------------------------------------------------------------- JSON
def to_jsonable(obj):
    """Recursively convert numpy types; NaN/inf -> None. Result passes allow_nan=False."""
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return to_jsonable(obj.tolist())
    if isinstance(obj, np.generic):
        return to_jsonable(obj.item())
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, Fraction):
        return str(obj)
    if obj is None or isinstance(obj, (bool, int, str)):
        return obj
    raise TypeError(f"not JSON-serialisable: {type(obj)!r}")


def write_json_new(path, obj):
    with open(path, "x", encoding="utf-8") as f:  # 'x': never overwrite
        json.dump(to_jsonable(obj), f, allow_nan=False, ensure_ascii=True)


# ----------------------------------------------------------------------------- hashing
def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


# ----------------------------------------------------------------------------- measurement
def check_span(span):
    if not isinstance(span, (list, tuple)) or len(span) != 2:
        raise ValueError("active_span must be [lo, hi]")
    for v in span:
        if isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)):
            raise ValueError("active_span values must be integers")
    lo, hi = int(span[0]), int(span[1])
    if lo > hi:
        raise ValueError("active_span lo > hi")
    return lo, hi


def span_rows(T, span):
    """Inclusive span clipped to [0, T-1]; None if empty."""
    lo, hi = check_span(span)
    a, b = max(lo, 0), min(hi, T - 1)
    return (a, b) if a <= b else None


def dense_rows(T, span):
    lo, hi = check_span(span)
    return list(range(max(0, lo - DENSE_PAD), min(T - 1, hi + DENSE_PAD) + 1))


def validate_arrays(xy, usable, ts):
    """Return validated private copies (inputs are never modified)."""
    xy = np.array(xy, dtype=np.float64, copy=True)
    usable = np.array(usable, copy=True)
    ts = np.array(ts, copy=True)
    if xy.ndim != 3 or xy.shape[1:] != (N_POINTS, 2):
        raise ValueError(f"xy_px must be [T,{N_POINTS},2], got {xy.shape}")
    T = xy.shape[0]
    if usable.shape != (T, N_POINTS):
        raise ValueError(f"usable must be [T,{N_POINTS}], got {usable.shape}")
    if usable.dtype != np.bool_:
        if not (np.issubdtype(usable.dtype, np.integer) and np.isin(usable, (0, 1)).all()):
            raise ValueError("usable must be bool or 0/1 integers")
        usable = usable.astype(bool)
    if ts.shape != (T,) or not np.issubdtype(ts.dtype, np.integer):
        raise ValueError("timestamp_us must be integer [T]")
    ts = ts.astype(np.int64)
    if T > 1 and not np.all(np.diff(ts) > 0):
        raise ValueError("timestamp_us must be strictly increasing")
    return xy, usable, ts


def _side_summary(rel, rel_ok, palm_ok, ts, rows, scale_ok):
    out = {"scale_available": scale_ok, "span_frames": 0, "palm_all5_frames": 0,
           "valid_frames": 0, "adjacent_valid_pairs": 0, "x_range": None, "y_range": None,
           "covered_seconds": 0.0, "path_total": None,
           "mean_path_per_covered_second": None, "max_speed": None,
           "largest_invalid_run_frames": 0}
    if rows is None:
        return out
    a, b = rows
    out["span_frames"] = b - a + 1
    out["palm_all5_frames"] = int(palm_ok[a:b + 1].sum())
    if not scale_ok:  # relative measurements unavailable: no invented scale
        out.update(valid_frames=None, adjacent_valid_pairs=None, covered_seconds=None,
                   largest_invalid_run_frames=None)
        return out
    m, r, t = rel_ok[a:b + 1], rel[a:b + 1], ts[a:b + 1]
    out["valid_frames"] = int(m.sum())
    if m.any():
        out["x_range"] = [float(r[m, 0].min()), float(r[m, 0].max())]
        out["y_range"] = [float(r[m, 1].min()), float(r[m, 1].max())]
    run = best = 0
    for v in m:
        run = 0 if v else run + 1
        best = max(best, run)
    out["largest_invalid_run_frames"] = best
    pair = m[:-1] & m[1:]  # consecutive source rows, both endpoints valid; no bridging
    out["adjacent_valid_pairs"] = int(pair.sum())
    if pair.any():
        dt = np.diff(t)[pair] / 1e6
        dist = np.linalg.norm(np.diff(r, axis=0)[pair], axis=1)
        out["covered_seconds"] = float(dt.sum())
        out["path_total"] = float(dist.sum())
        out["mean_path_per_covered_second"] = out["path_total"] / out["covered_seconds"]
        out["max_speed"] = float((dist / dt).max())
    return out


def measure(xy, usable, ts, span):
    """Full-length JSON-safe series + per-side summary within the inclusive span.

    Relative units: shoulder widths (median shoulder distance, px). Speeds: widths/s.
    """
    xy, usable, ts = validate_arrays(xy, usable, ts)
    T = xy.shape[0]
    pv = usable & np.isfinite(xy).all(axis=-1)
    delta = np.full(T, np.nan)
    if T > 1:
        delta[1:] = np.diff(ts) / 1e6
    sh_ok = pv[:, SH_L] & pv[:, SH_R]
    sh_d = np.full(T, np.nan)
    if sh_ok.any():
        sh_d[sh_ok] = np.linalg.norm(xy[sh_ok, SH_L] - xy[sh_ok, SH_R], axis=1)
    good = sh_ok & np.isfinite(sh_d) & (sh_d > 0)
    scale = float(np.median(sh_d[good])) if good.any() else None
    nose_ok = pv[:, NOSE]
    rows = span_rows(T, span)
    series = {"time_s": ts / 1e6, "timestamp_us": ts, "delta_s": delta,
              "nose_valid": nose_ok, "shoulders_valid": sh_ok,
              "shoulder_distance_px": np.where(good, sh_d, np.nan)}
    summary = {}
    for side, base in HAND_BASE.items():
        idx = [base + k for k in PALM_LOCAL]
        palm_pts_ok = pv[:, idx]
        palm_ok = palm_pts_ok.all(axis=1)
        palm = np.full((T, 2), np.nan)
        if palm_ok.any():
            palm[palm_ok] = xy[palm_ok][:, idx].mean(axis=1)
        rel_ok = palm_ok & nose_ok & (scale is not None)
        rel = np.full((T, 2), np.nan)
        if rel_ok.any():
            rel[rel_ok] = (palm[rel_ok] - xy[rel_ok, NOSE]) / scale
        series[side] = {"palm_point_valid": palm_pts_ok, "palm_all5_valid": palm_ok,
                        "rel_valid": rel_ok, "palm_px_x": palm[:, 0], "palm_px_y": palm[:, 1],
                        "rel_x": rel[:, 0], "rel_y": rel[:, 1]}
        summary[side] = _side_summary(rel, rel_ok, palm_ok, ts, rows, scale is not None)
    return to_jsonable({
        "T": T, "span_requested": list(check_span(span)),
        "span_used": list(rows) if rows else None,
        "scale_px": scale, "scale_frames": int(good.sum()),
        "units": {"relative": "shoulder_widths (palm-nose)/scale; image y down",
                  "speed": "shoulder_widths_per_second", "time": "seconds = timestamp_us/1e6"},
        "series": series, "summary": summary})


# ----------------------------------------------------------------------------- inputs
def validate_config(cfg):
    if not isinstance(cfg, dict) or not isinstance(cfg.get("clips"), list) or not cfg["clips"]:
        raise ValueError("config must be {'clips': [...]} with at least one clip")
    hexre = re.compile(r"[0-9a-fA-F]{64}")
    ids, safes = set(), set()
    for c in cfg["clips"]:
        for k, typ in (("id", str), ("observation", str), ("video", str),
                       ("source_sha256", str), ("observation_sha256", str),
                       ("prediction", str), ("agreement", bool), ("dense", bool)):
            if not isinstance(c.get(k), typ):
                raise ValueError(f"clip field {k!r} missing or not {typ.__name__}")
        check_span(c.get("active_span"))
        for k in ("observation", "video"):
            if not os.path.isabs(c[k]):
                raise ValueError(f"{c['id']}: {k} must be an absolute path")
        for k in ("source_sha256", "observation_sha256"):
            if not hexre.fullmatch(c[k]):
                raise ValueError(f"{c['id']}: {k} is not a sha256 hex digest")
        safe = safe_name(c["id"])
        if not c["id"] or c["id"] in ids or safe.lower() in safes:
            raise ValueError(f"empty or duplicate clip id: {c['id']!r}")
        ids.add(c["id"])
        safes.add(safe.lower())
    return cfg["clips"]


def safe_name(s):
    return re.sub(r"[^A-Za-z0-9._-]", "_", s)[:80]


def load_observation(data):
    """Load npz from already-hashed bytes (allow_pickle=False)."""
    z = np.load(io.BytesIO(data), allow_pickle=False)
    if not isinstance(z, np.lib.npyio.NpzFile):
        raise ValueError("observation is not an npz archive")
    with z:
        for k in ("xy_px", "usable", "timestamp_us"):
            if k not in z.files:
                raise ValueError(f"observation missing field {k}")
        xy, usable, ts = validate_arrays(z["xy_px"], z["usable"], z["timestamp_us"])
        fid = np.array(z["frame_id"]) if "frame_id" in z.files else None
    if fid is not None:
        if fid.shape != ts.shape or not np.issubdtype(fid.dtype, np.integer) or \
                (fid.size and (fid.min() < 0 or np.any(np.diff(fid) <= 0))):
            raise ValueError("frame_id must be non-negative, strictly increasing integers [T]")
        fid = fid.astype(np.int64)
    return {"xy": xy, "usable": usable, "ts": ts, "frame_id": fid}


# ----------------------------------------------------------------------------- plots
def _arr(values):
    return np.array([np.nan if v is None else v for v in values], dtype=np.float64)


TRACE_STYLE = {("L", "x"): dict(color="c", ls="-", marker="."),
               ("L", "y"): dict(color="c", ls="--", marker="x"),
               ("R", "x"): dict(color="darkorange", ls="-.", marker="."),
               ("R", "y"): dict(color="darkorange", ls=":", marker="+")}


def draw_traces(ax, rec, fs):
    res = rec["measure"]
    s = res["series"]
    t = _arr(s["time_s"])
    for (side, comp), style in TRACE_STYLE.items():  # NaN breaks lines at gaps
        ax.plot(t, _arr(s[side][f"rel_{comp}"]), lw=1, ms=3, label=f"{side} {comp}", **style)
    if res["span_used"]:
        a, b = res["span_used"]
        ax.axvspan(t[a], t[b], color="0.85", zorder=0)
        if a == b:
            ax.axvline(t[a], color="0.6", zorder=0)
    if res["T"] > 1:
        ax.set_xlim(t[0], t[-1])
    T = res["T"]
    cov = {k: sum(bool(v) for v in s[k]["palm_all5_valid"]) for k in HAND_BASE}
    status = "MATCH" if rec["agreement"] else "MISMATCH"
    ax.set_title(f"{rec['id']}\nstored {status} pred={rec['prediction']}\n"
                 f"hand cov L {cov['L']}/{T} R {cov['R']}/{T}", fontsize=fs)
    if res["scale_px"] is None:
        ax.text(0.5, 0.5, "NO SHOULDER SCALE", transform=ax.transAxes, ha="center", fontsize=fs)
    ax.set_xlabel("time [s]", fontsize=fs)
    ax.set_ylabel("(palm-nose) [shoulder widths]", fontsize=fs)
    ax.tick_params(labelsize=fs)


def make_plots(records, plot_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    files = []
    for p0 in range(0, len(records), PANELS_PER_PAGE):
        chunk = records[p0:p0 + PANELS_PER_PAGE]
        nrows = max(1, math.ceil(len(chunk) / 4))
        fig, axes = plt.subplots(nrows, 4, figsize=(17, 3.6 * nrows), squeeze=False)
        for ax in axes.flat[len(chunk):]:
            ax.axis("off")
        for ax, rec in zip(axes.flat, chunk):
            draw_traces(ax, rec, 6)
        axes.flat[0].legend(fontsize=6, loc="upper right")
        fig.text(0.01, 0.005, "Shaded = stored active span. " + DISCLAIMER, fontsize=7)
        fig.tight_layout(rect=(0, 0.02, 1, 1))
        fn = os.path.join(plot_dir, f"overview_{p0 // PANELS_PER_PAGE + 1:02d}.png")
        fig.savefig(fn, dpi=110)
        plt.close(fig)
        files.append(fn)
    for rec in records:
        fig, ax = plt.subplots(figsize=(12, 5.5))
        draw_traces(ax, rec, 9)
        ax.legend(fontsize=8)
        sm = rec["measure"]["summary"]
        lines = [f"{k}: frames {v['valid_frames']} pairs {v['adjacent_valid_pairs']} "
                 f"covered_s {v['covered_seconds']} path {v['path_total']} "
                 f"mean {v['mean_path_per_covered_second']} max {v['max_speed']} "
                 f"max_gap {v['largest_invalid_run_frames']}" for k, v in sm.items()]
        fig.text(0.01, 0.01, "\n".join(lines + [DISCLAIMER]), fontsize=7, family="monospace")
        fig.tight_layout(rect=(0, 0.1, 1, 1))
        fn = os.path.join(plot_dir, f"clip_{safe_name(rec['id'])}.png")
        fig.savefig(fn, dpi=110)
        plt.close(fig)
        files.append(fn)
    return files


# ----------------------------------------------------------------------------- dense render
def pts_to_us(pts, time_base):
    """Exact rational PTS -> integer microseconds (Python round: half-to-even)."""
    return round(Fraction(pts) * Fraction(time_base) * 1_000_000)


def iter_decoded(video_path):
    """Yield (ordinal, rel_us, frame, time_base, abs_us). Frames without PTS or with
    non-increasing rounded-us time are ignored; origin = first kept PTS."""
    import av
    with av.open(video_path, mode="r") as c:
        st = c.streams.video[0]
        tb = st.time_base
        if tb is None:
            raise RuntimeError("video stream has no time_base")
        last = origin = None
        ordinal = -1
        for fr in c.decode(st):
            if fr.pts is None:
                continue
            abs_us = pts_to_us(fr.pts, tb)
            if last is not None and abs_us <= last:
                continue
            last = abs_us
            if origin is None:
                origin = abs_us
            ordinal += 1
            yield ordinal, abs_us - origin, fr, tb, abs_us


def iter_rows(video_path, rows, ts, fid):
    """Yield exactly the requested rows in order; raise on any mismatch/missing frame."""
    k = 0
    gen = iter_decoded(video_path)
    try:
        for ordinal, rel, fr, tb, abs_us in gen:
            if k == len(rows):
                break
            r = rows[k]
            key, want = (ordinal, int(fid[r])) if fid is not None else (rel, int(ts[r]))
            if key < want:
                continue
            if key > want:
                raise RuntimeError(f"row {r}: no decoded frame with "
                                   f"{'ordinal' if fid is not None else 'rel_us'} {want}")
            if rel != int(ts[r]):
                raise RuntimeError(f"row {r}: decoded rel_us {rel} != saved ts {int(ts[r])}")
            yield r, fr.to_image(), {"decoded_ordinal": ordinal, "pts": int(fr.pts),
                                     "time_base": str(tb), "pts_abs_us": abs_us, "rel_us": rel}
            k += 1
    finally:
        gen.close()
    if k < len(rows):
        raise RuntimeError(f"stream ended before row {rows[k]}")


def crop_box(w, h):
    return (round(CROP[0] * w), round(CROP[1] * h), round(CROP[2] * w), round(CROP[3] * h))


def make_tile(img, xy_row, pv_row, label_lines, font):
    """Unmarked fixed crop on top, identical crop with skeleton overlay below."""
    from PIL import Image, ImageDraw
    img = img.convert("RGB")
    box = crop_box(*img.size)
    cw, ch = box[2] - box[0], box[3] - box[1]
    th = max(1, round(ch * TILE_W / cw))
    base = img.crop(box).resize((TILE_W, th), Image.BILINEAR)
    sx, sy = TILE_W / cw, th / ch
    over = base.copy()
    d = ImageDraw.Draw(over)

    def P(i):
        return ((xy_row[i, 0] - box[0]) * sx, (xy_row[i, 1] - box[1]) * sy)

    def dot(i, col):
        x, y = P(i)
        d.ellipse([x - 2, y - 2, x + 2, y + 2], fill=col)

    for a, b in POSE_ARM_EDGES:
        if pv_row[a] and pv_row[b]:
            d.line([P(a), P(b)], fill=RGB["pose"], width=2)
    for i in POSE_ARM_POINTS:
        if pv_row[i]:
            dot(i, RGB["pose"])
    for side, hb in HAND_BASE.items():
        for a, b in HAND_EDGES:
            if pv_row[hb + a] and pv_row[hb + b]:
                d.line([P(hb + a), P(hb + b)], fill=RGB[side], width=2)
        for k in range(21):
            if pv_row[hb + k]:
                dot(hb + k, RGB[side])
    tile = Image.new("RGB", (TILE_W, 2 * th + LABEL_H), (20, 20, 20))
    tile.paste(base, (0, 0))
    tile.paste(over, (0, th))
    dt = ImageDraw.Draw(tile)
    for j, line in enumerate(label_lines):
        dt.text((4, 2 * th + 3 + 14 * j), line, fill=(255, 255, 255), font=font)
    return tile, box


def save_page(tiles, path, header_lines, font):
    from PIL import Image, ImageDraw
    cell_h = max((t.size[1] for t in tiles), default=200)
    W = 4 * TILE_W + 5 * GAP
    H = HEADER_H + 2 * cell_h + 3 * GAP
    page = Image.new("RGB", (W, H), (60, 60, 60))
    d = ImageDraw.Draw(page)
    for j, line in enumerate(header_lines):
        d.text((GAP, 3 + 14 * j), line, fill=(255, 255, 255), font=font)
    for i in range(TILES_PER_PAGE):
        x = GAP + (i % 4) * (TILE_W + GAP)
        y = HEADER_H + GAP + (i // 4) * (cell_h + GAP)
        if i < len(tiles):
            page.paste(tiles[i], (x, y))
        else:
            d.rectangle([x, y, x + TILE_W - 1, y + cell_h - 1], fill=(35, 35, 35))
            d.text((x + TILE_W // 2 - 15, y + cell_h // 2), "EMPTY", fill=(200, 200, 200), font=font)
    with open(path, "xb") as f:
        page.save(f, format="PNG")


def render_dense(clip, obs, out_dir):
    from PIL import ImageFont
    font = ImageFont.load_default()
    ts, fid, xy, usable = obs["ts"], obs["frame_id"], obs["xy"], obs["usable"]
    pv = usable & np.isfinite(xy).all(axis=-1)
    lo, hi = check_span(clip["active_span"])
    rows = dense_rows(len(ts), clip["active_span"])
    entry = {"id": clip["id"], "dense": True, "status": "rendered", "rows": rows,
             "crop_fractions": CROP, "tile_width": TILE_W, "pages": [],
             "note": "All-frames view of selected interval only; not expert review."}
    if not rows:
        entry["status"] = "no_rows"
        return entry
    n_pages = math.ceil(len(rows) / TILES_PER_PAGE)
    tiles, recs = [], []

    def flush():
        p = len(entry["pages"]) + 1
        fn = os.path.join(out_dir, f"page_{p:03d}.png")
        hdr = [f"{clip['id']} | rows {rows[0]}-{rows[-1]} | active {lo}-{hi} | page {p}/{n_pages}",
               "top: unmarked crop, bottom: same crop + overlay (L cyan, R orange, pose green). "
               "All-frames view, not expert review."]
        save_page(tiles, fn, hdr, font)
        recs.extend({"tile": i, "row": None, "label": "EMPTY"}
                    for i in range(len(recs), TILES_PER_PAGE))
        entry["pages"].append({"file": os.path.relpath(fn, os.path.dirname(os.path.dirname(out_dir))),
                               "tiles": list(recs)})
        tiles.clear()
        recs.clear()

    for r, img, info in iter_rows(clip["video"], rows, ts, fid):
        status = "IN" if lo <= r <= hi else "OUT"
        hands = " ".join(f"{s}:{'all5' if pv[r, [hb + k for k in PALM_LOCAL]].all() else '--'}"
                         for s, hb in HAND_BASE.items())
        label = [f"row {r}  t={ts[r] / 1e6:.6f}s  {status}",
                 f"ord {info['decoded_ordinal']} pts {info['pts']} {hands}"]
        tile, box = make_tile(img, xy[r], pv[r], label, font)
        tiles.append(tile)
        recs.append({"tile": len(recs), "row": r, "active": status, "saved_ts_us": int(ts[r]),
                     "frame_id": None if fid is None else int(fid[r]),
                     "frame_size": list(img.size), "crop_box_px": list(box), **info})
        if len(tiles) == TILES_PER_PAGE:
            flush()
    if tiles:
        flush()
    return entry


# ----------------------------------------------------------------------------- main
def input_hashes(clips):
    out = {}
    for c in clips:
        h = {}
        for k in ("observation", "video"):
            try:
                h[k] = sha256_file(c[k])
            except OSError as e:
                h[k] = None
                h[k + "_error"] = repr(e)
        out[c["id"]] = h
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True, help="new directory; must not exist")
    args = ap.parse_args(argv)
    out = os.path.abspath(args.out)
    if os.path.lexists(out):
        raise SystemExit(f"refusing to use existing output path: {out}")
    with open(args.config, "rb") as f:
        cfg_bytes = f.read()
    cfg_sha = sha256_bytes(cfg_bytes)
    clips = validate_config(json.loads(cfg_bytes.decode("utf-8")))

    before, obs = {}, {}
    for c in clips:  # verify hashes before any work; load npz from hashed bytes
        with open(c["observation"], "rb") as f:
            data = f.read()
        before[c["id"]] = {"observation": sha256_bytes(data), "video": sha256_file(c["video"])}
        if before[c["id"]]["observation"] != c["observation_sha256"].lower():
            raise SystemExit(f"{c['id']}: observation sha256 mismatch")
        if before[c["id"]]["video"] != c["source_sha256"].lower():
            raise SystemExit(f"{c['id']}: source video sha256 mismatch")
        obs[c["id"]] = load_observation(data)

    records = []
    for c in clips:
        o = obs[c["id"]]
        res = measure(o["xy"], o["usable"], o["ts"], c["active_span"])
        res["series"]["frame_id"] = None if o["frame_id"] is None else o["frame_id"].tolist()
        records.append({k: c[k] for k in ("id", "prediction", "agreement", "dense",
                                          "active_span", "observation", "video")}
                       | {"has_frame_id": o["frame_id"] is not None, "measure": res})

    os.makedirs(out, exist_ok=False)  # created exactly once
    plot_dir, dense_dir = os.path.join(out, "plots"), os.path.join(out, "dense")
    os.mkdir(plot_dir)
    os.mkdir(dense_dir)
    plot_files = make_plots(records, plot_dir)

    render, dense_fail = [], 0
    for c in clips:
        if not c["dense"]:
            render.append({"id": c["id"], "dense": False, "status": "skipped_not_dense"})
            continue
        d = os.path.join(dense_dir, safe_name(c["id"]))
        os.mkdir(d)
        try:
            render.append(render_dense(c, obs[c["id"]], d))
        except Exception as e:  # recorded, reported, non-zero exit; never substituted
            dense_fail += 1
            render.append({"id": c["id"], "dense": True, "status": "error", "error": repr(e)})
            print(f"DENSE ERROR {c['id']}: {e!r}", file=sys.stderr)

    after = input_hashes(clips)
    with open(args.config, "rb") as f:
        cfg_sha_after = sha256_bytes(f.read())
    hash_ok = after == before and cfg_sha_after == cfg_sha
    write_json_new(os.path.join(out, "render_index.json"),
                   {"schema": SCHEMA + "/render_index", "config_sha256": cfg_sha, "clips": render})
    write_json_new(os.path.join(out, "measurements.json"), {
        "schema": SCHEMA, "config_path": os.path.abspath(args.config),
        "config_sha256": cfg_sha, "config_sha256_after": cfg_sha_after,
        "input_sha256_before": before, "input_sha256_after": after, "input_hashes_unchanged": hash_ok,
        "palm_local_indices": PALM_LOCAL, "disclaimer": DISCLAIMER,
        "plots": [os.path.relpath(p, out) for p in plot_files], "clips": records})
    if not hash_ok:
        print("ERROR: input hashes changed during run", file=sys.stderr)
        return 2
    return 3 if dense_fail else 0


if __name__ == "__main__":
    sys.exit(main())
