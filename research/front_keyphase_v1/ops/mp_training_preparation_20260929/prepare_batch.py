#!/usr/bin/env python3
"""Batch preparation wrapper: raw clips -> observations -> per-sample feature inputs.

Evidence produced here is numeric time-grid compatibility only. Semantic action
boundaries and source-keypoint pixel correspondence are NOT verified.
Usage: python prepare_batch.py --config config.json
"""
import argparse
import hashlib
import json
import math
import re
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

SAFE_ID = re.compile(r"[A-Za-z0-9_]+")
RESERVED = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}
FPS, GRID_TOL_US = 30, 2
TIMING_OK = "NUMERIC_GRID_COMPATIBLE_SEMANTIC_ALIGNMENT_UNVERIFIED"
ROLE = "existing_train_preparation_only"
SHOULDERS = [11, 12]  # assumption: 75 = 33 pose + 21 left + 21 right (MediaPipe order); used only for a reported rate
REQUIRED = ("sample_id", "clip_id", "label", "signer_id", "start_s", "end_s",
            "provided_frame_interval", "native_npz_sha256", "raw_path", "raw_sha256")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def check_pins(pins):
    bad = []
    for p, want in pins.items():
        try:
            if not Path(p).is_absolute():
                raise ValueError("pin path is not absolute")
            got = sha256(p)
            if got != str(want).lower():
                bad.append(f"{p}: sha256 {got} != pinned {want}")
        except Exception as e:
            bad.append(f"{p}: {e!r}")
    return bad


def dump_new(path, obj):  # exclusive create: never overwrites existing evidence
    with open(path, "x", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=str)


def get(o, k):
    return o[k] if isinstance(o, dict) else getattr(o, k)


def num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def sidecars(npz):
    return [c for c in (npz.with_suffix(".sidecar.json"), npz.with_name(npz.name + ".sidecar.json")) if c.is_file()]


def preflight(rows):
    errs, samples, clips = [], set(), {}
    if not isinstance(rows, list) or not rows:
        return ["rows must be a non-empty list"]
    for i, r in enumerate(rows):
        missing = [k for k in REQUIRED if k not in r]
        if missing:
            errs.append(f"row {i}: missing {missing}")
            continue
        for k in ("sample_id", "clip_id"):
            v = r[k]
            if not isinstance(v, str) or not SAFE_ID.fullmatch(v) or v.upper() in RESERVED:
                errs.append(f"row {i}: unsafe {k} {v!r}")
        if str(r["sample_id"]).lower() in samples:  # case-insensitive: Windows filesystems collide
            errs.append(f"row {i}: duplicate sample_id {r['sample_id']!r}")
        samples.add(str(r["sample_id"]).lower())
        if clips.setdefault(str(r["clip_id"]).lower(), r["clip_id"]) != r["clip_id"]:
            errs.append(f"row {i}: clip_id {r['clip_id']!r} differs only by case from another clip")
    return errs


def probe_timing(av, raw, meta):
    """Independent decode-time scan (no pixel conversion). Fills meta; returns relative microseconds."""
    with av.open(str(raw)) as c:
        s = c.streams.video[0]
        tb, pts, missing = s.time_base, [], 0
        meta.update(average_rate=str(s.average_rate), time_base=str(tb))
        for fr in c.decode(s):
            if fr.pts is None:
                missing += 1
            else:
                pts.append(fr.pts)
    meta.update(decoded_frames=len(pts) + missing, missing_pts=missing,
                first_pts_s=float(pts[0] * tb) if pts and tb else None)
    if tb is None or missing or not pts:
        raise ValueError(f"timestamps missing or absent: {meta}")
    if any(b <= a for a, b in zip(pts, pts[1:])):
        raise ValueError("decoded PTS not strictly increasing")
    if abs(pts[0] * tb) > 1e-6:
        raise ValueError(f"first decoded PTS is not zero: {meta['first_pts_s']}")
    rel_us = np.array([round((p - pts[0]) * tb * 1_000_000) for p in pts], np.int64)  # exact Fraction math
    grid = np.round(np.arange(len(pts)) * 1e6 / FPS).astype(np.int64)
    meta["max_grid_err_us"] = int(np.abs(rel_us - grid).max())
    if meta["max_grid_err_us"] > GRID_TOL_US:
        raise ValueError(f"not on {FPS} fps grid (no shift/offset guessed): {meta}")
    return rel_us


def check_obs(obs, obs_sha, raw_sha, expected_id, rel_us):
    obs.validate()
    sc = obs.sidecar() if callable(obs.sidecar) else obs.sidecar
    qs = obs.quality_summary()
    if sc.get("source_sha256") != raw_sha:
        raise ValueError(f"sidecar source_sha256 {sc.get('source_sha256')} != raw {raw_sha}")
    if sc.get("extractor_id") != expected_id:
        raise ValueError(f"sidecar extractor_id {sc.get('extractor_id')!r} != expected {expected_id!r}")
    if sc.get("observation_sha256") not in (None, obs_sha):
        raise ValueError("sidecar observation_sha256 does not match observation file")
    dropped = sc.get("dropped_non_monotonic_pts", qs.get("dropped_non_monotonic_pts") if isinstance(qs, dict) else None)
    if dropped != 0:
        raise ValueError(f"dropped_non_monotonic_pts must be 0, got {dropped!r}")
    T = len(rel_us)
    xy, us = np.asarray(obs.xy_px), np.asarray(obs.usable)
    ts, fid = np.asarray(obs.timestamp_us), np.asarray(obs.frame_id)
    if obs.n_frames != T or xy.shape != (T, 75, 2) or us.shape != (T, 75) or us.dtype != bool or ts.shape != (T,):
        raise ValueError(f"observation shape/frame count mismatch vs {T} decoded frames")
    if not np.isfinite(xy[us]).all():
        raise ValueError("non-finite xy at usable landmarks")
    if T > 1 and np.any(np.diff(ts) <= 0):
        raise ValueError("observation timestamps not strictly increasing")
    grid = np.round(np.arange(T) * 1e6 / FPS).astype(np.int64)
    if np.abs(ts - rel_us).max() > 1 or np.abs(ts - grid).max() > GRID_TOL_US:  # 1 us: rounding only
        raise ValueError("observation timestamps differ from independent decode timestamps")
    if not np.array_equal(fid, np.arange(T)):
        raise ValueError("frame_id != arange(T)")
    return qs


def base(r):
    return {"sample_id": r["sample_id"], "clip_id": r["clip_id"], "label": r["label"], "signer_id": r["signer_id"],
            "source": r, "status": "FAILED", "exception": None, "feature_path": None, "feature_sha256": None,
            "observation_path": None, "observation_sha256": None, "selected_first": None, "selected_last": None,
            "selected_count": 0, "timing_status": "NOT_ESTABLISHED", "usable_hand_rate": None,
            "shoulder_rate": None, "role": ROLE, "semantic_alignment_verified": False}


def prepare_row(r, res, obs, rel_us, C, files):
    a, b = r["provided_frame_interval"]
    if not all(isinstance(v, int) and not isinstance(v, bool) for v in (a, b)) or not 0 <= a <= b:
        raise ValueError(f"bad provided_frame_interval {r['provided_frame_interval']!r}")
    s, e = r["start_s"], r["end_s"]
    rel_s = rel_us / 1e6
    if not (num(s) and num(e) and 0 <= s < e <= rel_s[-1] + 1 / FPS + 1e-9):
        raise ValueError(f"bounds [{s}, {e}) outside video (last {rel_s[-1]} s + 1/{FPS})")
    sel = np.nonzero((rel_s >= s) & (rel_s < e))[0]
    res.update(selected_count=int(len(sel)))
    if len(sel) < 4:
        raise ValueError(f"only {len(sel)} frames selected (<4)")
    res.update(selected_first=int(sel[0]), selected_last=int(sel[-1]))
    if (sel[0], sel[-1]) != (a, b):
        raise ValueError(f"selected [{sel[0]}, {sel[-1]}] != provided [{a}, {b}]")
    ts, fid, us = np.asarray(obs.timestamp_us)[sel], np.asarray(obs.frame_id)[sel], np.asarray(obs.usable)[sel]
    xy, present = C["to_native_like"](np.asarray(obs.xy_px)[sel], us)
    feat = C["build_channels"](xy, present, ts)
    x, bv, hv = (np.asarray(get(feat, k)) for k in ("x", "body_valid", "hand_valid"))
    body_scale, hs = float(get(feat, "body_scale")), get(feat, "hand_scales")
    hand_scales = np.array([float(hs["left"]), float(hs["right"])], np.float32)  # NaN allowed for absent hand
    n = len(sel)
    if x.shape != (n, 280) or x.dtype != np.float32 or bv.shape != (n, 51) or hv.shape != (n, 42) \
            or int(get(feat, "frames")) != n:
        raise ValueError(f"feature shapes {x.shape}/{x.dtype}/{bv.shape}/{hv.shape} unexpected for {n} frames")
    if not (np.isfinite(x).all() and np.isfinite(bv.astype(np.float64)).all()
            and np.isfinite(hv.astype(np.float64)).all() and math.isfinite(body_scale)):
        raise ValueError("non-finite features, masks or body_scale")
    res.update(usable_hand_rate=float(hv.astype(bool).any(1).mean()),
               shoulder_rate=float(us[:, SHOULDERS].all(1).mean()))
    if not hv.astype(bool).any():
        raise ValueError("no usable hand in window")
    fp = C["out"] / "inputs" / f"{r['sample_id']}.npz"
    with open(fp, "xb") as f:
        np.savez(f, x=x, body_valid=bv, hand_valid=hv, timestamp_us=ts, frame_id=fid,
                 body_scale=np.asarray(body_scale), hand_scales=hand_scales)
    files[str(fp)] = sha = sha256(fp)
    res.update(status="OK", feature_path=str(fp), feature_sha256=sha)


def process_clip(cid, rows, C):
    out, r0 = C["out"], rows[0]
    rec = {"clip_id": cid, "config_hash": C["hash"], "status": "FAILED", "timing": {}, "files": {},
           "exception": None, "semantic_alignment_verified": False}
    results = rec["rows"] = [base(r) for r in rows]
    try:
        for k in ("raw_path", "raw_sha256", "signer_id", "observation_path", "observation_sha256"):
            if len({json.dumps(r.get(k), ensure_ascii=False) for r in rows}) != 1:
                raise ValueError(f"rows of clip disagree on {k}")
        reused = bool(r0.get("observation_path"))
        if reused and not r0.get("observation_sha256"):
            raise ValueError("observation_path given without observation_sha256")
        npz = Path(r0["observation_path"]) if reused else out / "observations" / f"{cid}.npz"
        allowed = {npz.resolve(), *(p.resolve() for p in sidecars(npz))} if reused else set()
        partial = [str(p) for p in (out / "observations").glob(f"{cid}.*") if p.resolve() not in allowed]
        partial += [str(p) for r in rows if (p := out / "inputs" / f"{r['sample_id']}.npz").exists()]
        if partial:
            raise FileExistsError(f"partial files without receipt; refusing clip: {partial}")
        raw = Path(r0["raw_path"])
        rec["raw_path"] = str(raw)
        rec["files"][str(raw)] = raw_sha = rec["raw_sha256"] = sha256(raw)
        if raw_sha != str(r0["raw_sha256"]).lower():
            raise ValueError(f"raw sha256 {raw_sha} != config {r0['raw_sha256']}")
        rel_us = probe_timing(C["av"], raw, rec["timing"])
        if not reused:
            rec["files"][C["contract_path"]] = C["contract_sha"]
            with C["Landmarker"](json.loads(C["contract_text"])) as lm:  # new Landmarker per clip
                if lm.extractor_id() != C["expected"]:
                    raise ValueError(f"extractor_id {lm.extractor_id()!r} != expected {C['expected']!r}")
                obs = C["extract"](str(raw), lm, cid)
            obs.save(str(npz))
        obs_sha = sha256(npz)
        if reused and obs_sha != str(r0["observation_sha256"]).lower():
            raise ValueError(f"observation sha256 {obs_sha} != config {r0['observation_sha256']}")
        rec["files"][str(npz)] = obs_sha
        for sc in sidecars(npz):
            rec["files"][str(sc)] = sha256(sc)
        obs = C["Observation"].load(str(npz))  # always use what is on disk
        qs = check_obs(obs, obs_sha, raw_sha, C["expected"], rel_us)
        rec["observation"] = {"path": str(npz), "sha256": obs_sha, "reused": reused, "quality_summary": qs}
    except Exception:
        rec["exception"] = tb = traceback.format_exc()
        for res in results:
            res.update(status="FAILED", exception=tb)
        return rec
    for r, res in zip(rows, results):
        res.update(observation_path=str(npz), observation_sha256=obs_sha, timing_status=TIMING_OK)
        try:
            prepare_row(r, res, obs, rel_us, C, rec["files"])
        except Exception:
            res.update(status="FAILED", exception=traceback.format_exc())
    rec["status"] = "OK" if all(res["status"] == "OK" for res in results) else "FAILED"
    return rec


def resume(rp, rows, cfg_hash):
    try:
        rec = json.loads(rp.read_text(encoding="utf-8"))
        if rec.get("config_hash") != cfg_hash:
            raise ValueError("existing receipt has a different config_hash; refusing clip")
        saved = rec["rows"]
        if [s["sample_id"] for s in saved] != [r["sample_id"] for r in rows]:
            raise ValueError("receipt rows do not match config rows")
        bad = [p for p, h in rec["files"].items() if not Path(p).is_file() or sha256(p) != h]
        if bad:
            raise ValueError(f"saved file hashes invalid: {bad}")
        for s in saved:
            s["resume_note"] = "resumed from existing receipt; recorded failures retained, not retried"
        return saved, rec["status"]
    except Exception:
        results = [base(r) for r in rows]
        for res in results:
            res["exception"] = traceback.format_exc()
        return results, "FAILED"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", required=True)
    t0, started = time.time(), datetime.now(timezone.utc).isoformat()
    cfg = json.loads(Path(ap.parse_args().config).read_text(encoding="utf-8"))
    cfg_hash = hashlib.sha256(json.dumps(cfg, sort_keys=True, ensure_ascii=False,
                                         separators=(",", ":")).encode("utf-8")).hexdigest()
    out = Path(cfg["output_root"])
    if not out.is_absolute():
        sys.exit("output_root must be absolute")
    errs = preflight(cfg.get("rows"))
    if errs:
        sys.exit("config rejected:\n" + "\n".join(errs))
    bad = check_pins(cfg.get("pins", {}))
    if bad:
        sys.exit("pin mismatch before run:\n" + "\n".join(bad))
    for p in reversed(cfg.get("python_paths", [])):
        sys.path.insert(0, str(p))
    import av
    from wis_lab.capture.landmarker import Landmarker
    from wis_lab.capture.video_source import extract
    from wis_lab.core.observation import Observation
    from pilot.web_input import to_native_like
    from pilot.data import build_channels
    contract_text = Path(cfg["contract_path"]).read_text(encoding="utf-8")
    json.loads(contract_text)  # fail early on malformed contract
    C = dict(out=out, hash=cfg_hash, expected=cfg["expected_extractor_id"], av=av, Landmarker=Landmarker,
             extract=extract, Observation=Observation, to_native_like=to_native_like, build_channels=build_channels,
             contract_text=contract_text, contract_path=str(cfg["contract_path"]),
             contract_sha=sha256(cfg["contract_path"]))
    for d in ("observations", "inputs", "receipts"):
        (out / d).mkdir(parents=True, exist_ok=True)
    groups = {}
    for r in cfg["rows"]:
        groups.setdefault(r["clip_id"], []).append(r)
    by_id, clip_status = {}, {}
    for i, (cid, rows) in enumerate(groups.items(), 1):
        rp = out / "receipts" / f"{cid}.json"
        if rp.exists():
            results, status, how = *resume(rp, rows, cfg_hash), "resumed"
        else:
            rec, how = process_clip(cid, rows, C), "processed"
            results, status = rec["rows"], rec["status"]
            try:
                dump_new(rp, rec)
            except Exception:
                status, tb = "FAILED", traceback.format_exc()
                for res in results:
                    res.update(status="FAILED", exception=(res["exception"] or "") + "\nreceipt write failed:\n" + tb)
        clip_status[cid] = status
        by_id.update((res["sample_id"], res) for res in results)
        n_ok = sum(res["status"] == "OK" for res in results)
        print(f"[{i}/{len(groups)}] {cid}: {how} {status} ({n_ok}/{len(rows)} rows OK)", flush=True)
    pins_after = check_pins(cfg.get("pins", {}))
    ordered = [by_id[r["sample_id"]] for r in cfg["rows"]]  # denominator = every config row
    labels = {}
    for res in ordered:
        c = labels.setdefault(str(res["label"]), {"total": 0, "OK": 0, "FAILED": 0})
        c["total"] += 1
        c[res["status"]] = c.get(res["status"], 0) + 1
    failures = [{"sample_id": r["sample_id"], "clip_id": r["clip_id"], "exception": r["exception"]}
                for r in ordered if r["status"] != "OK"]
    n_ok = len(ordered) - len(failures)
    manifest = {"config_hash": cfg_hash, "started_utc": started, "elapsed_s": round(time.time() - t0, 3),
                "role": ROLE, "timing_evidence": TIMING_OK, "pins_before": "OK", "pins_after": pins_after or "OK",
                "counts": {"rows_total": len(ordered), "rows_ok": n_ok, "rows_failed": len(failures),
                           "clips_total": len(groups), "clips_ok": sum(s == "OK" for s in clip_status.values()),
                           "clips_failed": sum(s != "OK" for s in clip_status.values()), "by_label": labels},
                "clip_status": clip_status, "failures": failures, "rows": ordered}
    try:
        dump_new(out / "MANIFEST.json", manifest)
    except FileExistsError:  # earlier run's manifest is evidence; keep it
        dump_new(out / f"MANIFEST_{datetime.now(timezone.utc):%Y%m%dT%H%M%S%fZ}.json", manifest)
    print(f"rows {n_ok}/{len(ordered)} OK, {len(failures)} failed; clips {manifest['counts']['clips_ok']}/"
          f"{len(groups)} OK; pins_after {'OK' if not pins_after else 'MISMATCH'}; config {cfg_hash[:12]}")
    return 1 if failures or pins_after else 0


if __name__ == "__main__":
    sys.exit(main())
