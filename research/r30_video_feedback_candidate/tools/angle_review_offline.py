"""Offline runner for the R30 video-review angle comparison.

Runs the candidate's pure core helper over YOUR OWN local files and writes the JSON next to you.
It opens no camera, starts no server, contacts no network and sends nothing anywhere; every path
is one you type. Nothing is written except the --out file you name.

    py -3.13 tools/angle_review_offline.py ^
        --observation  <run>/trials/<trial>.npz ^
        --capture      <run>/trials/<trial>.capture.zip ^
        --contract     <project>/config/contract.json ^
        --reference-word 감사합니다 ^
        --source-manifest <project>/config/source_manifest.json ^
        --project-root    <project> ^
        --out          angle_review.json

Or, with a reference observation you already have:

    ... --reference-npz <project>/cache/<extractor>/NIKL_x_01.npz ^
        --reference-clip-id NIKL_x_01 --reference-word-id x

The reference is rebuilt with core.recognize.build_reference and the contract's own
sampling_A.budget - the same function and the same budget the server uses when it prepares its
reference set - so it reproduces that prepared sample rather than approximating it. It is still a
rebuild, not the server's live in-memory object; if the contract or the observation cache differs
from the running server's, the two will differ and this runner will say what it used.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))

from wis_lab.core.observation import Observation, sha256_file    # noqa: E402
from wis_lab.core.recognize import build_reference               # noqa: E402
from wis_lab.core import video_review as V                       # noqa: E402


def read_manifest(path: str) -> dict:
    """The same two lines service/diagnostic_capture.read_manifest runs.

    Inlined so this runner needs numpy only - no cv2, no mediapipe, no fastapi, no service layer.
    """
    with zipfile.ZipFile(path) as archive:
        return json.loads(archive.read("manifest.json"))


def load_reference(args, contract: dict) -> dict:
    budget = int(contract["sampling_A"]["budget"])
    if args.reference_npz:
        if not (args.reference_clip_id and args.reference_word_id):
            raise SystemExit("--reference-npz needs --reference-clip-id and --reference-word-id")
        obs = Observation.load(args.reference_npz)
        return build_reference(obs, args.reference_clip_id, args.reference_word_id, budget)
    if not (args.reference_word and args.source_manifest and args.project_root):
        raise SystemExit("give either --reference-npz or "
                         "--reference-word with --source-manifest and --project-root")
    with open(args.source_manifest, encoding="utf-8") as fh:
        manifest = json.load(fh)
    clip = next((c for c in manifest["clips"]
                 if args.reference_word in (c.get("word_id"), c.get("clip_id"))), None)
    if clip is None:
        raise SystemExit(f"no clip in the manifest for {args.reference_word!r}")
    if not clip.get("observation_npz"):
        raise SystemExit(f"{clip['clip_id']} has no extracted observation in this manifest")
    obs = Observation.load(os.path.join(args.project_root, clip["observation_npz"]))
    return build_reference(obs, clip["clip_id"], clip["word_id"], budget)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="offline R30 angle comparison over local files")
    ap.add_argument("--observation", required=True, help="the trial's saved .npz")
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--capture", help="the trial's .capture.zip")
    group.add_argument("--manifest-json", help="a manifest already extracted to a .json file")
    ap.add_argument("--contract", required=True, help="the frozen contract.json that was in use")
    ap.add_argument("--reference-word", help="word_id or clip_id in --source-manifest")
    ap.add_argument("--source-manifest", help="the project's config/source_manifest.json")
    ap.add_argument("--project-root", help="root the manifest's observation_npz paths are under")
    ap.add_argument("--reference-npz")
    ap.add_argument("--reference-clip-id")
    ap.add_argument("--reference-word-id")
    ap.add_argument("--out", help="write the full JSON payload here")
    ap.add_argument("--print-frames", action="store_true", help="one line per saved frame")
    ap.add_argument("--verify-only", action="store_true",
                    help="check the hashes and the frame identity, then stop")
    args = ap.parse_args(argv)

    with open(args.contract, encoding="utf-8") as fh:
        contract = json.load(fh)
    if args.capture:
        manifest = read_manifest(args.capture)
    else:
        with open(args.manifest_json, encoding="utf-8") as fh:
            manifest = json.load(fh)
    obs = Observation.load(args.observation)

    on_disk = sha256_file(args.observation)
    sidecar = (obs.sidecar or {}).get("observation_sha256")
    print(f"observation      {args.observation}")
    print(f"  sha256 on disk {on_disk}")
    print(f"  sidecar says   {sidecar}   {'OK' if sidecar == on_disk else 'MISMATCH'}")
    print(f"  manifest says  {manifest.get('observation_sha256')}   "
          f"{'OK' if manifest.get('observation_sha256') == on_disk else 'MISMATCH'}")
    print(f"  frames {obs.n_frames}, saved RGB frames {len(manifest.get('frames') or [])}, "
          f"capture status {manifest.get('status')}")
    if args.verify_only:
        return 0 if sidecar == on_disk == manifest.get("observation_sha256") else 1

    reference = load_reference(args, contract)
    print(f"reference        {reference['clip_id']} / {reference['word_id']} "
          f"({len(reference['features'])} sampled rows, {reference['selector']})")
    print(f"cost config      {V.cost_config(contract)}  band_ratio={V.band_ratio(contract)}")

    started = time.perf_counter_ns()
    try:
        payload = V.frame_angle_comparison(obs, manifest, reference, contract)
    except V.ReviewError as exc:
        print(f"REFUSED: {exc}")
        return 2
    elapsed_ms = round((time.perf_counter_ns() - started) / 1e6, 2)

    # strict JSON is part of the contract, so it is checked here too rather than assumed
    text = json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2)
    print(f"status           {payload['status']}")
    if payload["status"] == "OK":
        print(f"counts           {payload['counts']}")
        print(f"alignment        path_length={payload['alignment']['path_length']} "
              f"query_rows={payload['alignment']['query_rows']} "
              f"unaligned_reason={payload['alignment']['unaligned_reason']}")
    else:
        print(f"reason           {payload.get('reason')} {payload.get('detail', '')}")
    print(f"compute_ms       {elapsed_ms}   (this machine, this trial, once; not a benchmark)")

    if args.print_frames and payload["status"] == "OK":
        print("\n idx  frame_id   ts_us        row  state              L            R")
        for row in payload["frames"]:
            def hand(side):
                h = (row.get("hands") or {}).get(side)
                if not h:
                    return "-"
                if h["status"] != "COMPARED":
                    return h.get("reason", "NOT_COMPARABLE")[:11]
                values = [a["mean_abs_diff_deg"] for a in h["angles"]]
                return f"{min(values):.0f}~{max(values):.0f}deg"
            print(f"{row['index']:>4}  {str(row['frame_id']):>8}  {str(row['timestamp_us']):>11}  "
                  f"{str(row['observation_row']):>4}  {row['state']:<18} "
                  f"{hand('L'):<12} {hand('R')}")
        print("\ndegrees are raw differences between two projected 2D angles. They are not a "
              "score,\na probability or a correctness judgement, and no pass line exists.")

    if args.out:
        with open(args.out, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text + "\n")
        print(f"\nwrote {args.out} ({len(text)} bytes). Nothing else was written or sent.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
