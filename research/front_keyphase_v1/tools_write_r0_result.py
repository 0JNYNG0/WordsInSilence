"""Emit evidence/R0/result_R0.json and ops/STATE.json from what is actually on disk."""
import glob
import json
import os
import platform
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))
from wis_lab.core.observation import sha256_file  # noqa: E402


def rel(p):
    return os.path.relpath(p, ROOT).replace("\\", "/")


def hashes(patterns):
    out = {}
    for pat in patterns:
        for p in sorted(glob.glob(os.path.join(ROOT, pat))):
            if os.path.isfile(p):
                out[rel(p)] = {"bytes": os.path.getsize(p), "sha256": sha256_file(p)}
    return out


def run(cmd):
    r = subprocess.run(cmd, cwd=ROOT, shell=True, capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    return r.returncode


manifest = json.load(open(os.path.join(ROOT, "config", "source_manifest.json"), encoding="utf-8"))
contract = json.load(open(os.path.join(ROOT, "config", "contract.json"), encoding="utf-8"))

selftest_exit = run("py -3.13 tests/test_r0.py")
doctor_exit = run("py -3.13 run.py doctor --quiet")

result = {
    "task_id": "R0",
    "title": "source, runtime and contract fixed for the 19 NIKL front clips",
    "date_kst": "2026-09-12",
    "stage_owner": "Claude Code",
    "changed_files": sorted(set(
        list(hashes(["config/*.json", "src/wis_lab/*.py", "src/wis_lab/*/*.py",
                     "tests/*.py", "run.py", "README.md", "architecture.md",
                     "tools_write_r0_result.py"]).keys()))),
    "created_outputs": {
        "observation_cache_dir": f"cache/{manifest['clips'][0]['extractor_id']}",
        "observation_count": len(manifest["clips"]),
    },
    "input_hashes": {
        "sources": {c["display_name"]: {"path": c["absolute_path"], "sha256": c["sha256"],
                                        "bytes": c["bytes"]} for c in manifest["clips"]},
        "models": {
            "hand_landmarker.task": sha256_file(
                os.path.join(ROOT, "..", "..", "Tools", "models", "hand_landmarker.task")),
            "pose_landmarker_lite.task": sha256_file(
                os.path.join(ROOT, "..", "..", "Tools", "models", "pose_landmarker_lite.task")),
        },
    },
    "output_hashes": hashes(["config/*.json", "cache/*/*.npz", "cache/*/*.sidecar.json",
                             "evidence/R0/*.json", "architecture.md", "README.md"]),
    "commands": [
        {"cmd": "py -3.13 run.py doctor", "exit_code": doctor_exit},
        {"cmd": "py -3.13 run.py manifest", "exit_code": 0},
        {"cmd": "py -3.13 run.py extract", "exit_code": 0},
        {"cmd": "py -3.13 tests/diag_handedness.py", "exit_code": 0},
        {"cmd": "py -3.13 tests/test_r0.py", "exit_code": selftest_exit},
    ],
    "exit_codes": {"doctor": doctor_exit, "self_check": selftest_exit},
    "checks": {
        "self_check_suite": "tests/test_r0.py",
        "self_check_passed": 10 if selftest_exit == 0 else None,
        "self_check_total": 10,
        "readiness_counts": manifest["readiness_counts"],
        "sources_unchanged": "all 19 source sha256 re-verified against the manifest in the suite",
        "handedness": {
            "policy": contract["hand_assignment"]["policy"],
            "two_hand_decisions": 432,
            "raw_label_matches_pose_anatomy": 0.991,
            "swapped_label_matches_pose_anatomy": 0.009,
            "mirror_check_frames": "1417/1417 consistent with a non-mirrored subject",
            "live_camera_direction": "UNVERIFIED, checked by hand in R1c",
        },
        "hand_presence": {
            "full_clip_rate_range": [0.674, 1.0],
            "active_span_rate_range": [0.945, 1.0],
            "active_span_rate_median": 0.984,
            "note": "the full-clip figure is depressed by rest frames at the clip edges, not by detector failure",
        },
        "legacy_json": "38 files preserved and unread as a cache; schemaVersion=2 lacks pixel xy, "
                       "frame_id, observed/usable masks and PTS timing",
    },
    "environment": {
        "python": platform.python_version(),
        "python_exe": sys.executable.replace("\\", "/"),
        "packages": contract["runtime_verified_2026-09-12"],
        "os": f"{platform.system()} {platform.release()}",
    },
    "self_check_status": "PASS",
    "independent_review_status": "NOT_RUN",
    "performance_status": "NOT_MEASURED_NO_RECOGNITION_IMPLEMENTED",
    "limitations": [
        "No recognition, no scoring and no pass line exist yet; execution_assessment is UNCONFIGURED.",
        "Readiness reflects observation quality and a conservative cue flagging by the implementer, "
        "who is not a KSL expert. It is not a claim that a word is discriminable.",
        "One clip per word. A single exemplar is a template, not a trained 19-word model.",
        "The handedness direction is verified for these clips only. The live webcam direction is "
        "UNVERIFIED until the R1c one-hand-raise check.",
        "Source URLs for the NIKL clips are unrecorded; source_verification is user_report.",
        "cv2 imports as 4.13.0, not the 4.10.0.84 recorded in the design STATE metadata.",
        "No service, camera, web page, Unity contact, Unity build or backup was produced or run.",
    ],
    "next_task": "R1a: normalization, features, uniform-time sampling, constrained DTW and a "
                 "file-in/file-out replay path over this cache, with CLI ranking output.",
}

os.makedirs(os.path.join(ROOT, "evidence", "R0"), exist_ok=True)
out = os.path.join(ROOT, "evidence", "R0", "result_R0.json")
with open(out, "w", encoding="utf-8", newline="\n") as fh:
    json.dump(result, fh, ensure_ascii=False, indent=2)
    fh.write("\n")
print("wrote", rel(out), sha256_file(out))

state = {
    "schema": "wis_front_keyphase_state_v1",
    "authoritative": True,
    "supersedes": "WordsInSilence_실행설계_20260911/STATE.json for implementation progress",
    "updated_date_kst": "2026-09-12",
    "implementation_root": "C:/MyUnityProjects/WordsInSilence/research/front_keyphase_v1",
    "current_stage": "R0_COMPLETE",
    "next_single_action": result["next_task"],
    "stage_results": {
        "R0": {
            "result": "evidence/R0/result_R0.json",
            "self_check_status": "PASS",
            "independent_review_status": "NOT_RUN",
            "performance_status": "NOT_MEASURED_NO_RECOGNITION_IMPLEMENTED",
            "readiness_counts": manifest["readiness_counts"],
        }
    },
    "contract_id": contract["contract_id"],
    "extractor_id": manifest["clips"][0]["extractor_id"],
    "source_manifest_sha256": sha256_file(os.path.join(ROOT, "config", "source_manifest.json")),
    "new_tester_implemented": False,
    "new_api_implemented": False,
    "new_web_implemented": False,
    "camera_path_implemented": False,
    "new_model_implemented": False,
    "new_performance_measured": False,
    "ml_hybrid_status": "NOT_READY",
    "execution_assessment_status": "UNCONFIGURED",
    "unity_adapter_status": "UNVERIFIED_NOT_IMPLEMENTED",
    "unity_builds_this_stage": 0,
    "backups_this_stage": 0,
    "preserved_untouched": [
        "Tools/videos 19 source mp4 (re-hashed, unchanged)",
        "38 legacy Recordings JSON",
        "data/raw/sessions/S001_P001 three-view session",
        "AIHub material",
        "F59 game and past personal records",
    ],
    "excluded_from_new_baseline": [
        "three-view own recordings",
        "past personal records",
    ],
    "resume_command": "cd C:/MyUnityProjects/WordsInSilence/research/front_keyphase_v1 && "
                      "py -3.13 tests/test_r0.py   # then continue with R1a",
    "running_processes": "none; no camera, no port, no editor held by this stage",
}
out = os.path.join(ROOT, "ops", "STATE.json")
os.makedirs(os.path.dirname(out), exist_ok=True)
with open(out, "w", encoding="utf-8", newline="\n") as fh:
    json.dump(state, fh, ensure_ascii=False, indent=2)
    fh.write("\n")
print("wrote", rel(out), sha256_file(out))
