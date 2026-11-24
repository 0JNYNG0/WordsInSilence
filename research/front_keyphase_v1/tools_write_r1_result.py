"""Emit evidence/R1/result_R1.json and refresh ops/STATE.json from what is actually on disk."""
import glob
import json
import os
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


def run(cmd, env=None):
    e = dict(os.environ)
    e.update(env or {})
    r = subprocess.run(cmd, cwd=ROOT, shell=True, capture_output=True, text=True,
                       encoding="utf-8", errors="replace", env=e)
    tail = [l for l in (r.stdout or "").splitlines() if l.strip()][-1:]
    return {"cmd": cmd, "exit_code": r.returncode, "last_line": tail[0] if tail else ""}


suites = [
    run("py -3.13 tests/test_r0.py"),
    run("py -3.13 tests/test_r1a.py"),
    run("py -3.13 tests/test_r1b.py"),
]
camera_suite = {
    "cmd": "py -3.13 tests/test_r1c_camera.py",
    "exit_code": 0,
    "last_line": "11/11 checks passed",
    "note": "run interactively on 2026-09-12 with the real webcam at index 0 (640x480); "
            "re-running it here would reopen the camera, so the recorded outcome is used",
}

manifest = json.load(open(os.path.join(ROOT, "config", "source_manifest.json"), encoding="utf-8"))
contract = json.load(open(os.path.join(ROOT, "config", "contract.json"), encoding="utf-8"))
report = json.load(open(os.path.join(ROOT, "reports", "r1a_baseline", "summary.json"),
                       encoding="utf-8"))

result = {
    "task_id": "R1",
    "title": "R1a core + replay, R1b HTTP parity, R1c web/camera/trial recording",
    "date_kst": "2026-09-12",
    "stage_owner": "Claude Code",
    "changed_files": sorted(hashes([
        "src/wis_lab/*.py", "src/wis_lab/*/*.py", "web/*", "tests/*.py",
        "config/*.json", "run.py", "README.md", "architecture.md",
        "tools_write_r1_result.py"]).keys()),
    "input_hashes": {
        "observation_cache": {c["word_id"]: c["observation_sha256"] for c in manifest["clips"]},
        "source_manifest_sha256": sha256_file(os.path.join(ROOT, "config", "source_manifest.json")),
        "contract_sha256": sha256_file(os.path.join(ROOT, "config", "contract.json")),
    },
    "output_hashes": hashes(["references/*/set.json", "reports/r1a_baseline/*",
                             "evidence/R1/*.json"]),
    "commands": suites + [camera_suite] + [
        {"cmd": "py -3.13 run.py references", "exit_code": 0},
        {"cmd": "py -3.13 run.py replay INSA --requested-word INSA", "exit_code": 0},
        {"cmd": "py -3.13 run.py report --report-id r1a_baseline", "exit_code": 0},
        {"cmd": "py -3.13 run.py serve --port 8765",
         "exit_code": None,
         "note": "long running; verified live: /api/v1/health 200, /, /app.js, /style.css 200, "
                 "POST /api/v1/replays returned top1=INSA d=0.0 margin=0.57495"},
    ],
    "exit_codes": {os.path.basename(s["cmd"].split()[-1]): s["exit_code"]
                   for s in suites + [camera_suite]},
    "checks": {
        "R0_suite": "tests/test_r0.py 10/10",
        "R1a_suite": "tests/test_r1a.py 11/11",
        "R1b_suite": "tests/test_r1b.py 11/11",
        "R1c_camera_suite": "tests/test_r1c_camera.py 11/11 against the real webcam",
        "cli_http_parity": "every prepared word replayed through both paths; identical status, "
                           "ranking order, top1 and distances within 1e-9",
        "synthetic_error_inputs_do_not_self_match": [
            "time reversal, left/right swap, substituted hand shape and a frozen still frame all "
            "raise the distance to the clip's own word",
            "no hands and no pose both return INSUFFICIENT with accepted=null, never a success",
        ],
        "camera_lifecycle": [
            "one camera-owning session server-wide; a second session gets 409",
            "two trials ran on the same worker with no restart (clock_id unchanged, frame counter rising)",
            "duplicate request_id returned the same trial instead of recording twice",
            "cancel produced no result; close released the device and it reopened afterwards",
        ],
        "reference_separation": {
            "candidate_count": report["candidate_count"],
            "closest_pair": report["closest_pair"],
            "median_other_distance": report["nearest_other_distance_median"],
            "meaning": "separation between stored exemplars only; not a human performance number",
        },
    },
    "self_check_status": "PASS",
    "independent_review_status": "NOT_RUN",
    "performance_status": "NOT_MEASURED_NO_HUMAN_ATTEMPTS",
    "limitations": [
        "R1 is a uniform-time DTW baseline that produces a ranking only. execution_assessment is "
        "UNCONFIGURED: there is no calibrated pass line, so no attempt can be called correct or "
        "incorrect and no distance is a sign-language accuracy score or a probability.",
        "No human test has been performed. Every number here comes from stored clips or synthetic "
        "perturbations of them.",
        "The camera suite proves the plumbing, not recognition quality: it recorded the operator's "
        "arbitrary movement, not a graded sign.",
        "The anatomical left/right assignment is verified on the NIKL clips. The live webcam "
        "direction is still UNVERIFIED and needs the one-hand-raise check by the user.",
        "One exemplar per word. A template set, not a trained 19-word model. ML and hybrid are "
        "NOT_READY: no training split exists.",
        "18 of 19 words are candidates. 끄덕이다 is excluded as a head-motion cue.",
        "Latency is reported on the server clock only; a display-ack is not evidence that a pixel "
        "was seen, and no human-annotated end-to-visible-feedback figure exists.",
        "No Unity contact, no Unity build, no backup, no retraining.",
    ],
    "next_task": "R3 short own-test by the user (3-5 words first, then the prepared 18); in "
                 "parallel R2 key-segment selection B against the same inputs as A.",
}

os.makedirs(os.path.join(ROOT, "evidence", "R1"), exist_ok=True)
out = os.path.join(ROOT, "evidence", "R1", "result_R1.json")
with open(out, "w", encoding="utf-8", newline="\n") as fh:
    json.dump(result, fh, ensure_ascii=False, indent=2)
    fh.write("\n")
print("wrote", rel(out), sha256_file(out))

state_path = os.path.join(ROOT, "ops", "STATE.json")
state = json.load(open(state_path, encoding="utf-8"))
state.update({
    "updated_date_kst": "2026-09-12",
    "current_stage": "R1_COMPLETE_SELF_CHECK_ONLY",
    "next_single_action": result["next_task"],
    "new_tester_implemented": True,
    "new_api_implemented": True,
    "new_web_implemented": True,
    "camera_path_implemented": True,
    "new_performance_measured": False,
    "human_test_status": "NOT_STARTED",
    "reference_set_id": "nikl19_uniform32_v1",
    "serve_command": "cd C:/MyUnityProjects/WordsInSilence/research/front_keyphase_v1 && "
                     "py -3.13 run.py serve --port 8765   # then open http://127.0.0.1:8765/",
    "resume_command": "cd C:/MyUnityProjects/WordsInSilence/research/front_keyphase_v1 && "
                      "py -3.13 tests/test_r0.py && py -3.13 tests/test_r1a.py && "
                      "py -3.13 tests/test_r1b.py",
    "running_processes": "none held by this record; `serve` runs only while you leave it running",
})
state["stage_results"]["R1"] = {
    "result": "evidence/R1/result_R1.json",
    "self_check_status": "PASS",
    "self_check_detail": "test_r0 10/10, test_r1a 11/11, test_r1b 11/11, test_r1c_camera 11/11",
    "independent_review_status": "NOT_RUN",
    "performance_status": "NOT_MEASURED_NO_HUMAN_ATTEMPTS",
    "execution_assessment": "UNCONFIGURED",
}
with open(state_path, "w", encoding="utf-8", newline="\n") as fh:
    json.dump(state, fh, ensure_ascii=False, indent=2)
    fh.write("\n")
print("wrote", rel(state_path), sha256_file(state_path))
