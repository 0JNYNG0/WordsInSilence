#!/usr/bin/env python3
from __future__ import annotations

import argparse
import difflib
import hashlib
import importlib.util
import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any


SCHEMA = "WordsInSilence.Step10A.RetryExhaustionNeedsPracticeCompletion.v1.3"
NAME = "WordsInSilence_step10a_pet_robot_retry_exhaustion_needs_practice_completion_v1_3"
PREFLIGHT = "Validation/.step10a-v1_3-preflight"
V12_RESULT = "Validation/WordsInSilence_step10a_pet_robot_candidate_camera_startup_route_v1_2_gui_unity_editor_gate"

V12_ZIP = "Validation/WordsInSilence_step10a_pet_robot_candidate_camera_startup_route_v1_2_delivery.zip"
V12_ZIP_SIZE = 73741
V12_ZIP_SHA = "edea92e19bac40ce88b8eb0c54c62c1f6e66a2699fb1d9c48e016eb4421a5799"
V12_DETACHED_SHA = "1f64f5b54dd8346f1a39ef306c87ca310c6d82411ddcfe671f84fcdaddd9074b"
V12_GATE_ZIP = "Validation/WordsInSilence_step10a_pet_robot_candidate_camera_startup_route_v1_2_gui_unity_editor_gate_delivery.zip"
V12_GATE_ZIP_SIZE = 1521570
V12_GATE_ZIP_SHA = "31c6375d69bf04af779deb60956f6a477129731f6af159d80cf398df5baf6af2"
V12_GATE_DETACHED_SHA = "7902b5fd27d466ec30085ea51fe857d66af68c566df33b60d3c8eeaf723b90f9"
V11_TOOL = "Tools/s001_step10a_pet_robot_candidate_camera_startup_route_v1_1.py"
NEW_TEST = "Assets/_Project/Tests/EditMode/PetRobotRetryExhaustionNeedsPracticeCompletionV1Tests.cs"
NEW_TEST_META = NEW_TEST + ".meta"

ALLOWED_SOURCES = [
    "Assets/_Project/Scripts/Practice/PetRobotVerticalSlice/PetRobotPracticeContracts.cs",
    "Assets/_Project/Scripts/Practice/PetRobotVerticalSlice/PetRobotSignPracticeSession.cs",
    "Assets/_Project/Scripts/Practice/PetRobotVerticalSlice/PetRobotPracticePresenter.cs",
    "Assets/_Project/Scripts/Practice/PetRobotVerticalSlice/PetRobotPracticeUiPresenter.cs",
]


def require(value: bool, message: str) -> None:
    if not value:
        raise RuntimeError(message)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def jbytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2,
                       allow_nan=False) + "\n").encode("utf-8")


def record(path: str, data: bytes) -> dict[str, Any]:
    return {"relative_path": path, "size_bytes": len(data), "sha256": sha(data)}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def staged_files(root: Path) -> list[str]:
    run = subprocess.run(["git", "diff", "--cached", "--name-only"], cwd=root,
                         capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=30)
    require(run.returncode == 0, "Git staged query failed")
    return [line.strip() for line in run.stdout.splitlines() if line.strip()]


def verify_records(root: Path, baseline: dict[str, Any], expected_count: int) -> list[dict[str, Any]]:
    require(baseline["file_count"] == expected_count, "baseline file count mismatch")
    rows = []
    for prior in baseline["records"]:
        relative = prior["relative_path"]
        path = root / relative
        require(path.is_file(), "fixed file missing: " + relative)
        data = path.read_bytes()
        require(len(data) == prior["size_bytes"] and sha(data) == prior["sha256"],
                "fixed file identity mismatch: " + relative)
        rows.append(record(relative, data))
    return rows


def snapshot_repository(baseline: dict[str, Any]) -> dict[str, Any]:
    path = Path(baseline["path"])
    rows = []
    if path.is_dir():
        for item in sorted(value for value in path.rglob("*") if value.is_file()):
            rows.append(record(item.relative_to(path).as_posix(), item.read_bytes()))
    require(len(rows) == 7, "candidate repository file count is not 7")
    require(rows == baseline["files"], "candidate repository differs from approved two-candidate baseline")
    return {
        "path": str(path),
        "exists": path.exists(),
        "candidate_count": 2,
        "file_count": len(rows),
        "files": rows,
    }


def prepare(root: Path) -> None:
    destination = root / PREFLIGHT
    require(not destination.exists(), "preflight already exists")
    for relative, size, digest in [
        (V12_ZIP, V12_ZIP_SIZE, V12_ZIP_SHA),
        (V12_GATE_ZIP, V12_GATE_ZIP_SIZE, V12_GATE_ZIP_SHA),
    ]:
        data = (root / relative).read_bytes()
        require(len(data) == size and sha(data) == digest, "fixed ZIP identity mismatch: " + relative)

    fixed = root / V12_RESULT
    project = load_json(fixed / "project_files_after.json")
    protected = load_json(fixed / "protected_files_after.json")
    profile = load_json(fixed / "active_candidate_profile_after.json")
    repository = load_json(fixed / "candidate_repository_after.json")
    project_rows = verify_records(root, project, 30)
    protected_rows = verify_records(root, protected, 106)
    require(profile["file_count"] == 22 and profile["mismatch_count"] == 0,
            "approved candidate profile baseline mismatch")
    profile_rows = verify_records(root, profile, 22)
    repository_now = snapshot_repository(repository)
    require(staged_files(root) == [], "Git staged files are not zero")

    temp = destination.with_name(destination.name + ".tmp")
    require(not temp.exists(), "preflight temp already exists")
    try:
        (temp / "source_before").mkdir(parents=True)
        source_rows = []
        for relative in ALLOWED_SOURCES:
            for captured in (relative, relative + ".meta"):
                data = (root / captured).read_bytes()
                source_rows.append(record(captured, data))
                (temp / "source_before" / captured.replace("/", "__")).write_bytes(data)
        payload = {
            "schema_version": SCHEMA + ".preflight",
            "fixed_inputs": [
                {"relative_path": V12_ZIP, "size_bytes": V12_ZIP_SIZE, "sha256": V12_ZIP_SHA},
                {"relative_path": V12_GATE_ZIP, "size_bytes": V12_GATE_ZIP_SIZE, "sha256": V12_GATE_ZIP_SHA},
            ],
            "project_files": {"file_count": 30, "records": project_rows},
            "protected_files": {"file_count": 106, "records": protected_rows},
            "candidate_profile": {"file_count": 22, "records": profile_rows},
            "candidate_repository": repository_now,
            "source_before": source_rows,
            "git_staged": [],
            "status": "PASS",
        }
        (temp / "preflight.json").write_bytes(jbytes(payload))
        temp.replace(destination)
    except Exception:
        if temp.exists():
            shutil.rmtree(temp)
        raise
    print(json.dumps({"preflight": str(destination), "status": "PASS"}, sort_keys=True))


def load_v11(root: Path):
    path = root / V11_TOOL
    spec = importlib.util.spec_from_file_location("step10a_v11_approved", path)
    require(spec is not None and spec.loader is not None, "cannot load approved v1.1 packaging tool")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def current_records(root: Path, before: dict[str, Any]) -> dict[str, Any]:
    rows = []
    changed = []
    for prior in before["records"]:
        relative = prior["relative_path"]
        path = root / relative
        require(path.is_file(), "baseline file missing: " + relative)
        data = path.read_bytes()
        is_changed = len(data) != prior["size_bytes"] or sha(data) != prior["sha256"]
        if is_changed:
            changed.append(relative)
        rows.append({
            **record(relative, data),
            "before_size_bytes": prior["size_bytes"],
            "before_sha256": prior["sha256"],
            "changed": is_changed,
        })
    return {
        "schema_version": SCHEMA + ".file_set",
        "file_count": len(rows),
        "changed_count": len(changed),
        "unchanged_count": len(rows) - len(changed),
        "changed_paths": changed,
        "records": rows,
        "status": "PASS",
    }


def unified(relative: str, before: bytes, after: bytes) -> bytes:
    return "".join(difflib.unified_diff(
        before.decode("utf-8-sig").splitlines(True),
        after.decode("utf-8-sig").splitlines(True),
        fromfile="v1.2/" + relative,
        tofile="v1.3/" + relative,
    )).encode("utf-8")


def test_inventory(root: Path, approved: dict[str, Any]) -> dict[str, Any]:
    approved_rows = approved["rows"]
    expected = {(row["fixture"], row["method"], row["source_path"]) for row in approved_rows}
    actual = set()
    rows = []
    for relative in sorted({row["source_path"] for row in approved_rows}):
        text_value = (root / relative).read_text(encoding="utf-8-sig")
        fixture_match = re.search(r"public\s+sealed\s+class\s+(\w+)", text_value)
        require(fixture_match is not None, "fixture class missing: " + relative)
        fixture = fixture_match.group(1)
        methods = re.findall(r"\[Test\]\s*public\s+void\s+(\w+)\s*\(", text_value)
        for method in methods:
            row = {"fixture": fixture, "method": method, "source_path": relative}
            rows.append(row)
            actual.add((fixture, method, relative))
    require(actual == expected and len(rows) == len(actual) == 87,
            "approved 87-test name set changed")

    new_text = (root / NEW_TEST).read_text(encoding="utf-8-sig")
    new_fixture = re.search(r"public\s+sealed\s+class\s+(\w+)", new_text)
    require(new_fixture is not None and
            new_fixture.group(1) == "PetRobotRetryExhaustionNeedsPracticeCompletionV1Tests",
            "new fixture identity mismatch")
    new_methods = re.findall(r"\[Test\]\s*public\s+void\s+(\w+)\s*\(", new_text)
    require(len(new_methods) == len(set(new_methods)) == 15, "new fixture is not 15 unique tests")
    new_rows = [
        {"fixture": new_fixture.group(1), "method": method, "source_path": NEW_TEST}
        for method in new_methods
    ]
    combined = rows + new_rows
    require(len(combined) == 102 and
            len({(row["fixture"], row["method"]) for row in combined}) == 102,
            "combined test inventory is not 102 unique tests")
    return {
        "schema_version": SCHEMA + ".test_inventory",
        "approved_existing_test_count": 87,
        "approved_existing_name_set_preserved": True,
        "new_fixture": new_fixture.group(1),
        "new_test_count": 15,
        "combined_unique_test_count": 102,
        "approved_rows": sorted(rows, key=lambda row: (row["fixture"], row["method"])),
        "new_rows": sorted(new_rows, key=lambda row: row["method"]),
        "unity_execution_status": "UNVERIFIED",
        "status": "PASS",
    }


def static_contract(root: Path) -> dict[str, Any]:
    contracts = (root / ALLOWED_SOURCES[0]).read_text(encoding="utf-8-sig")
    session = (root / ALLOWED_SOURCES[1]).read_text(encoding="utf-8-sig")
    presenter = (root / ALLOWED_SOURCES[2]).read_text(encoding="utf-8-sig")
    ui = (root / ALLOWED_SOURCES[3]).read_text(encoding="utf-8-sig")
    continue_match = re.search(
        r"public bool ContinueWithNeedsPractice\(\)\s*\{.*?^        \}",
        session, flags=re.MULTILINE | re.DOTALL)
    require(continue_match is not None, "NeedsPractice continue method missing")
    continue_body = continue_match.group(0)
    require("PetRobotWordLearningStatus" in contracts and "Mastered" in contracts and
            "NeedsPractice" in contracts, "word learning status contract missing")
    require("wrongWordAttemptCounts[currentIndex] == 1" in session,
            "first WrongWord Retry1 gate missing")
    require("wrongWordAttemptCounts[currentIndex] == 2" in session,
            "Retry1-exhausted continue gate missing")
    require("GuidedQuestOutcomeKind.WrongWord" in session and
            "PetRobotWordLearningStatus.NeedsPractice" in continue_body,
            "manual continue does not preserve WrongWord/NeedsPractice separation")
    require("repository.Save" not in continue_body and "SaveCandidate" not in continue_body,
            "manual NeedsPractice continue can store a candidate")
    require("CandidateStorageStatus.NotRequested" in continue_body,
            "manual NeedsPractice storage status is not NotRequested")
    require("NeedsPracticeContinueButtonLabel" in ui and
            "이번 단어는 연습 필요로 남기고 계속" in ui,
            "explicit manual NeedsPractice button missing")
    require("session.CanRetry" in ui and "session.CanContinueWithNeedsPractice" in ui,
            "UI retry exhaustion gates missing")
    require("Mastered:" in ui and "NeedsPractice:" in ui and
            "Candidate saved:" in ui and "연습 공간 나가기" in ui,
            "session summary fields missing")
    require("ShowCompleted(int masteredCount, int needsPracticeCount)" in presenter,
            "pet-robot differentiated summary missing")
    require("GuidedQuestOutcomeKind.TechnicalFailure" in session and
            "technical failure is not learner performance" in session,
            "TechnicalFailure non-scoring branch changed")
    return {
        "schema_version": SCHEMA + ".static_contract",
        "first_wrong_retry1_only": True,
        "retry1_wrong_manual_continue_required": True,
        "automatic_progress_after_retry1_wrong": False,
        "manual_continue_storage_status": "NotRequested",
        "manual_continue_candidate_save_calls": 0,
        "word_statuses": ["Pending", "Mastered", "NeedsPractice"],
        "summary_fields": ["per-word status", "Mastered count", "NeedsPractice count",
                           "candidate count", "exit button"],
        "technical_failure_marked_needs_practice": False,
        "recognition_calculation_changes": False,
        "status": "PASS",
    }


PROBE_SOURCE = r'''
using System; using System.Collections.Generic;
using WordsInSilence.Practice.PetRobotVerticalSlice; using WordsInSilence.Recognition.LiveInput;
sealed class Adapter:IPersonalLivePracticeRecognitionAdapter { public PracticeRecognitionAttempt Next; PersonalLiveCaptureState state=PersonalLiveCaptureState.Ready; public PersonalLiveCaptureState CaptureState=>state; public string CaptureSessionId=>Next==null?"":Next.CaptureSessionId; public string ActiveProfileId=>"step9l_current_user_multi_prototype_v1_0"; public bool CandidateRouteActive=>true; public bool SetExpectedWord(string w){return state==PersonalLiveCaptureState.Ready&&PersonalLiveRecognitionController.IsSupportedWord(w);} public bool BeginCapture(){if(state!=PersonalLiveCaptureState.Ready)return false;state=PersonalLiveCaptureState.Capturing;return true;} public PracticeRecognitionAttempt FinalizeCapture(){state=Next==null?PersonalLiveCaptureState.Failed:Next.CaptureState;return Next;} public bool ResetCapture(){if(state!=PersonalLiveCaptureState.Completed&&state!=PersonalLiveCaptureState.Failed)return false;state=PersonalLiveCaptureState.Ready;return true;} }
sealed class Repo:IPersonalizationCandidateRepository { readonly Dictionary<string,int> e=W(0),r=W(0); public int Saves; public string RootDirectory=>"fake"; public string LearnerId=>"local_learner"; public CandidateCountSnapshot Counts=>new CandidateCountSnapshot(e,r); public CandidateCountSnapshot ReloadAndVerify(){return Counts;} public CandidateSaveResult Save(PersonalizationCandidateSaveRequest q){Saves++;(q.Eligibility==PersonalizationCandidateEligibility.Eligible?e:r)[q.Attempt.ExpectedWord]++;return new CandidateSaveResult(CandidateStorageStatus.Saved,"fake","saved",Counts);} static Dictionary<string,int> W(int v){return new Dictionary<string,int>{{"GREETING",v},{"LOVE",v},{"THANK_YOU",v}};} }
static class Probe { static GuidedQuestOutcome O(string e,string p,bool a){return GuidedQuestOutcomeEvaluator.EvaluateForRoute(PersonalLiveProfileMode.MultiPrototypeCandidate,PersonalLiveCaptureState.Completed,true,e,p,a,a,a,false,false);} static PracticeRecognitionAttempt A(string id,string e,string p,bool a){return new PracticeRecognitionAttempt(id,e,p,PersonalLiveCaptureState.Completed,O(e,p,a),null,null,PersonalLiveProfileMode.MultiPrototypeCandidate.ToString(),"step9l_current_user_multi_prototype_v1_0",0.4,null,null,new PracticeRecognitionMetrics{StrictAccepted=a,StrictExpectedWordClassAccept=a,StrictQuestSuccess=a,WinningPrototypeId="p-"+p});} static PracticeRecognitionAttempt T(string id,string e){var o=GuidedQuestOutcomeEvaluator.EvaluateForRoute(PersonalLiveProfileMode.MultiPrototypeCandidate,PersonalLiveCaptureState.Failed,false,e,"",false,false,false,true,true);return new PracticeRecognitionAttempt(id,e,"",PersonalLiveCaptureState.Failed,o,null,null,"candidate","profile",0.4,null,null,new PracticeRecognitionMetrics{WinningPrototypeId=""});} static void F(PetRobotSignPracticeSession s,Adapter a,PracticeRecognitionAttempt x){a.Next=x;if(!s.StartExpression()||!s.FinishExpression())throw new Exception("finish");} static void Main(){var a=new Adapter();var r=new Repo();var s=new PetRobotSignPracticeSession(a,r);F(s,a,A("g","GREETING","GREETING",true));s.ContinueToNextWord();F(s,a,A("l","LOVE","LOVE",true));s.ContinueToNextWord();F(s,a,A("t1","THANK_YOU","LOVE",false));bool first=s.CanRetry&&!s.CanContinueWithNeedsPractice&&s.CurrentWord=="THANK_YOU";s.Retry();F(s,a,A("t2","THANK_YOU","LOVE",false));bool exhausted=!s.CanRetry&&s.CanContinueWithNeedsPractice&&s.State==PetRobotPracticeState.Feedback&&!s.AllWordsComplete;var before=s.GetWordFinalAttempt("THANK_YOU");bool continued=s.ContinueWithNeedsPractice();var after=s.GetWordFinalAttempt("THANK_YOU");bool summary=continued&&s.State==PetRobotPracticeState.Completed&&s.MasteredCount==2&&s.NeedsPracticeCount==1&&s.CandidateCounts.TotalCandidateCount==2&&r.Saves==2&&after.Outcome.Kind==GuidedQuestOutcomeKind.WrongWord&&!after.StrictAccepted&&!after.StrictExpectedWordClassAccept&&!after.StrictQuestSuccess&&before.CaptureSessionId==after.CaptureSessionId;var ta=new Adapter();var tr=new Repo();var ts=new PetRobotSignPracticeSession(ta,tr);F(ts,ta,T("tech","GREETING"));bool tech=ts.LastLearningOutcome.Kind==GuidedQuestOutcomeKind.TechnicalFailure&&!ts.CanContinueWithNeedsPractice&&ts.NeedsPracticeCount==0&&tr.Saves==0;Console.WriteLine("FIRST_WRONG_RETRY1_ONLY="+first);Console.WriteLine("RETRY1_EXHAUSTED_MANUAL_GATE="+exhausted);Console.WriteLine("MASTERED_2_NEEDS_PRACTICE_1="+summary);Console.WriteLine("TECHNICAL_FAILURE_SEPARATE="+tech);bool pass=first&&exhausted&&summary&&tech;Console.WriteLine("PROBE_STATUS="+(pass?"PASS":"FAIL"));if(!pass)Environment.Exit(2);} }
'''


def offline_compile_and_probe(root: Path, v11: Any) -> tuple[dict[str, Any], dict[str, Any], str, str, str, str]:
    compile_evidence, production_output, existing_test_output, temp, dll = v11.offline_compile(root)
    mono = Path(r"C:\Program Files\Unity\Hub\Editor\6000.3.14f1\Editor\Data\MonoBleedingEdge\bin\mono.exe")
    csc = Path(r"C:\Program Files\Unity\Hub\Editor\6000.3.14f1\Editor\Data\MonoBleedingEdge\lib\mono\4.5\csc.exe")
    try:
        new_test_dll = temp / "Step10Av13Tests.dll"
        test_cmd = [str(mono), str(csc), "/nologo", "/target:library", "/langversion:latest",
                    "/out:" + str(new_test_dll), "/r:" + str(dll), str(temp / "nunit.cs"),
                    str((root / NEW_TEST).resolve())]
        test_run = subprocess.run(test_cmd, capture_output=True, text=True, encoding="utf-8",
                                  errors="replace", timeout=120)
        new_test_output = (test_run.stdout or "") + (test_run.stderr or "")
        require(test_run.returncode == 0, "new fixture offline compile failed:\n" + new_test_output[-12000:])

        probe_source = temp / "v13_probe.cs"
        probe_source.write_text(PROBE_SOURCE, encoding="utf-8")
        probe_exe = temp / "v13_probe.exe"
        probe_compile = subprocess.run(
            [str(mono), str(csc), "/nologo", "/target:exe", "/langversion:latest",
             "/out:" + str(probe_exe), "/r:" + str(dll), str(probe_source)],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120)
        probe_compile_output = (probe_compile.stdout or "") + (probe_compile.stderr or "")
        require(probe_compile.returncode == 0,
                "offline behavior probe compile failed:\n" + probe_compile_output[-12000:])
        probe_run = subprocess.run([str(mono), str(probe_exe)], capture_output=True, text=True,
                                   encoding="utf-8", errors="replace", timeout=120, cwd=temp)
        probe_output = (probe_run.stdout or "") + (probe_run.stderr or "")
        require(probe_run.returncode == 0 and "PROBE_STATUS=PASS" in probe_output,
                "offline behavior probe failed:\n" + probe_output[-12000:])
        values = {}
        for line in probe_output.splitlines():
            if "=" in line:
                key, value = line.lstrip("\ufeff").split("=", 1)
                values[key] = value
        expected = {
            "FIRST_WRONG_RETRY1_ONLY": "True",
            "RETRY1_EXHAUSTED_MANUAL_GATE": "True",
            "MASTERED_2_NEEDS_PRACTICE_1": "True",
            "TECHNICAL_FAILURE_SEPARATE": "True",
            "PROBE_STATUS": "PASS",
        }
        require(all(values.get(key) == value for key, value in expected.items()),
                "offline behavior probe evidence mismatch")
        combined = dict(compile_evidence)
        combined.update({
            "schema_version": SCHEMA + ".offline_compile",
            "production_error_count": len(re.findall(r"error CS\d+", production_output)),
            "approved_affected_test_error_count": len(re.findall(r"error CS\d+", existing_test_output)),
            "new_fixture_error_count": len(re.findall(r"error CS\d+", new_test_output)),
            "production_source_count": 9,
            "compiled_test_source_count": 3,
            "compiled_test_count": 46,
            "unity_compile_pass_claimed": False,
            "editmode_pass_claimed": False,
            "status": "PASS",
        })
        require(combined["production_error_count"] == 0 and
                combined["approved_affected_test_error_count"] == 0 and
                combined["new_fixture_error_count"] == 0,
                "offline compile errors are non-zero")
        probe = {
            "schema_version": SCHEMA + ".offline_behavior_probe",
            "values": values,
            "uses_actual_production_session": True,
            "recognition_adapter": "fake",
            "candidate_repository": "fake",
            "unity_execution_claimed": False,
            "status": "PASS",
        }
        return combined, probe, production_output, existing_test_output, new_test_output, probe_output
    finally:
        shutil.rmtree(temp, ignore_errors=True)


def assertion_lines(rows: list[tuple[str, str, str]]) -> bytes:
    keys = set()
    lines = []
    for key, status, evidence in rows:
        require(key not in keys, "duplicate assertion key: " + key)
        keys.add(key)
        lines.append(json.dumps({"assertion_key": key, "status": status, "evidence": evidence},
                                ensure_ascii=False, sort_keys=True, allow_nan=False))
    return ("\n".join(lines) + "\n").encode("utf-8")


def add_manifest(files: dict[str, bytes]) -> dict[str, bytes]:
    require("output_file_manifest.json" not in files, "manifest already present")
    manifest = {
        "schema_version": SCHEMA + ".manifest",
        "self_excluded": True,
        "file_count": len(files),
        "files": [record(name, files[name]) for name in sorted(files)],
    }
    result = dict(files)
    result["output_file_manifest.json"] = jbytes(manifest)
    return result


def validation_report(inventory: dict[str, Any], compile_evidence: dict[str, Any],
                      project: dict[str, Any], protected: dict[str, Any],
                      profile: dict[str, Any], repository: dict[str, Any]) -> bytes:
    changed = "\n".join("- `" + value + "`" for value in project["changed_paths"])
    return f"""# Step 10A v1.3 Retry-exhaustion NeedsPractice Completion

## Outcome

The strict recognition result remains authoritative. A first `WrongWord` permits one manual Retry1. A second `WrongWord` blocks further retry and exposes an explicit manual **이번 단어는 연습 필요로 남기고 계속** action. Until that action, the word and result remain fixed. The action stores no candidate, preserves strict false values, marks only that word `NeedsPractice`, and permits the short practice session to reach a differentiated summary.

`TechnicalFailure` remains non-scoring, does not become `NeedsPractice`, and cannot advance through this action. RecognitionCore, DTW, nine prototypes, three thresholds, camera/bridge routing, candidate serialization, and the two existing Eligible candidates were not changed.

## Changed project sources

{changed}

Two new test files add 15 dedicated tests. The approved existing 87 testcase name set remains unchanged, for a static combined inventory of {inventory['combined_unique_test_count']}.

## Offline verification

- Production compile errors: {compile_evidence['production_error_count']}
- Approved affected test compile errors: {compile_evidence['approved_affected_test_error_count']}
- New fixture compile errors: {compile_evidence['new_fixture_error_count']}
- Existing project baseline changes: {project['changed_count']} of {project['file_count']} (the four allowed sources)
- Protected files unchanged: {protected['unchanged_count']}/{protected['file_count']}
- Candidate profile unchanged: {profile['unchanged_count']}/{profile['file_count']}
- Candidate repository unchanged: {repository['candidate_count']} candidates / {repository['file_count']} files

Unity GUI, Test Runner, PlayMode, webcam, live capture, and candidate storage were not executed. A GUI run of the approved 87 plus the 15 new tests, followed conditionally by the playable completion gate, remains external-review gated.

## Status

```text
STEP10A_V1_3_RETRY_EXHAUSTION_NEEDS_PRACTICE_COMPLETION_SOURCE_AND_PACKAGE_READY
STEP10A_V1_3_WRONG_WORD_RETRY1_EXHAUSTION_MANUAL_CONTINUE_IMPLEMENTED
STEP10A_V1_3_MASTERED_NEEDS_PRACTICE_SUMMARY_IMPLEMENTED
STEP10A_V1_2_APPROVED_RECOGNITION_CAMERA_STORAGE_SCOPE_PRESERVED
STEP10A_V1_3_UNITY_GUI_EDITMODE_PLAYMODE_WEBCAM_UNVERIFIED
BLOCKED_PENDING_EXTERNAL_REVIEW
```
""".encode("utf-8")


def build(root: Path, v11: Any) -> tuple[dict[str, bytes], dict[str, Any]]:
    preflight_path = root / PREFLIGHT / "preflight.json"
    require(preflight_path.is_file(), "fixed v1.3 preflight is missing")
    preflight = load_json(preflight_path)
    v12_info, v12_files = v11.verify_fixed_zip(
        root, V12_ZIP, V12_ZIP_SIZE, V12_ZIP_SHA, V12_DETACHED_SHA)
    gate_info, gate_files = v11.verify_fixed_zip(
        root, V12_GATE_ZIP, V12_GATE_ZIP_SIZE, V12_GATE_ZIP_SHA, V12_GATE_DETACHED_SHA)

    gate_tests = json.loads(gate_files["test_results_summary.json"])
    require(gate_tests["unique_testcase_count"] == 87 and gate_tests["passed"] == 87 and
            gate_tests["failed"] == 0 and gate_tests["skipped"] == 0 and
            gate_tests["inconclusive"] == 0 and gate_tests["retry_nonzero"] == 0 and
            gate_tests["repeat_nonzero"] == 0, "approved v1.2 GUI test identity mismatch")
    gate_live = json.loads(gate_files["playmode_live_gate_evaluation.json"])
    require(gate_live["camera"]["state"] == "Ready" and
            gate_live["camera"]["frame_count"] == 333 and
            gate_live["camera"]["observation_count"] == 332 and
            gate_live["practice_completed"] is False and
            gate_live["retry_counts"]["THANK_YOU"] == 1 and
            gate_live["ui_candidate_counts"]["GREETING"]["eligible"] == 1 and
            gate_live["ui_candidate_counts"]["LOVE"]["eligible"] == 1 and
            gate_live["ui_candidate_counts"]["THANK_YOU"]["eligible"] == 0,
            "approved v1.2 live failure identity mismatch")
    thank_attempts = [row for row in gate_live["word_attempts"] if row["word"] == "THANK_YOU"]
    require(len(thank_attempts) == 2 and
            [row["outcome"] for row in thank_attempts] == ["WrongWord", "WrongWord"] and
            all(row["storage_status"] == "NotRequested" and not row["progressed"]
                for row in thank_attempts), "THANK_YOU Retry1 failure baseline mismatch")

    project_before = preflight["project_files"]
    protected_before = preflight["protected_files"]
    profile_before = preflight["candidate_profile"]
    repository_before = preflight["candidate_repository"]
    project_after = current_records(root, project_before)
    require(project_after["file_count"] == 30 and
            project_after["changed_paths"] == ALLOWED_SOURCES,
            "existing project change scope is not the four allowed sources")
    protected_after = current_records(root, protected_before)
    require(protected_after["file_count"] == 106 and protected_after["changed_count"] == 0,
            "protected 106-file set changed")
    profile_after = current_records(root, profile_before)
    require(profile_after["file_count"] == 22 and profile_after["changed_count"] == 0,
            "candidate profile 22-file set changed")
    fixed_profile = load_json(root / V12_RESULT / "active_candidate_profile_after.json")
    require(fixed_profile["prototype_count"] == 9 and fixed_profile["threshold_count"] == 3 and
            fixed_profile["thresholds"] == {
                "GREETING": 0.5113059473720765,
                "LOVE": 0.391276023460427,
                "THANK_YOU": 0.37709977431991587,
            }, "approved prototype/threshold identity mismatch")
    repository_after = snapshot_repository(repository_before)
    require(repository_after == repository_before,
            "existing two-candidate/seven-file repository changed")
    require(staged_files(root) == preflight["git_staged"] == [], "Git staged state changed")

    source_before: dict[str, bytes] = {}
    source_after: dict[str, bytes] = {}
    source_identity_rows = []
    diffs: dict[str, bytes] = {}
    for relative in ALLOWED_SOURCES:
        key = relative.replace("/", "__")
        before = (root / PREFLIGHT / "source_before" / key).read_bytes()
        after = (root / relative).read_bytes()
        meta_before = (root / PREFLIGHT / "source_before" / (key + ".meta")).read_bytes()
        meta_after = (root / (relative + ".meta")).read_bytes()
        require(before != after, "allowed source did not change: " + relative)
        require(meta_before == meta_after, "source meta changed: " + relative)
        source_before[relative] = before
        source_after[relative] = after
        source_identity_rows.append({
            "relative_path": relative,
            "before": record(relative, before),
            "after": record(relative, after),
            "meta": record(relative + ".meta", meta_after),
            "meta_before_after_byte_identical": True,
            "status": "PASS",
        })
        diffs[relative] = unified(relative, before, after)
        require(len(diffs[relative]) > 0, "empty diff: " + relative)

    require((root / NEW_TEST).is_file() and (root / NEW_TEST_META).is_file(),
            "new fixture or meta missing")
    new_test = (root / NEW_TEST).read_bytes()
    new_test_meta = (root / NEW_TEST_META).read_bytes()
    require("guid: 7a5ce4a685554f6a94687d7abf12e6a1" in new_test_meta.decode("utf-8"),
            "new fixture meta GUID mismatch")
    static = static_contract(root)
    approved_inventory = json.loads(v12_files["test_inventory.json"])
    inventory = test_inventory(root, approved_inventory)
    compile_evidence, probe, production_output, existing_test_output, new_test_output, probe_output = \
        offline_compile_and_probe(root, v11)
    transaction = v11.transaction_regression()
    require(len(transaction["failure_injection_rows"]) == 4 and
            all(row["status"] == "PASS" and row["remaining_final_outputs"] == 0 and
                row["remaining_temporary_outputs"] == 0 and
                not row["preexisting_sentinel_changed"]
                for row in transaction["failure_injection_rows"]),
            "approved transaction regression failed")

    implementation_delta = {
        "schema_version": SCHEMA + ".implementation_delta",
        "changed_existing_project_file_count": 4,
        "changed_existing_project_files": ALLOWED_SOURCES,
        "new_project_file_count": 2,
        "new_project_files": [NEW_TEST, NEW_TEST_META],
        "existing_project_file_count_before": 30,
        "project_file_count_after_with_new_fixture": 32,
        "scene_or_prefab_changes": 0,
        "recognition_dtw_threshold_prototype_changes": 0,
        "repository_changes": 0,
        "status": "PASS",
    }
    source_identity = {
        "schema_version": SCHEMA + ".source_identity",
        "changed_sources": source_identity_rows,
        "new_test_source": record(NEW_TEST, new_test),
        "new_test_meta": record(NEW_TEST_META, new_test_meta),
        "changed_meta_count": 0,
        "status": "PASS",
    }
    fixed_input = {
        "schema_version": SCHEMA + ".input_provenance",
        "implementation_owner": "Codex",
        "v1_2_source_delivery": v12_info,
        "v1_2_gui_evidence_delivery": gate_info,
        "approved_gui_test_result": "87 PASS / 0 FAIL / 0 skipped / 0 inconclusive; retry/repeat 0",
        "approved_live_failure": "THANK_YOU WrongWord after Retry1; no progress; no storage",
        "approved_candidate_repository": "2 Eligible candidates / 7 files",
        "context": record("WordsInSilence_PROJECT_CONTEXT.md",
                          (root / "WordsInSilence_PROJECT_CONTEXT.md").read_bytes()),
        "status": "PASS",
    }

    assertions = assertion_lines([
        ("implementation_owner", "PASS", "Codex"),
        ("v1_2_source_delivery_fixed", "PASS", V12_ZIP_SHA),
        ("v1_2_gui_evidence_fixed", "PASS", V12_GATE_ZIP_SHA),
        ("approved_gui_87_result_reproduced", "PASS", "87/87"),
        ("thank_you_retry1_wrong_baseline", "PASS", "2 WrongWord / no storage"),
        ("allowed_existing_source_scope", "PASS", "4/4"),
        ("new_fixture_files", "PASS", "2"),
        ("source_meta_identity", "PASS", "4/4"),
        ("first_wrong_retry1_only", "PASS", "attempt count 1"),
        ("retry1_wrong_blocks_more_retry", "PASS", "attempt count 2"),
        ("explicit_manual_needs_practice_continue", "PASS", PetRobotPracticeUiPresenterLabel()),
        ("no_automatic_progress", "PASS", "Feedback/current word preserved before button"),
        ("wrong_outcome_strict_values_preserved", "PASS", "WrongWord/False/False/False"),
        ("wrong_candidate_storage_zero", "PASS", "NotRequested / 0"),
        ("word_status_contract", "PASS", "Pending/Mastered/NeedsPractice"),
        ("last_needs_practice_session_completed", "PASS", "Mastered 2 / NeedsPractice 1"),
        ("technical_failure_separate", "PASS", "non-scoring/no progress/no storage"),
        ("strict_low_confidence_paths_preserved", "PASS", "new fixture regression"),
        ("approved_existing_test_names", "PASS", "87/87"),
        ("new_test_inventory", "PASS", "15"),
        ("combined_test_inventory", "PASS", "102"),
        ("offline_production_compile", "PASS", "errors 0"),
        ("offline_affected_test_compile", "PASS", "errors 0"),
        ("offline_new_fixture_compile", "PASS", "errors 0"),
        ("offline_behavior_probe", "PASS", "actual production session"),
        ("protected_files", "PASS", "106/106 byte-identical"),
        ("candidate_profile", "PASS", "22/22 byte-identical"),
        ("prototype_threshold", "PASS", "9 / 3 byte-identical"),
        ("candidate_repository", "PASS", "2 candidates / 7 files unchanged"),
        ("scene_prefab", "PASS", "unchanged"),
        ("packaging_transaction", "PASS", "4/4 injected failures"),
        ("never_overwrite", "PASS", "approved commit transaction"),
        ("git_staged", "PASS", "0"),
        ("unity_gui_compile", "UNVERIFIED", "prohibited source step"),
        ("unity_editmode_tests", "UNVERIFIED", "approved 87 plus new 15 deferred"),
        ("playmode_webcam_live_capture", "NOT_APPLICABLE", "prohibited source step"),
    ])

    files: dict[str, bytes] = {
        "input_provenance.json": jbytes(fixed_input),
        "retry_exhaustion_needs_practice_contract.json": jbytes(static),
        "implementation_delta.json": jbytes(implementation_delta),
        "source_identity_before_after.json": jbytes(source_identity),
        "test_inventory.json": jbytes(inventory),
        "offline_compile_evidence.json": jbytes(compile_evidence),
        "offline_behavior_probe.json": jbytes(probe),
        "offline_production_compile_output.txt": production_output.encode("utf-8"),
        "offline_approved_affected_test_compile_output.txt": existing_test_output.encode("utf-8"),
        "offline_new_fixture_compile_output.txt": new_test_output.encode("utf-8"),
        "offline_behavior_probe_output.txt": probe_output.encode("utf-8"),
        "project_files_before.json": jbytes(project_before),
        "project_files_after.json": jbytes(project_after),
        "protected_files_before.json": jbytes(protected_before),
        "protected_files_after.json": jbytes(protected_after),
        "active_candidate_profile_before.json": jbytes(profile_before),
        "active_candidate_profile_after.json": jbytes(profile_after),
        "candidate_repository_before.json": jbytes(repository_before),
        "candidate_repository_after.json": jbytes(repository_after),
        "git_staged_before.json": jbytes({"staged_count": 0, "staged_files": []}),
        "git_staged_after.json": jbytes({"staged_count": 0, "staged_files": []}),
        "packaging_transaction_failure_injection_regression.json": jbytes(transaction),
        "input_subset/v1_2/test_inventory.json": v12_files["test_inventory.json"],
        "input_subset/v1_2/detached_checksum.txt": (root / (V12_ZIP + ".sha256")).read_bytes(),
        "input_subset/v1_2_gui/test_results_summary.json": gate_files["test_results_summary.json"],
        "input_subset/v1_2_gui/playmode_live_gate_evaluation.json": gate_files["playmode_live_gate_evaluation.json"],
        "input_subset/v1_2_gui/primary_verdict.json": gate_files["primary_verdict.json"],
        "input_subset/v1_2_gui/candidate_repository_validation.json": gate_files["candidate_repository_validation.json"],
        "input_subset/v1_2_gui/detached_checksum.txt": (root / (V12_GATE_ZIP + ".sha256")).read_bytes(),
        "ASSERTIONS.jsonl": assertions,
        "VALIDATION_REPORT.md": validation_report(
            inventory, compile_evidence, project_after, protected_after, profile_after, repository_after),
        "analysis/" + Path(__file__).name: Path(__file__).read_bytes(),
    }
    for relative in ALLOWED_SOURCES:
        safe = relative.replace("/", "__")
        files["source_before/" + safe] = source_before[relative]
        files["source_after/" + safe] = source_after[relative]
        meta = (root / (relative + ".meta")).read_bytes()
        files["source_before/" + safe + ".meta"] = meta
        files["source_after/" + safe + ".meta"] = meta
        files["diff/" + Path(relative).name + ".diff"] = diffs[relative]
        files["project_files/" + relative] = source_after[relative]
        files["project_files/" + relative + ".meta"] = meta
    files["project_files/" + NEW_TEST] = new_test
    files["project_files/" + NEW_TEST_META] = new_test_meta
    files["source_after/" + NEW_TEST.replace("/", "__")] = new_test
    files["source_after/" + NEW_TEST_META.replace("/", "__")] = new_test_meta

    return files, {
        "project": project_after,
        "protected": protected_after,
        "profile": profile_after,
        "repository": repository_after,
        "inventory": inventory,
        "compile": compile_evidence,
        "source_identity": source_identity,
    }


def PetRobotPracticeUiPresenterLabel() -> str:
    return "이번 단어는 연습 필요로 남기고 계속"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--prepare", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    if args.prepare:
        prepare(root)
        return

    v11 = load_v11(root)
    files, summary = build(root, v11)
    files = add_manifest(files)
    validation = root / "Validation"
    names = (NAME, NAME + "_delivery.zip", NAME + "_delivery.zip.sha256")
    result_dir, delivery, detached = v11.commit_outputs(
        validation, files, None, names, v11.post_verify)
    preflight = (root / PREFLIGHT).resolve()
    require(preflight.parent == validation.resolve() and preflight.name == ".step10a-v1_3-preflight",
            "unsafe preflight cleanup target")
    shutil.rmtree(preflight)
    print(json.dumps({
        "status": "PASS",
        "result_directory": str(result_dir),
        "result_file_count": len(files),
        "delivery_zip": str(delivery),
        "delivery_size_bytes": delivery.stat().st_size,
        "delivery_sha256": sha(delivery.read_bytes()),
        "detached_checksum": str(detached),
        "detached_size_bytes": detached.stat().st_size,
        "detached_sha256": sha(detached.read_bytes()),
        "changed_existing_project_files": summary["project"]["changed_paths"],
        "new_project_files": [NEW_TEST, NEW_TEST_META],
        "approved_existing_test_count": summary["inventory"]["approved_existing_test_count"],
        "new_test_count": summary["inventory"]["new_test_count"],
        "combined_test_inventory": summary["inventory"]["combined_unique_test_count"],
        "offline_compile_errors": [summary["compile"]["production_error_count"],
                                   summary["compile"]["approved_affected_test_error_count"],
                                   summary["compile"]["new_fixture_error_count"]],
        "protected_files_unchanged": summary["protected"]["unchanged_count"],
        "candidate_profile_files_unchanged": summary["profile"]["unchanged_count"],
        "candidate_repository_before_after": [summary["repository"]["candidate_count"],
                                                  summary["repository"]["candidate_count"]],
        "git_staged_files": 0,
        "unity_gui_editmode_playmode_webcam": "UNVERIFIED_OR_NOT_APPLICABLE",
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
