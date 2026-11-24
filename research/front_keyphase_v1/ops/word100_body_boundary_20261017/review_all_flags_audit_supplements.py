"""Audit supplemental viewed-image claims; does not perform semantic review."""
from pathlib import Path
import json, hashlib
from datetime import datetime, timezone
from collections import Counter
U=Path(__file__).resolve().parent
D=U/"review_all_flags_materials"
def load(p): return json.loads(Path(p).read_text(encoding="utf-8-sig"))
def digest(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def lines(p): return [json.loads(s) for s in Path(p).read_text(encoding="utf-8-sig").splitlines() if s.strip()]
audit=load(U/"review_all_flags_AUDIT_003.json")
assert audit["expected_window_conditions"]==838
assert not any(audit[k] for k in ("invalid_ledger_rows","material_errors","missing_condition_ids","incomplete_context_condition_ids","unresolved_condition_ids"))
known={c["condition_review_id"]:c for c in audit["conditions"]}
tailpath=U/"review_all_flags_batch1_tail_ledger.jsonl"
tail=lines(tailpath)
expected={}
for n in range(30,58):
    o=load(D/f"batch_1_overview_v3_{n:03d}.json")
    for row in o["rows"]: expected[row["condition_review_id"]]=row
assert len(expected)==138
assert len(tail)==len({r["condition_review_id"] for r in tail})==138
assert set(expected)=={r["condition_review_id"] for r in tail}
images={}
frames=set()
overlays=set()
for r in tail:
    assert r["batch"]==1 and r["reviewer"]=="roi_missing_protocol"
    assert r["unresolved_essential_geometric_cut"] is False and r["additional_frames_needed"] is False
    for im in r["viewed_images"]:
        assert digest(im["path"])==im["sha256"]
        images[im["path"]]=im["sha256"]
    frames.update((r["clip_id"],f) for f in r["actual_viewed_frames"])
    overlays.update((r["clip_id"],f) for f in r["actual_overlay_compared_frame_ids"])
result={
"status":"BATCH1_PAGES030_057_SAMPLED_VISUAL_REVIEW_COMPLETE",
"created_utc":datetime.now(timezone.utc).isoformat(),"reviewer":"roi_missing_protocol",
"batch":1,"pages":[30,57],"condition_count":len(tail),
"window_count":len({r["window_id"] for r in tail}),
"actually_viewed_overview_pages":28,"viewed_image_count":len(images),
"unique_clip_frame_count":len(frames),"unique_clip_overlay_frame_count":len(overlays),
"comparison_counts":dict(Counter(r["comparison"] for r in tail)),
"finding_counts":dict(Counter(r["finding"] for r in tail)),
"ledger":str(tailpath),"ledger_sha256":digest(tailpath),
"conditions":[r["condition_review_id"] for r in tail],
"viewed_images":[{"path":p,"sha256":h} for p,h in images.items()],
"new_essential_geometric_cut_observed_in_samples":0,
"unresolved_additional_frame_requests":0,
"training_allowed":False,"semantic_pass":False,
"limitations":["All assigned condition windows were inspected via explicit sparse source/overlay samples, not every frame.",
"Sensor observation failures remain risks; lack of observed geometric cutoff does not establish correct sign semantics or guaranteed training benefit."]}
with (U/"review_all_flags_batch1_tail_RESULT.json").open("x",encoding="utf-8") as f:json.dump(result,f,ensure_ascii=False,indent=2)
sources=[
(U/"review_all_flags_root_detail_supplements.jsonl",True),
(U/"review_all_flags_protocol_PARTIAL_DETAIL_45_48.json",False)]
checks=[];errors=[]
for source,is_jsonl in sources:
    rs=lines(source) if is_jsonl else load(source)["rows"]
    for r in rs:
        cid=r["condition_review_id"];wid=cid.split("_")[0]
        try:
            assert cid in known and known[cid]["represented"]
            mpath=D/(wid+".json");m=load(mpath)
            if "material_manifest_sha256" in r: assert digest(mpath)==r["material_manifest_sha256"]
            if "record_sha256" in r: assert m["record_sha256"]==r["record_sha256"]
            if "source_sha256" in r: assert m["source_sha256"]==r["source_sha256"]
            available={str(Path(i["path"]).resolve()):i for i in m["images"]}
            displayed=set()
            for im in r["viewed_images"]:
                match=available[str(Path(im["path"]).resolve())]
                assert digest(im["path"])==im["sha256"]==match["sha256"]
                assert im["displayed_frame_ids"]==match["displayed_frame_ids"]
                displayed.update(match["displayed_frame_ids"])
            assert set(r["actual_viewed_frames"])<=displayed
            if "actual_overlay_compared_frame_ids" in r: assert set(r["actual_overlay_compared_frame_ids"])<=set(r["actual_viewed_frames"])
            assert r.get("every_source_frame_viewed",False) is False
            assert r.get("all_window_frames_viewed",False) is False
            checks.append({"condition_review_id":cid,"source":str(source),"images":len(r["viewed_images"]),"frames":r["actual_viewed_frames"]})
        except Exception as e: errors.append({"condition_review_id":cid,"source":str(source),"error":repr(e)})
report={"status":"SUPPLEMENT_EVIDENCE_AUDIT_ONLY","created_utc":datetime.now(timezone.utc).isoformat(),
"input_pins":{str(p):digest(p) for p,_ in sources},
"main_audit_sha256":digest(U/"review_all_flags_AUDIT_003.json"),
"supplement_claims_checked":len(checks),"errors":errors,"checks":checks,
"training_allowed":False,"semantic_pass":False,"tool_sha256":digest(__file__)}
with (U/"review_all_flags_SUPPLEMENT_AUDIT.json").open("x",encoding="utf-8") as f:json.dump(report,f,ensure_ascii=False,indent=2)
print(json.dumps({"tail_conditions":len(tail),"tail_windows":result["window_count"],"tail_images":len(images),"supplements":len(checks),"errors":errors},ensure_ascii=False))

