"""Structural/hash verification only; not independent semantic correctness."""
from pathlib import Path
import json, hashlib
B=Path(__file__).resolve().parent
def read(p): return json.loads(Path(p).read_text(encoding="utf-8-sig"))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
r=read(B/"REVIEW_DEV_1567.json"); m=read(B/"DEV_PROCESS_REVIEW_MANIFEST.json")
assert sha(r["lineage"]["manifest_path"])==r["lineage"]["manifest_sha256"]
assert sha(r["lineage"]["phase_addendum_path"])==r["lineage"]["phase_addendum_sha256"]
selected={x["id"]:x for x in m["rows"] if x["word_id"]=="WORD1567"}
assert {x["id"] for x in r["rows"]}==set(selected)
page_checks=0; frame_checks=0; unique=set(); rel_count=0; interval_count=0; shared={}
for row in r["rows"]:
    src=selected[row["id"]]
    for k in ("clip_id","word_id","person","source_sha256","physical_query"): assert row[k]==src[k],(row["id"],k)
    assert row["person"] in (11,12,13)
    assert row["reviewed_pages"]==[dict(path=p["path"],sha256=p["sha256"]) for p in src["pages"]]
    for p in row["reviewed_pages"]:
        assert sha(p["path"])==p["sha256"];page_checks+=1
    assert row["reviewed_frames"]==src["frames"]
    assert row["reviewed_native_frames"]==[f["frame"] for f in src["frames"]]
    frames={f["frame"] for f in row["reviewed_frames"]}
    for f in row["reviewed_frames"]:
        assert sha(f["path"])==f["sha256"];frame_checks+=1;unique.add((row["clip_id"],f["frame"]))
    rise="rises" in row["physical_query"]
    for v in row["relations"]:
        assert v["frame"] in frames
        assert v["setup_relation"] in (0,1,None) and v["terminal_relation"] in (0,1,None)
        assert v["setup_relation"] is None or v["terminal_relation"]==1-v["setup_relation"]
        # Convert both query names to the same anatomical right-above predicate.
        above=v["terminal_relation"] if rise else v["setup_relation"]
        key=(row["clip_id"],v["frame"])
        if key in shared and above is not None and shared[key] is not None: assert above==shared[key],key
        if above is not None or key not in shared: shared[key]=above
        rel_count+=1
    for v in row["motion_intervals"]:
        assert v["start"]<v["end"] and v["label"] in (0,1,None)
        assert set(range(v["start"],v["end"]+1))<=frames
        interval_count+=1
    for seq in row["sequence_links"]:
        assert seq["visible_chain"]=="yes"
        assert set(seq["setup_frames"]+seq["terminal_frames"])<=frames
        assert min(seq["setup_frames"])<max(seq["terminal_frames"])
        assert any([x["start"],x["end"]]==seq["motion_interval"] and x["label"]==1 for x in row["motion_intervals"])
    assert row["whole_word_correctness"] is None and row["required_repetition_count"] is None and row["learner_error"] is None
assert len(r["rows"])==6 and page_checks==12
out=dict(status="PASS",scope="Schema, source/time/frame binding and hashes only; not independent physical annotation accuracy",annotation=str(B/"REVIEW_DEV_1567.json"),annotation_sha256=sha(B/"REVIEW_DEV_1567.json"),manifest_sha256=sha(B/"DEV_PROCESS_REVIEW_MANIFEST.json"),queries=6,source_clips=3,page_checks=page_checks,frame_checks=frame_checks,unique_source_frames=len(unique),relations=rel_count,motion_intervals=interval_count,source_frame_height_contradictions=0,new_inference=0,new_training=0)
p=B/"REVIEW_DEV_1567_VERIFIED.json"
assert not p.exists()
p.write_text(json.dumps(out,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
print(json.dumps(out,ensure_ascii=False))

