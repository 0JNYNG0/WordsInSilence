"""Metadata/hash verification, not independent physical correctness."""
from pathlib import Path
import json,hashlib,collections
B=Path(__file__).resolve().parent
read=lambda p:json.loads(Path(p).read_text(encoding="utf-8-sig"))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
r=read(B/"REVIEW_DEV_PARTIAL_1567.json")
assert sha(r["manifest"]["path"])==r["manifest"]["sha256"]
m=read(r["manifest"]["path"])
selected={v["view_id"]:v for v in m["views"] if "WORD1567" in v["clip_id"] and v["construction_kind_audit_only"]!="native"}
assert {v["view_id"] for v in r["rows"]}==set(selected)
pages=set();slots=0;motions=0;positive=0
for row in r["rows"]:
 v=selected[row["view_id"]]; fm=v["ordered_frame_map"]
 for key in ("query_id","clip_id","recorded_person_id","physical_query","pixel_view_key","source_sha256"):assert row[key]==v[key],key
 assert row["recorded_person_id"] in (11,12,13)
 assert row["reviewed_pages"]==[dict(path=p["path"],sha256=p["sha256"]) for p in v["review_pages"]]
 for p in row["reviewed_pages"]:assert sha(p["path"])==p["sha256"];pages.add(p["path"])
 assert row["reviewed_output_indices"]==list(range(len(fm)))
 assert sorted(x for p in v["review_pages"] for x in p["output_indices"])==row["reviewed_output_indices"]
 assert sha(row["source_lineage"]["original_review_path"])==row["source_lineage"]["original_review_sha256"]
 for side in ("setup_witnesses","terminal_witnesses"):
  for w in row[side]:
   f=fm[w["output_index"]]
   assert f["source_native_frame"]==w["source_frame"] and f["image_available"]
 for w in row["motion_witnesses"]:
  ss=w["source_frames"];ii=w["output_indices"]
  assert w["label"]==1 and len(ss)>=2 and len(ss)==len(ii)
  assert ss==list(range(ss[0],ss[-1]+1)) and ii==list(range(ii[0],ii[-1]+1))
  assert ss==[fm[i]["source_native_frame"] for i in ii]
  assert all(fm[i]["image_available"] for i in ii)
  motions+=1
 y=row["observed_process_evidence"];assert y in (0,1,None)
 assert row["visible_chain"]==("yes" if y==1 else "uncertain" if y is None else "partial")
 if y==1:
  assert row["setup_witnesses"] and row["terminal_witnesses"] and row["motion_witnesses"]
  ids=[x["output_index"] for x in row["setup_witnesses"]+row["terminal_witnesses"]]
  a,b=min(ids),max(ids);ss=[f["source_native_frame"] for f in fm[a:b+1]]
  assert ss==list(range(ss[0],ss[-1]+1)) and all(f["image_available"] for f in fm[a:b+1])
  positive+=1
 assert row["whole_word_correctness"] is None and row["learner_error"] is None and row["required_repetitions"] is None
 slots+=len(fm)
assert set(r["actual_viewed_sheet_paths"])==pages and len(r["actual_viewed_sheet_paths"])==len(pages)==38
counts=collections.Counter(str(v["observed_process_evidence"]) for v in r["rows"])
assert dict(counts)=={"0":18,"1":6,"None":2}
out=dict(status="PASS",scope="Annotation schema, exact reviewed-sheet hashes, source/native-output-axis support, and contiguous positive support; not independent semantic label accuracy.",annotation=str(B/"REVIEW_DEV_PARTIAL_1567.json"),annotation_sha256=sha(B/"REVIEW_DEV_PARTIAL_1567.json"),manifest_sha256=sha(r["manifest"]["path"]),views=26,source_clips=3,queries=6,pages=len(pages),reviewed_output_slots=slots,label_counts=dict(counts),direct_motion_intervals=motions,positive_process_chains_crossing_seam_or_blackout=0,omitted_views=0,omitted_output_indices=0,new_training=0,new_inference=0)
p=B/"REVIEW_DEV_PARTIAL_1567_VERIFIED.json";assert not p.exists()
p.write_text(json.dumps(out,indent=2)+"\n",encoding="utf-8")
print(json.dumps(out))

