"""Prepare full-native proposals from exact saved public input/result pairs only."""
from pathlib import Path
import json,hashlib,numpy as np
from process_auto_proposal import canonical_anchor,make_proposal
U=Path(__file__).resolve().parent;O=U.parent
P=O/"word100_learned_guide_feedback_20261102"
N=O/"word100_learned_motion_invariance_20261102"
K=O/"word100_coherent_sensor_input_20261102"
read=lambda p:json.loads(Path(p).read_text(encoding="utf-8-sig"))
def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for part in iter(lambda:f.read(1048576),b""):h.update(part)
 return h.hexdigest()
def main():
 out=U/"PROCESS_AUTO_NATIVE_MANIFEST.json";assert not out.exists()
 pins={}
 def pin(p,expected=None):
  p=str(Path(p).resolve());h=sha(p)
  if expected is not None:assert h==expected,p
  if p in pins:assert pins[p]==h
  pins[p]=h;return h
 for p in [Path(__file__),U/"process_auto_proposal.py",P/"GUIDE_SUPPORT.json",P/"INPUT_MANIFEST.json",P/"NORMALIZATION.json",P/"engine.py",P/"train.py",N/"INPUT_MANIFEST.json",N/"prepare_real_fit.py",K/"INPUT_MANIFEST.json",K/"prepare_coherent.py",U/"PROCESS_REVIEW_MANIFEST.json",U/"DEV_PROCESS_REVIEW_MANIFEST.json"]:pin(p)
 guides=read(P/"GUIDE_SUPPORT.json")["guides"]
 canonical=sorted([r for r in read(P/"INPUT_MANIFEST.json")["rows"] if r["role"]=="fit" and r["person"]==1],key=lambda r:r["word_id"])
 assert [r["word_id"] for r in canonical]==["WORD0739","WORD1133","WORD1567","WORD2523"]
 lengths=read(P/"NORMALIZATION.json")["prototype_lengths"];assert [r["frames"] for r in canonical]==lengths
 anchors={}
 for r,n in zip(canonical,lengths):
  pin(r["input"],r["input_sha256"])
  with np.load(r["input"],allow_pickle=False) as a:ids=a["frame_id"].copy()
  anchors[r["word_id"]]=canonical_anchor(guides,r["word_id"],ids,n)
 # Whitelist ONLY identity/query fields. No review bounds, point labels, intervals,
 # sequence links, or DEV observations enter proposal construction.
 identity=lambda r:{k:r[k] for k in ("id","clip_id","word_id","person","physical_query","source_sha256")}
 queries={"fit":[identity(r) for r in read(U/"PROCESS_REVIEW_MANIFEST.json")["rows"]],
          "development":[identity(r) for r in read(U/"DEV_PROCESS_REVIEW_MANIFEST.json")["rows"]]}
 full={"fit":[r for r in read(N/"INPUT_MANIFEST.json")["rows"] if r["role"]=="fit" and r["kind"]=="original" and r["person"] in range(1,10)],
       "development":read(K/"INPUT_MANIFEST.json")["rows"]}
 assert len(full["fit"])==36 and len(full["development"])==12
 rows=[];inputs=[]
 for role in ("fit","development"):
  people=set(range(1,10)) if role=="fit" else {11,12,13}
  for meta in full[role]:
   assert meta["person"] in people
   inp=meta["input"];ih=meta.get("input_sha256",meta.get("analysis_sha256"));pin(inp,ih)
   result=read(meta["result"]);rh=pin(meta["result"])
   assert result["word_id"]==meta["word_id"] and result["source_sha256"]==meta["source_sha256"]
   assert result["path_axes"]==["query_native_index","learned_canonical_index"]
   assert result["individual_reference_comparisons"]==0 and result["learned_prototype_comparisons"]==1
   assert result.get("reference_video_reads",0)==0
   assert result["actual_sensor_extent"]==[0,0,1920,1080]
   with np.load(inp,allow_pickle=False) as z:
    a={k:z[k].copy() for k in ("frame_id","timestamp_us","x","mask","rgb","image_available")}
   ids=a["frame_id"];ts=a["timestamp_us"];n=len(ids)
   assert result["frames"]==meta["frames"]==n
   assert a["x"].shape==a["mask"].shape==(n,69) and a["rgb"].shape==(n,768)
   assert a["image_available"].shape==(n,) and a["mask"].dtype==bool and a["image_available"].dtype==bool
   assert np.isfinite(a["x"][a["mask"]]).all() and np.isfinite(a["rgb"]).all()
   assert np.all(np.diff(ids)>0) and np.all(np.diff(ts)>0)
   assert np.allclose(result["overlay"]["times"],ts/1e6,rtol=0,atol=1e-9)
   proposal=make_proposal(result["path"],ids,ts,anchors[meta["word_id"]])
   item=dict(clip_id=meta["clip_id"],word_id=meta["word_id"],person=meta["person"],role=role,
     source=meta["source"],source_sha256=meta["source_sha256"],input=inp,input_sha256=ih,
     result=meta["result"],result_sha256=rh,frames=n,actual_sensor_extent=result["actual_sensor_extent"],
     existing_input_rgb_context="Frozen full-native RGB windows. Slice these rows for the new head; no re-extraction/copy from a separately edited view.",
     image_available_count=int(a["image_available"].sum()),frame_ids=ids.tolist(),timestamp_us=ts.tolist())
   inputs.append(item)
   selected=[q for q in queries[role] if q["clip_id"]==meta["clip_id"]]
   assert selected
   for q in selected:
    assert q["source_sha256"]==meta["source_sha256"] and q["word_id"]==meta["word_id"] and q["person"]==meta["person"]
    rows.append(dict(item,query_id=q["id"],physical_query=q["physical_query"],proposal=proposal))
 assert len(rows)==60 and sum(r["role"]=="development" for r in rows)==15
 for p,h in pins.items():assert sha(p)==h
 obj=dict(status="SAVED_FULL_NATIVE_INPUT_AND_SINGLE_PROPOSAL_BINDING_VERIFIED",rows=rows,unique_inputs=inputs,
   anchors=anchors,pins=pins,counts=dict(fit_queries=45,development_queries=15,fit_clips=36,development_clips=12,total_queries=60),
   source_axis="Explicit canonical source-frame to learned-prototype-row conversion with identity assertion; query coordinates are native-array indices.",
   manual_review_boundaries_used_for_proposals=False,labels_read_by_proposal_builder=False,
   proposal_coverage_is_not_completion=True,no_model_forward=True,new_DTW=0,old_scores_recomputed=0,
   full_native_proposals_scored=False,whole_goal_complete=False)
 out.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
 print(json.dumps(obj["counts"]))
if __name__=="__main__":main()

