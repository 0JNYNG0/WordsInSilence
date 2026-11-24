"""Independent saved-result scope/source/69-feature audit; no inference or merge.

Run only after root requests completed recovery output review. Does not import
producer or prior feature functions. The mathematical candidate acceptance
audit is owned by semantic_model and is deliberately separate.
"""
from pathlib import Path
from collections import Counter
import hashlib,json,math,datetime as dt
import numpy as np
HERE=Path(__file__).resolve().parent;U=HERE.parent
def read(p):return json.loads(Path(p).read_text(encoding="utf-8-sig"))
def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(1048576),b""):h.update(b)
 return h.hexdigest()
def load(p):
 with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
checks=Counter();failures=[]
def check(ok,name,detail=None):
 checks[name]+=1
 if not bool(ok):failures.append({"check":name,"detail":detail})
def same(a,b):return np.array_equal(a,b,equal_nan=True)
TRIPLETS=[(1,2,3),(2,3,4),(0,5,6),(5,6,7),(6,7,8),(0,9,10),(9,10,11),(10,11,12),(0,13,14),(13,14,15),(14,15,16),(0,17,18),(17,18,19),(18,19,20)]

def independent_features(xy,observed,usable,ts,frames):
 """Formula verification only; no original producer function or model."""
 xy=xy.astype(float);valid=observed&usable&np.isfinite(xy).all(2)
 n=len(xy);x=np.zeros((n,69),float);mask=np.zeros((n,69),bool)
 center=(xy[:,11]+xy[:,12])/2
 scale=np.hypot(xy[:,11,0]-xy[:,12,0],xy[:,11,1]-xy[:,12,1])
 body=valid[:,11]&valid[:,12]&np.isfinite(scale)&(scale>1e-8)
 norm=np.full_like(xy,np.nan);norm[body]=(xy[body]-center[body,None])/scale[body,None,None]
 def put(c,val,ok):
  good=np.asarray(ok)&np.isfinite(val);mask[:,c]=good;x[good,c]=val[good]
 def angle(ids):
  a,b,c=ids;u=xy[:,a]-xy[:,b];v=xy[:,c]-xy[:,b]
  lengths=np.hypot(u[:,0],u[:,1])*np.hypot(v[:,0],v[:,1])
  ok=valid[:,ids].all(1)&np.isfinite(lengths)&(lengths>1e-12)
  theta=np.arctan2(abs(u[:,0]*v[:,1]-u[:,1]*v[:,0]),u[:,0]*v[:,0]+u[:,1]*v[:,1])/math.pi
  return theta,ok
 for off,col,sh,el,pw in [(33,0,11,13,15),(54,28,12,14,16)]:
  for k,j in enumerate([off,pw,el,off+8]):
   for d in range(2):put(col+2*k+d,norm[:,j,d],body&valid[:,j])
  palm_ids=[off+i for i in [0,5,9,13,17]];palm=np.mean(norm[:,palm_ids],1)
  for d in range(2):put(col+8+d,palm[:,d],body&valid[:,palm_ids].all(1))
  val,ok=angle([sh,el,pw]);put(col+10,val,ok)
  for k,tri in enumerate(TRIPLETS):
   val,ok=angle([off+t for t in tri]);put(col+11+k,val,ok)
  vec=xy[:,off+9]-xy[:,off];length=np.hypot(vec[:,0],vec[:,1])
  ok=valid[:,[off,off+9]].all(1)&np.isfinite(length)&(length>1e-8)
  for d in range(2):
   val=np.divide(vec[:,d],length,out=np.full(n,np.nan),where=ok);put(col+25+d,val,ok)
  val=np.hypot(*(norm[:,off+5]-norm[:,off+17]).T);put(col+27,val,body&valid[:,[off+5,off+17]].all(1))
 delta=norm[:,54]-norm[:,33];ok=body&valid[:,[33,54]].all(1)
 put(56,delta[:,0],ok);put(57,delta[:,1],ok);put(58,np.hypot(delta[:,0],delta[:,1]),ok)
 for col,off,pw in [(59,33,15),(60,54,16)]:
  v=norm[:,off]-norm[:,pw];put(col,np.hypot(v[:,0],v[:,1]),body&valid[:,[off,pw]].all(1))
 for col,off in [(61,33),(65,54)]:
  t=np.diff(ts)/1e6;cons=(np.diff(frames)==1)&(t>0)&(t<=.15)
  ok=np.r_[False,body[1:]&body[:-1]&valid[1:,off]&valid[:-1,off]&cons]
  velocity=np.full((n,2),np.nan)
  velocity[1:][ok[1:]]=(norm[1:,off]-norm[:-1,off])[ok[1:]]/t[ok[1:],None]
  for d in range(2):put(col+d,velocity[:,d],ok)
  speed=np.hypot(velocity[:,0],velocity[:,1]);moving=ok&np.isfinite(speed)&(speed>1e-8)
  for d in range(2):put(col+2+d,np.divide(velocity[:,d],speed,out=np.full(n,np.nan),where=moving),moving)
 return x.astype(np.float32),mask

def main():
 out=HERE/"RECOVERY_SOURCE_VERIFICATION.json"
 assert not out.exists(),"Preserve first verification result"
 prep=read(U/"RECOVERY_PREPARED.json");frozen=read(U/"RECOVERY_FROZEN.json");result=read(U/"RECOVERY_RESULT_MANIFEST.json")
 check(sha(U/"RECOVERY_FROZEN.json")==result["frozen_sha256"],"result_freeze_pin")
 for p,h in frozen["pins"].items():check(sha(p)==h,"frozen_pin",p)
 check(len(result["records"])==len(prep["clips"])==8,"eight_outputs")
 fixed={(r["alias"],r["frame_id"]):r for r in prep["opportunities"]}
 check(len(fixed)==303,"prepared303")
 allframes=[];totals=Counter();perclip=[];maxerror=0.
 for clip in prep["clips"]:
  alias=clip["alias"];r=next(r for r in result["records"] if r["alias"]==alias)
  check(sha(r["output"])==r["output_sha256"],"output_pin",alias);check(sha(r["ledger"])==r["ledger_sha256"],"ledger_pin",alias)
  z=load(r["output"]);ledger=read(r["ledger"]);old=load(clip["files"]["input"]);base=load(clip["files"]["native"])
  ix=np.asarray(clip["indices"]);fulln=len(base["frame_id"]);ns=len(ix)
  check(len(z["frame_id"])==r["frames"]==ns,"selected_count",alias)
  check(same(z["source_index"],ix),"selected_indices",alias)
  for key in ["frame_id","timestamp_us"]:check(same(z[key],base[key][ix]),"native_ids_pts",[alias,key])
  check(same(z["baseline_x"],old["x"][ix]) and same(z["baseline_mask"],old["mask"][ix]),"baseline_exact",alias)
  bm=old["mask"][ix];check(z["mask"][bm].all() and same(z["x"][bm],old["x"][ix][bm]),"every_old_supported_feature_exact",alias)
  check(same(z["newly_supported"],z["mask"]&~bm),"new_mask_exact",alias)
  for key in ["x","mask"]:check(z[key].shape==(ns,69),"full69shape",[alias,key])
  check(len(ledger["frames"])==ns,"ledger_all_selected",alias)
  fullxy=base["xy_px"].copy();fullobs=base["observed"].copy();fulluse=base["usable"].copy()
  for local,global_i in enumerate(ix):
   fid=int(z["frame_id"][local]);fr=ledger["frames"][local];p=fixed[(alias,fid)]
   allframes.append((alias,fid))
   for key in ["alias","source_index","frame_id","timestamp_us","eligibility","raw_count","baseline_sides","inference_required"]:
    check(fr[key]==p[key],"ledger_original_opportunity",[alias,fid,key])
   additions={a["side"]:a for a in fr["additions"]}
   check(len(additions)==len(fr["additions"]),"unique_added_side",[alias,fid])
   check(all(e["eligible"] for e in p["eligibility"] if e["side"] in additions),"add_only_eligible",[alias,fid])
   for key,src in [("native_xy_px","xy_px"),("native_observed","observed"),("native_usable","usable")]:
    check(same(z[key][local,:33],base[src][global_i,:33]),"all_pose_exact",[alias,fid,key])
    for side,off in [(0,33),(1,54)]:
     sl=slice(off,off+21)
     if side not in additions:
      check(same(z[key][local,sl],base[src][global_i,sl]),"all_unadded_side_exact",[alias,fid,side,key])
     else:check(not base["observed"][global_i,sl].any(),"no_overwrite_partial_or_existing",[alias,fid,side])
   for side in [0,1]:
    if side not in additions:check(same(z["hand_score"][local,side],base["hand_score"][global_i,side]),"old_score_exact",[alias,fid,side])
   if not p["inference_required"]:
    check(fr["full"]==fr["crop"]==fr["decisions"]==fr["additions"]==[],"skipped_no_detector_evidence",[alias,fid])
    check(fr["full_calls"]==fr["crop_calls"]==0,"skipped_zero_calls",[alias,fid])
   else:check(fr["status"]=="NEW_IMAGE_CANDIDATE_EXECUTED","executed_eligible",[alias,fid])
   totals["new_full_image_calls"]+=fr["full_calls"];totals["new_crop_image_calls"]+=fr["crop_calls"]
   totals["added_native_hand_frames"]+=len(additions)
   totals["added_roi_usable_joint_frames"]+=sum(a["roi_usable_points"] for a in additions.values())
  fullxy[ix]=z["native_xy_px"];fullobs[ix]=z["native_observed"];fulluse[ix]=z["native_usable"]
  box=clip["box"];inside=np.isfinite(fullxy).all(2)&(fullxy[:,:,0]>=box[0])&(fullxy[:,:,0]<box[2])&(fullxy[:,:,1]>=box[1])&(fullxy[:,:,1]<box[3])
  check(same(z["observed"],fullobs[ix]&inside[ix]),"roi_observed_exact",alias)
  check(same(z["usable"],fulluse[ix]&inside[ix]),"roi_usable_exact",alias)
  expectedxy=np.where((fullobs&inside)[ix,:,None],fullxy[ix],np.nan)
  check(same(z["xy_px"],expectedxy),"roi_xy_exact_no_outside_promotion",alias)
  ex,em=independent_features(fullxy,fullobs,fulluse&inside,base["timestamp_us"],base["frame_id"])
  check(same(z["mask"],em[ix]),"independent69mask",alias)
  delta=abs(z["x"].astype(float)-ex[ix].astype(float));maxerror=max(maxerror,float(delta.max()))
  check(np.allclose(z["x"],ex[ix],atol=2e-6,rtol=1e-6),"independent69values",[alias,float(delta.max())])
  check((z["x"][~z["mask"]]==0).all(),"unknown_stays_zero",alias)
  outside=np.ones(fulln,bool);outside[ix]=False
  discarded=int((em[outside]&~old["mask"][outside]).sum())
  check(discarded==ledger["outside_scope_derived_support_discarded"],"discarded_outside_derivative_count",alias)
  gain=int(z["newly_supported"].sum());preserved=int(bm.sum())
  check(gain==ledger["new_feature_cells"],"ledger_new_cells",alias)
  totals["new_supported_selected_feature_cells"]+=gain
  totals["preserved_original_selected_feature_cells"]+=preserved
  totals["out_of_scope_derived_support_discarded"]+=discarded
  perclip.append({"alias":alias,"frames":ns,"full69_cells":ns*69,"baseline_supported":preserved,"new_supported":gain,
                  "remaining_unknown":int((~z["mask"]).sum()),"max_abs_formula_error":float(delta.max())})
 check(len(allframes)==len(set(allframes))==303 and set(allframes)==set(fixed),"all303_exact")
 check(result["selected_frames"]==303 and result["executed_frames"]+result["skipped_frames"]==303,"manifest_denominator")
 check(result["executed_frames"]==sum(p["inference_required"] for p in fixed.values()),"executed_matches_prepared")
 for k,v in totals.items():check(result["counts"][k]==v,"manifest_count",[k,v,result["counts"].get(k)])
 output={"status":"PASS" if not failures else "FAIL","at_utc":dt.datetime.now(dt.timezone.utc).isoformat(),
   "checks":dict(checks),"check_total":sum(checks.values()),"failures":failures,"counts":dict(totals),
   "fixed_frames":303,"fixed_hand_opportunities":606,"fixed_feature_opportunities":303*69,"by_clip":perclip,
   "max_abs_formula_error":maxerror,"tolerance":{"atol":2e-6,"rtol":1e-6},
   "source_pins":{str(U/n):sha(U/n) for n in ["RECOVERY_PREPARED.json","RECOVERY_FROZEN.json","RECOVERY_RESULT_MANIFEST.json"]},
   "script_sha256":sha(__file__),"actions":{"detector_calls":0,"producer_imports":0,"new_alignment":0,"old_scores":0},
   "limitations":["Added support is not landmark, hand-identity or educational accuracy.",
     "Current detector raw/merge decision mathematical verification and RGB identity review are separate artifacts."]}
 out.write_text(json.dumps(output,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
 print(json.dumps({"status":output["status"],"checks":output["check_total"],"failures":len(failures),"sha256":sha(out)}))
 if failures:raise SystemExit(1)
if __name__=="__main__":main()
