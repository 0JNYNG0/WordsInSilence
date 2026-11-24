import os
for k in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"):os.environ.setdefault(k,"1")
import sys,json,shutil,traceback,importlib.util
from pathlib import Path
from collections import Counter
import numpy as np
sys.dont_write_bytecode=True
U=Path(__file__).resolve().parent;O=U.parent;ROOT=O.parents[2]
R=O/"word100_recheck_hand_time_20261021"

def module(name,p):
    spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec)
    sys.modules[name]=m;spec.loader.exec_module(m);return m
prev=module("rest_alignment_prior",R/"study.py")
C=module("rest_alignment_candidate",U/"candidate.py")
read,load,write,sha,now=prev.read,prev.load,prev.write,prev.sha,prev.now
adapter=prev.adapter;F=prev.s.F
legacy=prev.s.oldrun.frame_cost_matrix;dp=prev.s.oldrun._optimal_path
CONDITIONS=("baseline","support_only","rest_alignment")


def freeze():
    old=read(R/"FROZEN.json");prev.pipe.check(old["protected"])
    assert not (R/"RUNNING.json").exists() and read(R/"SYNC_CHECK.json")["status"]=="PASS"
    rows=[r for r in old["rows"] if r["primary"]]
    pins={}
    for r in rows:
        for k in ("prepared","native","observation","quality","raw","assignment","detections"):
            pins[r[k]]=sha(r[k])
        for p in [R/"rechecked"/(r["clip_id"]+".npz"),R/"schedules"/(r["clip_id"]+"__source.npz")]:
            pins[str(p)]=sha(p)
    for folder,pattern in [("pairs","combined__*"),("schedules","*__NIA*"),("queries","combined__*")]:
        for p in (R/folder).glob(pattern):pins[str(p)]=sha(p)
    for p in [U/"candidate.py",U/"study.py",R/"candidate.py",R/"FINAL_AUDIT.json",
              R/"VERIFY.json",R/"FROZEN.json",ROOT/"research/front_keyphase_v1/src/wis_lab/core/dtw.py",
              ROOT/"research/front_keyphase_v1/src/wis_lab/core/features.py"]:
        pins[str(p)]=sha(p)
    (U/"before").mkdir()
    for p in [ROOT/"WordsInSilence_PROJECT_CONTEXT.md",ROOT/"NEW_CHAT_HANDOFF.md",O/"STATE.json"]:
        shutil.copyfile(p,U/"before"/p.name)
    write(U/"FROZEN.json",dict(at=now(),rows=rows,pins=pins,protected=old["protected"],roles=old["roles"],
        authorization="2026-10-21 user: reference per-hand lowering in temporal alignment; public comparison; no training before joint review.",
        conditions=CONDITIONS,
        scope="Same 3 words,9 TRAIN signers,27 public clips,216 leave-query-person-out pairs. Reuse158-preservation80-control evidence; no claim new100-word DTW test.",
        method="Reference-only frozen per-hand rest schedules. At certified rest columns zero shape/orient/handpos/facerel/motion/bend and bimanual weights. Arm anchors kept; unknown/active unchanged. Source coordinates, new angle quality, reference expected denominators frozen.",
        baseline="Reuse previous combined result, saved paths and global .35 evidence gate.",
        support_only="Old path; new reference-scheduled evidence fraction; exact same angle coverage denominator.",
        rest_alignment="New path from reference-scheduled costs; new scheduled evidence fraction. No query availability drives weights or expected range.",
        unchanged=dict(band=.2,one_sided_penalty=1.5,min_alignment=.35,min_coverage=.5,min_frames=8,min_people=5,
                       geometry=True,coordinate_values=True,assignment=True,ROI=True),
        verification="Independent per-group cost/evidence arithmetic and DP, allpair angle summary check; wholehand54 newpath deletions; fixed3frame core-window deletion27; no false whole-action pass. Reference denominator fixed. Offline source active/rest conflict diagnostics and largest path changes visual review.",
        visual_policy="Inspect same-source positive resting vs active witnesses for pairs with new conflicts and per-word largest path change; original images before overlaid coordinates. No post-outcome rule edits.",
        invariants=["No loss of reference expected time when query disappears","No point/identity/angle-state modification",
                    "Known crossing/identity80 controls preserved by input pins","No automatic correct/wrong sign verdict"],
        risks=["Rest column could attract an unrelated active query phase; measure rather than assume",
               "Global evidence can remain low because preparation/end outside annotation keep old rules",
               "Savedpoint counterfactuals are not new RGB occlusion or correctness labels"],
        forbidden=["training","classifier inference","user/mentor data","DEV/heldout","runtime model/web changes","camera","downloads","external transfer","Git"]))
    print("FROZEN27",flush=True)


def sample(row):
    z=load(row["prepared"]);z.update(load(R/"rechecked"/(row["clip_id"]+".npz")));return z


def pair_write(condition,pid,p,path,support,diag,expected_weight=None):
    arrays={k:p[k] for k in ("expected","raw","strict","difference_deg","path")}
    arrays.update({k:diag[k] for k in ("active_to_rest","rest_to_active")})
    if expected_weight is not None:arrays["expected_group_weight"]=expected_weight
    np.savez_compressed(U/"pairs"/(condition+"__"+pid+".npz"),**arrays)
    write(U/"pairs"/(condition+"__"+pid+".json"),dict(rows=p["rows"],alignment_support=support,
        **{k:v for k,v in diag.items() if k not in ("active_to_rest","rest_to_active")}))


def run():
    f=read(U/"FROZEN.json");prev.pipe.check(f["pins"]);prev.pipe.check(f["protected"])
    assert not (U/"RUNNING.json").exists()
    write(U/"RUNNING.json",dict(at=now(),pid=os.getpid()))
    for sub in ("pairs","queries"): (U/sub).mkdir()
    samples={r["clip_id"]:sample(r) for r in f["rows"]}
    sources={r["clip_id"]:load(R/"schedules"/(r["clip_id"]+"__source.npz")) for r in f["rows"]}
    summary={k:dict(total=Counter(),words={}) for k in CONDITIONS}; diagnostics=[]
    for qr in f["rows"]:
        cid=qr["clip_id"];q=samples[cid];required=f["roles"].get(qr["word_code"],{}).get("required_sides",["L","R"])
        refs=[r for r in f["rows"] if r["word_code"]==qr["word_code"] and r["signer_id"]!=qr["signer_id"]]
        pairs={k:[] for k in CONDITIONS}
        for rr in refs:
            rid=rr["clip_id"];r=samples[rid];pid=cid+"__"+rid
            sch=load(R/"schedules"/(pid+".npz"))
            old=load(R/"pairs"/("combined__"+pid+".npz"));oldj=read(R/"pairs"/("combined__"+pid+".json"))
            newpath,newsupport,mat=C.align(q,r,required,sch["rest"],F.GROUPS,legacy,dp)
            oldpath=old["path"];supportonly=float(mat["evidence"][tuple(oldpath.T)].mean())
            row=dict(query=cid,reference=rid,label=qr["label"],
                rest_frames=int(sch["rest"].sum()),rest_code_counts=dict(Counter(map(int,mat["rest_code"]))),
                old_support=oldj["alignment_support"],support_only=supportonly,new_support=newsupport,
                path_changed=not np.array_equal(oldpath,newpath),
                old_path_length=len(oldpath),new_path_length=len(newpath))
            for condition,path,support in [("baseline",oldpath,oldj["alignment_support"]),
                                           ("support_only",oldpath,supportonly),("rest_alignment",newpath,newsupport)]:
                if condition=="baseline":
                    p=dict(old,rows=oldj["rows"])
                else:
                    p=prev.candidate.pair(q,r,path,required,support>=.35,sch["expected"],adapter)
                diag=C.correspondence_diagnostics(q,r,path,sch["rest"],sch["expected"],sources[cid],sources[rid],adapter.cells)
                pair_write(condition,pid,p,path,support,diag,None if condition=="baseline" else mat["expected_group_weight"])
                pairs[condition].append(p)
                row[condition+"_active_to_rest_s"]=sum(diag["active_to_rest_seconds"])
                row[condition+"_rest_to_active_s"]=sum(diag["rest_to_active_seconds"])
            diagnostics.append(row)
        for condition in CONDITIONS:
            rows=prev.candidate.summarize(pairs[condition],required,adapter)
            write(U/"queries"/(condition+"__"+cid+".json"),dict(rows=rows,reference_people=[r["signer_id"] for r in refs]))
            for dest in [summary[condition]["total"],summary[condition]["words"].setdefault(qr["label"],Counter())]:
                dest.update(x["status"] for x in rows)
        print("COMPARED",cid,flush=True)
    write(U/"RESULT.json",dict(at=now(),comparison=summary,pairs=diagnostics,
        training=0,user_data=0,operational_changes=False))
    prev.pipe.check(f["pins"]);prev.pipe.check(f["protected"]);(U/"RUNNING.json").unlink()
    print("COMPLETE",summary,flush=True)


if __name__=="__main__":
    try:{"freeze":freeze,"run":run}[sys.argv[1]]()
    except BaseException:
        p=U/"FIRST_ERROR.txt"
        if not p.exists():p.write_text(traceback.format_exc(),encoding="utf8")
        raise
