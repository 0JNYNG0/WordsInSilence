"""Independently recalculate new experiment outputs; never trains or reads heldout arrays."""
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re

OPS = Path(__file__).resolve().parent
WEB = OPS.parents[1]
RUN = WEB / "runs/native_data_pair_v1"


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def main():
    result, plan = read(RUN / "results.json"), read(OPS / "plan.json")
    base, cand = read(Path(plan["base_inputs"])), read(Path(plan["candidate_report"]))
    classes_by_index = {r["class_index"]: r["label"] for r in base["source_train"] + base["nikl"]}
    classes = [classes_by_index[i] for i in range(93)]
    targets = {r["label"] for r in base["nikl"]}
    gt = {r["id"]:r["label"] for r in base["source_validation"]}
    extra_gt = {r["clip_id"]:plan["label_mapping"][r["raw_label"]] for r in cand["rows"] if r["split_role"]=="development_only"}
    all_gt = dict(gt, **extra_gt)
    target_gt = {k:v for k,v in all_gt.items() if v in targets}
    expected_sets = {"full":all_gt,"source696":gt,"additional3":extra_gt,"target_nikl18":target_gt}
    checks, summaries, ck_reports = [], {}, {}

    def check(name, value):
        checks.append({"name":name,"passed":bool(value)})

    for arm in ("control","treatment"):
        a=result["arms"][arm]; summaries[arm]={}
        check(arm+": complete fixed schedule", a["uniforms_consumed"]==2*plan["settings"]["steps"]*plan["settings"]["batch_size"])
        check(arm+": final-step policy and no augmentation", a["train"]["steps"]==660 and not a["train"]["augmentation_examples"] and a["train"]["intermediate_checkpoints_kept"]==0 and a["train"]["distinct_nikl_clips_seen"]==0)
        check(arm+": finite losses", len(a["train"]["losses"])==660 and all(math.isfinite(v) for v in a["train"]["losses"]))
        for key,expected in expected_sets.items():
            ev=a["evaluations"][key]; rows=ev["rows"]
            got={r["id"]:r["true"] for r in rows}
            check(arm+":"+key+": exact evaluation IDs and GT", got==expected and len(rows)==len(expected))
            check(arm+":"+key+": development signers only", all(re.search(r'_REAL(11|12|13)_',r["id"]) for r in rows))
            check(arm+":"+key+": allowed predictions", all(r["pred"] in (targets if key=="target_nikl18" else classes) for r in rows))
            c=Counter(r["true"] for r in rows); pred=Counter(r["pred"] for r in rows)
            tp=Counter(r["true"] for r in rows if r["true"]==r["pred"])
            conf=Counter(r["true"]+">"+r["pred"] for r in rows)
            f1=[2*tp[label]/(c[label]+pred[label]) for label in c]
            macro=sum(f1)/len(f1); accuracy=sum(tp.values())/len(rows)
            recalls={label:tp[label]/c[label] if c[label] else None for label in classes}
            check(arm+":"+key+": independent metrics", ev["correct"]==sum(tp.values()) and ev["n"]==len(rows) and abs(ev["accuracy"]-accuracy)<1e-12 and abs(ev["macro_f1"]-macro)<1e-12 and ev["macro_f1_denominator"]==len(c) and ev["ground_truth_counts"]==dict(c) and ev["per_class_recall"]==recalls and ev["sparse_confusion"]==dict(conf))
            summaries[arm][key]={"n":len(rows),"correct":sum(tp.values()),"accuracy":accuracy,"macro_f1":macro,"gt_classes":len(c)}
        import torch
        ck_path=RUN/f"checkpoint_{arm}_final.pt"
        ck=torch.load(ck_path,map_location="cpu",weights_only=True)
        check(arm+": saved checkpoint configuration", ck["classes"]==classes and ck["step"]==660 and ck["augmentation"] is False and ck["parent_checkpoint_sha256"]==plan["checkpoint_sha256"] and ck["fixed_setting"]["batch"]=={"total":32,"source":32,"nikl":0})
        check(arm+": finite saved weights", all(torch.isfinite(v).all().item() for v in ck["state_dict"].values()))
        ck_reports[arm]={"path":str(ck_path),"bytes":ck_path.stat().st_size,"sha256":hashlib.sha256(ck_path.read_bytes()).hexdigest()}

    check("arms share every evaluation ID and true label", all([(r["id"],r["true"]) for r in result["arms"]["control"]["evaluations"][k]["rows"]]==[(r["id"],r["true"]) for r in result["arms"]["treatment"]["evaluations"][k]["rows"]] for k in expected_sets))
    check("all original execution pins unchanged", all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==sha for p,sha in read(OPS/"execution_ready.json")["frozen_execution_inputs"].items()))
    check("two distinct final model files", ck_reports["control"]["sha256"]!=ck_reports["treatment"]["sha256"])
    changes=[]
    ca=result["arms"]["control"]["evaluations"]["full"]["rows"]
    ta=result["arms"]["treatment"]["evaluations"]["full"]["rows"]
    for a,b in zip(ca,ta):
        if a["pred"]!=b["pred"]:
            changes.append({"id":a["id"],"true":a["true"],"control":a["pred"],"treatment":b["pred"]})
    report={"status":"PASS_OUTPUT_INTEGRITY" if all(x["passed"] for x in checks) else "FAIL", "passed":sum(x["passed"] for x in checks),"total":len(checks),"checks":checks,"metrics":summaries,"changed_full_predictions":changes,"checkpoints":ck_reports,"scope":"single-seed native source-domain data addition; development comparison, no final heldout/new-webcam-person claim","heldout_arrays_read":0,"source_arrays_reread_for_audit":0,"training_runs_in_audit":0}
    with (OPS/"native_pair_result_audit.json").open("x",encoding="utf-8") as f:json.dump(report,f,ensure_ascii=False,indent=2)
    print(json.dumps({k:report[k] for k in ("status","passed","total","metrics")},ensure_ascii=True))
    return int(report["status"]!="PASS_OUTPUT_INTEGRITY")


if __name__=="__main__":
    raise SystemExit(main())
