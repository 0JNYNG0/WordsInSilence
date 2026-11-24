"""Read-only, separately implemented audit of saved public static observations."""
from pathlib import Path
import hashlib
import json
from datetime import datetime
import numpy as np

U = Path(__file__).resolve().parents[1]
D = U / "data"

def sha(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()

def main():
    inventory = json.loads((D / "DEEPHAND_INVENTORY.json").read_text(encoding="utf-8"))
    protocol = json.loads((D / "PUBLIC_JOINTS_PROTOCOL.json").read_text(encoding="utf-8"))
    meta = json.loads((D / "PUBLIC_JOINTS_META.json").read_text(encoding="utf-8"))
    source = [json.loads(s) for s in (D / "PUBLIC_JOINTS_SOURCE.jsonl").read_text(encoding="utf-8").splitlines()]
    rows = inventory["rows"]
    checks, errors = {}, []
    def check(k, condition):
        checks[k] = bool(condition)
        if not condition:
            errors.append(k)
    check("rows3359_preserved", len(rows) == len(source) == 3359)
    for name, expected in [("PUBLIC_JOINTS.npz", meta["npz_sha256"]), ("PUBLIC_JOINTS_SOURCE.jsonl", meta["source_jsonl_sha256"]), ("PUBLIC_JOINTS_PROTOCOL.json", meta["protocol_sha256"]), ("DEEPHAND_INVENTORY.json", meta["inventory_sha256"])]:
        check("sha_" + name, sha(D / name) == expected)
    check("task_unchanged", sha(protocol["source_model"]) == protocol["model_sha256"])
    check("code_unchanged", sha(U / "extract_public_joints.py") == protocol["code_sha256"])
    check("all3359_input_image_hashes_unchanged", all(sha(r["image"]) == r["image_sha256"] for r in rows))
    check("protected32_unchanged", len(meta["protected_before"]) == 32 and all(sha(p) == h == meta["protected_after"][p] for p,h in meta["protected_before"].items()))
    with np.load(D / "PUBLIC_JOINTS.npz", allow_pickle=False) as a:
        check("ordered_ids_match", a["image_ids"].tolist() == [r["id"] for r in rows])
        check("selected_shape", a["raw_xyz"].shape == (3359, 21, 3))
        check("selected_mask_shape", a["mask"].shape == (3359, 21))
        check("all_candidate_shape", a["candidate_raw_xyz"].shape == (3359, 2, 21, 3))
        check("all_selected_finite", np.isfinite(a["raw_xyz"]).all())
        check("all_candidates_finite", np.isfinite(a["candidate_raw_xyz"]).all())
        check("absence_not_deleted", np.array_equal(a["mask"].any(1), a["candidate_count"] > 0))
        check("zero_absence_selected", np.count_nonzero(a["raw_xyz"][~a["mask"]]) == 0)
        check("zero_absence_candidates", np.count_nonzero(a["candidate_raw_xyz"][~a["candidate_mask"]]) == 0)
        row_checks = dict(source_order=True, candidate_count=True, candidate_packing=True, nearest_center_selection=True, ambiguity_flag=True, selected_packing=True, masks=True, handedness=True, dimensions=True, latency=True)
        for i,(r,s) in enumerate(zip(rows,source)):
            row_checks["source_order"] &= s["index"] == i and s["image_id"] == r["id"] and s["image_sha256"] == r["image_sha256"]
            cs=s["candidates"]
            row_checks["candidate_count"] &= len(cs) == s["candidate_count"] == int(a["candidate_count"][i])
            dd=[]
            for j,c in enumerate(cs):
                xyz=np.asarray(c["raw_xyz"], dtype=np.float32)
                palm=xyz[[0,5,9,13,17],:2].mean(0)
                dist=float(np.linalg.norm(palm - .5))
                dd.append(dist)
                row_checks["candidate_packing"] &= np.array_equal(xyz, a["candidate_raw_xyz"][i,j]) and np.allclose(palm,c["palm_xy"],rtol=0,atol=1e-7) and abs(dist - c["center_distance"]) <= 1e-7 and abs(dist - float(a["candidate_center_distance"][i,j])) <= 1e-7
                hs=c["handedness"]
                if hs:
                    row_checks["handedness"] &= a["candidate_handedness"][i,j] == (hs[0]["category_name"] or "") and abs(float(a["candidate_handedness_confidence"][i,j]) - hs[0]["score"]) <= 1e-7
            expected=min(range(len(dd)),key=lambda j:(dd[j],j)) if dd else -1
            row_checks["nearest_center_selection"] &= expected == s["selected_candidate"] == int(a["selected_candidate"][i])
            uncertain=len(dd)==2 and abs(dd[0]-dd[1]) <= .02
            row_checks["ambiguity_flag"] &= uncertain == s["identity_uncertain"] == bool(a["identity_uncertain"][i])
            if expected >= 0:
                row_checks["selected_packing"] &= np.array_equal(a["raw_xyz"][i], np.asarray(cs[expected]["raw_xyz"],np.float32))
            row_checks["masks"] &= bool(a["mask"][i].all()) == (len(cs)>0) and int(a["candidate_mask"][i].all(1).sum()) == len(cs)
            row_checks["dimensions"] &= a["image_size_wh"][i].tolist() == s["image_size_wh"] == [r["width"],r["height"]]
            row_checks["latency"] &= float(a["inference_seconds"][i]) == s["inference_seconds"] and s["inference_seconds"] >= 0
        for k,v in row_checks.items(): check(k,v)
        counts={"rows":len(rows),"no_hand":int((a["candidate_count"]==0).sum()),"one_candidate":int((a["candidate_count"]==1).sum()),"two_candidates":int((a["candidate_count"]==2).sum()),"identity_uncertain":int(a["identity_uncertain"].sum())}
        check("meta_counts", all(counts[k] == meta[k] for k in counts))
    result={"status":"PASS" if not errors else "FAIL", "at":datetime.now().astimezone().isoformat(), "checks":checks,"counts":counts,"errors":errors,"inference_rerun":False,"verifier_kind":"separately implemented arithmetic and provenance checks by same Codex agent; not independent human accuracy annotation", "limitations":["All 21 flags of a detected hand mean model output exists, not that each joint is visible or correct.","The 164 absent hands are retained as zero-plus-mask and remain in experiment denominator.","IMAGE-mode external 92x132 crops are not continuous webcam tracking."]}
    out=U/"research/PUBLIC_JOINTS_INDEPENDENT_CHECK.json"
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({"status":result["status"],"checks":len(checks),"counts":counts,"errors":errors}))
    if errors: raise SystemExit(1)

if __name__ == "__main__": main()
