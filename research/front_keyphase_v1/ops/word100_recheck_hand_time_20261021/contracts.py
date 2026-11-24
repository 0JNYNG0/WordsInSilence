"""Adversarial same-hand correspondence controls on frozen public observations."""
import copy
import numpy as np
import study as d
f=d.read(d.U/"FROZEN.json");rows=[]
for r in f["rows"]:
    changes=d.read(d.U/"rechecked"/(r["clip_id"]+".json"))["changes"]
    if not changes:continue
    p=changes[0];i=p["index"]
    frame=d.read(r["detections"])["frames"][i];raw=d.load(r["raw"]);n=int(raw["raw_count"][i])
    sides=d.load(r["assignment"])["sides"][i,:n].tolist()
    args=(raw["hand_xy_px"][i,:n],sides,raw["pose_xy_px"][i],raw["pose_usable"][i],
          int(raw["width"][i]),int(raw["height"][i]),d.pipe.merge)
    baseline=d.candidate.existing_correspondence(frame["full"],*args)
    altered=copy.deepcopy(frame["full"])
    for x in altered:x["label"]="deliberately_wrong";x["score"]=0.
    assert d.candidate.existing_correspondence(altered,*args)==baseline
    duplicate=copy.deepcopy(frame["full"]);duplicate.append(copy.deepcopy(duplicate[p["full_index"]]))
    new=d.candidate.existing_correspondence(duplicate,*args)
    assert not any(x["accepted"] and x["old_index"]==p["old_index"] for x in new)
    perm=d.candidate.existing_correspondence(list(reversed(frame["full"])),*args)
    signature=lambda x: sorted((v["old_index"],v["side"],v["accepted"],v["reason"]) for v in x)
    assert signature(perm)==signature(baseline)
    rows.append(dict(clip_id=r["clip_id"],index=i,label_independent=True,
                     duplicate_blocks_confirmation=True,candidate_order_independent=True))
assert len(rows)==27
d.write(d.U/"CONTRACTS.json",dict(status="PASS",rows=rows,
    interpretation="Counterfactual proposal metadata/duplicate tests, not altered human sign or semantic annotation."))
print("CONTRACTS",len(rows))
