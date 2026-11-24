"""Pre-score input/causality contracts and CPU timing after extraction workers finish."""
from pathlib import Path
import json,time,hashlib
import numpy as np
import torch
import train_observer as tr
import encode_public as enc

U=Path(__file__).resolve().parent
def main():
    c=tr.read(U/"EXPERIMENT.json");torch.set_num_threads(4)
    assert (U/"ENCODE_COMPLETE.json").exists()
    joint,seen=tr.joints(c);rgb=tr.embeddings(c["encoder_candidates"][0],c)
    checks={}
    checks["all_rows_and_finite"]=len(joint)==3359 and bool(torch.isfinite(joint).all())
    checks["missing_zero_and_explicit_mask"]=bool((joint[~seen]==0).all())
    for cond in c["modalities"]+["MASK_T"]:
        x=tr.contexts(c,rgb,joint,cond);m=tr.Observer(x.shape[-1],45,cond!="RJ_F").eval()
        with torch.inference_mode():
            out=m(x[:5]);checks[cond+"_finite_shape"]=out.shape==(5,45) and bool(torch.isfinite(out).all())
        indices=np.array([r["context_indices"] for r in c["rows"]]);missing=torch.tensor(indices<0)
        checks[cond+"_missing_does_not_reuse_first_row"]=bool((x[:,:,:-2][missing]==0).all())
        if cond=="RJ_F":
            altered=x[:5].clone();altered[:,:2]=torch.randn_like(altered[:,:2])*100;altered[:,3:]=torch.randn_like(altered[:,3:])*100
            with torch.inference_mode():checks["current_frame_has_no_context_access"]=bool(torch.equal(m(x[:5]),m(altered)))
    counts={k:len(v) for k,v in tr.roles(c).items()};checks["split_counts"]=counts=={"fit":2693,"validation":284,"test":382}
    # A physical prediction consumes numeric features only, not any target/name metadata.
    altered=json.loads(json.dumps(c))
    for r in altered["rows"]:r["target"]=-999;r["label"]="changed";r["signer"]="changed"
    checks["target_and_names_excluded_from_inputs"]=bool(torch.equal(tr.contexts(c,rgb,joint,"RJ_T"),tr.contexts(altered,rgb,joint,"RJ_T")))
    assert all(checks.values()),checks
    tr.save(U/"verification/OBSERVER_INPUT_CHECK.json",{"checks":checks,"status":"PASS","training_started":False,"joint_observation_count":int(seen.sum()),"contract_sha256":tr.sha(U/"EXPERIMENT.json")})
    tr.save(U/"MODEL_TRAINING_ADDENDUM.json",{"recorded_before_training":True,"scope":"External physical handshape learning only",
      "additional_diagnostic":"MASK_T, context availability and fixed time only, no RGB/joints/labels as input; same frozen fitting protocol",
      "purpose":"Detect manual sampling-pattern shortcut; not a fifth claimed observation method",
      "RJ_F_capacity":"128->384->128 current-frame MLP, almost equal parameters to two128->128 kernel3 convolutions; minor bias-count difference remains",
      "encoder_taps":"Last spatial features pooled: V3 576D,V4 960D,Tiny192D; not claims of each model's maximal final-embedding performance",
      "temporal_limit":"Manual adjacent source frame IDs at nominal25fps; no video PTS decoding, no complete continuous words or real occlusion ground truth",
      "test_not_used":True,"contract_sha256":tr.sha(U/"EXPERIMENT.json")})
    # A second timing is warranted: extraction-time timing had concurrent CPU work.
    row=next(r for r in c["rows"] if r["role"]=="fit")
    image=enc.crop_tensor(row["image"])[None]
    models={}
    for name in c["encoder_candidates"]:
        m,forward,_=enc.load_encoder(name);x=enc.normalize(image,name)
        with torch.inference_mode():
            for _ in range(20):forward(x)
            samples=[]
            for _ in range(120):
                t=time.perf_counter();forward(x);samples.append(1000*(time.perf_counter()-t))
        models[name]={"p50_ms":float(np.percentile(samples,50)),"p95_ms":float(np.percentile(samples,95)),"samples_ms":samples}
        del m
    tr.save(U/"CPU_LATENCY.json",{"models":models,"device":"CPU","threads":4,"input":"one224x224crop","scope":"warm encoder only, no video decode/MediaPipe/DTW; not webcam FPS","extraction_workers_complete":True})
    print(json.dumps({"checks":len(checks),"latency":{n:{k:v[k] for k in ["p50_ms","p95_ms"]} for n,v in models.items()}}))

if __name__=="__main__":main()
