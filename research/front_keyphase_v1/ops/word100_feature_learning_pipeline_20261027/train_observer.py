"""Frozen external handshape comparison. Test logits are produced only in final mode."""
from pathlib import Path
import argparse, hashlib, json, os, time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

U = Path(__file__).resolve().parent
def read(p): return json.loads(p.read_text(encoding="utf-8-sig"))
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p, x):
    assert not p.exists(), str(p)
    p.write_text(json.dumps(x, ensure_ascii=False, indent=2), encoding="utf-8")

def metrics(y, logits, classes):
    pred = np.asarray(logits).argmax(1)
    y = np.asarray(y)
    confusion = np.zeros((len(classes),len(classes)),dtype=int)
    for a,b in zip(y,pred):confusion[a,b]+=1
    detail=[]
    for i,c in enumerate(classes):
        tp=int(confusion[i,i]);support=int(confusion[i].sum());predn=int(confusion[:,i].sum())
        precision=tp/predn if predn else 0.;recall=tp/support if support else 0.
        f1=2*precision*recall/(precision+recall) if precision+recall else 0.
        detail.append({"class":c,"support":support,"predicted":predn,"tp":tp,"precision":precision,"recall":recall,"f1":f1})
    return {"n":len(y),"correct":int((pred==y).sum()),"accuracy":float((pred==y).mean()),
            "macro_f1_supported":float(np.mean([r["f1"] for r in detail if r["support"]])),
            "supported_classes":sum(r["support"]>0 for r in detail),"per_class":detail,
            "confusion":confusion.tolist()}

def embeddings(name, config):
    p=U/"data"/(name+"_features.npz")
    z=np.load(p,allow_pickle=False)
    assert list(z["ids"])==[r["id"] for r in config["rows"]]
    x=torch.tensor(z["features"],dtype=torch.float32)
    assert torch.isfinite(x).all()
    return F.normalize(x,dim=-1)

def joints(config):
    p=U/"data/PUBLIC_JOINTS.npz";z=np.load(p,allow_pickle=False)
    index={str(k):i for i,k in enumerate(z["image_ids"])}
    ix=np.array([index[r["id"]] for r in config["rows"]])
    raw=z["raw_xyz"][ix].astype(np.float32)
    seen=z["mask"][ix].all(axis=1)
    rel=raw-raw[:,:1,:]
    # Within-hand scale only. Unobserved coordinates remain zero and explicitly masked.
    scale=np.linalg.norm(rel[:,[5,9,13,17]],axis=-1).max(axis=1)
    seen=seen & (scale>1e-6)
    rel=rel/np.maximum(scale[:,None,None],1e-6)
    rel[~seen]=0
    assert np.isfinite(rel).all()
    return torch.tensor(np.concatenate([rel.reshape(len(raw),63),seen[:,None]],axis=1)),seen

def roles(config):
    return {k:torch.tensor([i for i,r in enumerate(config["rows"]) if r["role"]==k]) for k in ("fit","validation","test")}

class Observer(nn.Module):
    def __init__(self, dim, nclasses, temporal):
        super().__init__();self.temporal=temporal
        self.project=nn.Sequential(nn.Linear(dim,128),nn.LayerNorm(128),nn.GELU())
        if temporal:
            self.sequence=nn.Sequential(nn.Conv1d(128,128,3,padding=1),nn.GELU(),nn.Conv1d(128,128,3,padding=1),nn.GELU())
        else:
            self.sequence=nn.Sequential(nn.Linear(128,384),nn.GELU(),nn.Linear(384,128),nn.GELU())
        self.output=nn.Linear(128,nclasses)
    def forward(self,x):
        if self.temporal:
            h=self.project(x).transpose(1,2);h=self.sequence(h)[:,:,2]
        else:h=self.sequence(self.project(x[:,2]))
        return self.output(h)

def contexts(config,rgb,joint,condition):
    ids=np.array([r["context_indices"] for r in config["rows"]])
    present=torch.tensor(ids>=0,dtype=torch.float32)
    safe=torch.tensor(np.maximum(ids,0))
    timevalues=torch.tensor(config["temporal"]["relative_seconds"],dtype=torch.float32)[None,:,None].expand(len(ids),-1,-1)
    if condition=="MASK_T":
        # Audit selection-pattern signal; no image or landmark values.
        values=torch.zeros((len(ids),5,0),dtype=torch.float32)
    else:
        parts=[]
        if condition in ("R_T","RJ_F","RJ_T"):parts.append(rgb)
        if condition in ("J_T","RJ_F","RJ_T"):parts.append(joint)
        values=torch.cat(parts,dim=-1)[safe]*present[:,:,None]
    return torch.cat([values,present[:,:,None],timevalues],dim=-1)

def train(model,x,y,fit,epochs,batch,lr,weight_decay,seed):
    generator=torch.Generator().manual_seed(seed)
    opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=weight_decay)
    trace=[]
    for epoch in range(epochs):
        model.train();order=fit[torch.randperm(len(fit),generator=generator)]
        total=0.;count=0
        for start in range(0,len(order),batch):
            ix=order[start:start+batch];opt.zero_grad(set_to_none=True)
            logits=model(x[ix]);loss=F.cross_entropy(logits,y[ix])
            if not torch.isfinite(loss):raise RuntimeError("non-finite loss")
            loss.backward();opt.step();total+=float(loss.detach())*len(ix);count+=len(ix)
        trace.append({"epoch":epoch+1,"train_loss":total/count,"seen_count":count})
    model.eval()
    return trace

def logits(model,x,indices):
    with torch.inference_mode():return torch.cat([model(x[ix]) for ix in indices.split(128)]).numpy()

def checkpoint(path,model,config,meta):
    assert not path.exists()
    torch.save({"state_dict":model.state_dict(),"contract_sha256":sha(U/"EXPERIMENT.json"),"meta":meta},path)

def probe(config):
    rr=roles(config);y=torch.tensor([r["target"] for r in config["rows"]]);spec=config["probe"]
    summaries=[]
    for name in config["encoder_candidates"]:
        x=embeddings(name,config)
        for seed in config["seeds"]:
            stem=f"probe_{name}_{seed}";result=U/"models"/(stem+".json")
            if result.exists():
                old=read(result);assert old["contract_sha256"]==sha(U/"EXPERIMENT.json")
                assert sha(U/"models"/(stem+".pt"))==old["checkpoint_sha256"]
                summaries.append(old);continue
            torch.manual_seed(seed);m=nn.Linear(x.shape[-1],len(config["classes"]))
            t=time.perf_counter();trace=train(m,x,y,rr["fit"],seed=seed,**{k:spec[k] for k in ["epochs","batch","lr","weight_decay"]})
            val=logits(m,x,rr["validation"])
            cp=U/"models"/(stem+".pt");checkpoint(cp,m,config,{"kind":"probe","name":name,"seed":seed,"dim":x.shape[-1]})
            np.savez_compressed(U/"models"/(stem+"_validation.npz"),indices=rr["validation"].numpy(),logits=val)
            result_data={"name":name,"seed":seed,"contract_sha256":sha(U/"EXPERIMENT.json"),"checkpoint_sha256":sha(cp),"trace":trace,
                         "validation":metrics(y[rr["validation"]].numpy(),val,config["classes"]),"seconds":time.perf_counter()-t,"test_used":False}
            save(result,result_data);summaries.append(result_data)
            print(stem,json.dumps({k:result_data["validation"][k] for k in ["correct","n","accuracy","macro_f1_supported"]}),flush=True)
    candidates=[]
    for name in config["encoder_candidates"]:
        group=[r for r in summaries if r["name"]==name]
        speed=read(U/"CPU_LATENCY.json")["models"][name]["p95_ms"]
        candidates.append({"name":name,"validation_macro_f1":float(np.mean([r["validation"]["macro_f1_supported"] for r in group])),
                           "validation_accuracy":float(np.mean([r["validation"]["accuracy"] for r in group])),"provisional_p95":speed})
    best=max(c["validation_macro_f1"] for c in candidates)
    eligible=[c for c in candidates if c["validation_macro_f1"]>=best-.01]
    chosen=min(eligible,key=lambda c:c["provisional_p95"])
    save(U/"ENCODER_SELECTION.json",{"selected":chosen["name"],"candidates":candidates,"eligible":eligible,"test_used":False,
                "scope":"This fixed source-domain probe only; not globally optimal or KSL validated","contract_sha256":sha(U/"EXPERIMENT.json")})
    print("selected",chosen["name"],flush=True)

def modalities(config):
    select=read(U/"ENCODER_SELECTION.json");rgb=embeddings(select["selected"],config);joint,seen=joints(config)
    rr=roles(config);y=torch.tensor([r["target"] for r in config["rows"]]);spec=config["temporal"]
    runs=[]
    for condition in config["modalities"]+["MASK_T"]:
        x=contexts(config,rgb,joint,condition)
        for seed in config["seeds"]:
            stem=f"observer_{condition}_{seed}";result=U/"models"/(stem+".json")
            if result.exists():
                old=read(result);assert old["contract_sha256"]==sha(U/"EXPERIMENT.json")
                assert sha(U/"models"/(stem+".pt"))==old["checkpoint_sha256"]
                runs.append(old);continue
            torch.manual_seed(seed);m=Observer(x.shape[-1],len(config["classes"]),condition!="RJ_F")
            t=time.perf_counter();trace=train(m,x,y,rr["fit"],seed=seed,**{k:spec[k] for k in ["epochs","batch","lr","weight_decay"]})
            val=logits(m,x,rr["validation"])
            cp=U/"models"/(stem+".pt");checkpoint(cp,m,config,{"kind":"observer","condition":condition,"seed":seed,"dim":x.shape[-1],"encoder":select["selected"]})
            np.savez_compressed(U/"models"/(stem+"_validation.npz"),indices=rr["validation"].numpy(),logits=val)
            result_data={"condition":condition,"seed":seed,"encoder":select["selected"],"contract_sha256":sha(U/"EXPERIMENT.json"),"checkpoint_sha256":sha(cp),"trace":trace,
                         "validation":metrics(y[rr["validation"]].numpy(),val,config["classes"]),"seconds":time.perf_counter()-t,"test_used":False}
            save(result,result_data);runs.append(result_data)
            print(stem,json.dumps({k:result_data["validation"][k] for k in ["correct","n","accuracy","macro_f1_supported"]}),flush=True)
    save(U/"TRAINING_FROZEN_BEFORE_TEST.json",{"contract_sha256":sha(U/"EXPERIMENT.json"),"encoder_selection_sha256":sha(U/"ENCODER_SELECTION.json"),
          "checkpoints":{str(p):sha(p) for p in sorted((U/"models").glob('*.pt'))},"conditions":config["modalities"]+["MASK_T"],"test_used":False})

def test(config):
    freeze=read(U/"TRAINING_FROZEN_BEFORE_TEST.json")
    assert freeze["contract_sha256"]==sha(U/"EXPERIMENT.json")
    assert freeze["encoder_selection_sha256"]==sha(U/"ENCODER_SELECTION.json")
    assert all(sha(Path(p))==h for p,h in freeze["checkpoints"].items())
    pinfile=U/"TEST_INPUT_PINS.json"
    pinpaths=[U/"EXPERIMENT.json",U/"MODEL_TRAINING_ADDENDUM.json",U/"CPU_LATENCY.json",U/"train_observer.py",
              U/"data/PUBLIC_JOINTS.npz",U/"data/DEEPHAND_INVENTORY.json"]
    pinpaths += [U/"data"/(n+"_features.npz") for n in config["encoder_candidates"]]
    current={str(p):sha(p) for p in pinpaths}
    if pinfile.exists():assert read(pinfile)["pins"]==current,"Test inputs/code changed"
    else:save(pinfile,{"pins":current,"test_predictions_started":False,"purpose":"freeze final numeric inputs and scoring implementation"})
    rr=roles(config);y=torch.tensor([r["target"] for r in config["rows"]]);select=read(U/"ENCODER_SELECTION.json")
    rgb=embeddings(select["selected"],config);joint,seen=joints(config);allresults=[]
    for filename in freeze["checkpoints"]:
        cp=Path(filename);stem=cp.stem;out=U/"models"/(stem+"_test.json")
        if out.exists():
            old=read(out);assert old["checkpoint_sha256"]==sha(cp);allresults.append(old);continue
        saved=torch.load(cp,map_location="cpu",weights_only=True);meta=saved["meta"]
        if meta["kind"]=="probe":x=embeddings(meta["name"],config);m=nn.Linear(meta["dim"],len(config["classes"]))
        else:x=contexts(config,rgb,joint,meta["condition"]);m=Observer(meta["dim"],len(config["classes"]),meta["condition"]!="RJ_F")
        m.load_state_dict(saved["state_dict"],strict=True);m.eval()
        numeric=U/"models"/(stem+"_test.npz")
        if numeric.exists():
            old=np.load(numeric,allow_pickle=False)
            assert str(old["checkpoint_sha256"])==sha(cp) and str(old["input_pins_sha256"])==sha(pinfile)
            assert np.array_equal(old["indices"],rr["test"].numpy())
            prediction=old["logits"]
            assert prediction.shape==(382,len(config["classes"])) and np.isfinite(prediction).all()
            print("recover saved test logits without inference",stem,flush=True)
        else:
            prediction=logits(m,x,rr["test"])
            np.savez_compressed(numeric,indices=rr["test"].numpy(),logits=prediction,
                                checkpoint_sha256=np.array(sha(cp)),input_pins_sha256=np.array(sha(pinfile)))
        detail={"stem":stem,"meta":meta,"checkpoint_sha256":sha(cp),"freeze_sha256":sha(U/"TRAINING_FROZEN_BEFORE_TEST.json"),
                "test":metrics(y[rr["test"]].numpy(),prediction,config["classes"]),"subgroups":{}}
        ii=rr["test"].numpy();truth=y[ii].numpy()
        tests={"detector_missing":~seen[ii],"detector_present":seen[ii]}
        for signer in config["summary"]["test"]["signers"]:tests[signer]=np.array([config["rows"][i]["signer"]==signer for i in ii])
        tests["near_label_change"]=np.array([any(j>=0 and config["rows"][j]["target"]!=config["rows"][i]["target"] for j in config["rows"][i]["context_indices"]) for i in ii])
        for key,mask in tests.items():
            if mask.any():detail["subgroups"][key]=metrics(truth[mask],prediction[mask],config["classes"])
        save(out,detail);allresults.append(detail)
        print(stem,"test",detail["test"]["correct"],"/",detail["test"]["n"],flush=True)
    save(U/"COMPARISON.json",{"results":allresults,"test_tuning":False,"contract_sha256":sha(U/"EXPERIMENT.json"),
                             "scope":"External physical45-class crop classification; no KSL correctness/occlusion/short-pulse validation"})

def main():
    args=argparse.ArgumentParser();args.add_argument("stage",choices=["probe","modalities","test"]);a=args.parse_args()
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    lock=U/"TRAINING_RUNNING.json"
    assert not lock.exists(),"Existing execution lock; inspect first"
    lock.write_text(json.dumps({"pid":os.getpid(),"stage":a.stage,"started":time.time()}))
    try:globals()[a.stage](read(U/"EXPERIMENT.json"))
    except BaseException as e:
        (U/"logs"/("TRAINING_ERROR_"+str(time.time_ns())+".json")).write_text(json.dumps({"stage":a.stage,"type":type(e).__name__,"message":str(e)}));raise
    finally:lock.unlink()

if __name__=="__main__":main()
