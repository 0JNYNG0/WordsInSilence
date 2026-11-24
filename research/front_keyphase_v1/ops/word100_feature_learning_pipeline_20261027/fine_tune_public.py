"""Fixed last-stage adaptation of three public encoders; sealed test is unused.

Frozen eval-mode prefix outputs are cached once to avoid repeating an identical
calculation. Full-vs-split forward is verified before any fitting. No downloads.
"""
from pathlib import Path
import argparse
import gc
import hashlib
import json
import os
import time
import traceback
from datetime import datetime

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import encode_public as enc
from train_observer import metrics

U = Path(__file__).resolve().parent
PROTOCOL = U / "FINETUNE_PROTOCOL.json"
LOCK = U / "FINETUNE_RUNNING.json"
STATUS = U / "FINETUNE_STATUS.json"
FIRST_ERROR = U / "logs/FINETUNE_FIRST_ERROR.json"
NAMES = ["mobilenet_v3_small", "mobilenetv4_conv_small", "tinynext_t"]
SEEDS = [27101, 27102, 27103]

def read(p): return json.loads(Path(p).read_text(encoding="utf-8-sig"))
def sha(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""): h.update(block)
    return h.hexdigest()
def now(): return datetime.now().astimezone().isoformat()
def write(p, data):
    p = Path(p)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, p)
def tensor_hash(t):
    z = t.detach().cpu().contiguous()
    return hashlib.sha256(str(z.dtype).encode() + str(tuple(z.shape)).encode() + z.numpy().tobytes()).hexdigest()
def state_hashes(m): return {k:tensor_hash(v) for k,v in m.state_dict().items()}
def save_torch(p, x):
    tmp = p.with_name(p.name + ".tmp")
    torch.save(x, tmp)
    os.replace(tmp, p)
def pins():
    expected = read(U / "FROZEN_PREPARATION.json")["protected"]
    assert len(expected) == 32
    assert all(sha(p) == h for p,h in expected.items()), "Protected model/runtime changed"
    return expected

def configure(name, m):
    m.eval().requires_grad_(False)
    if name == "mobilenet_v3_small": modules = list(m.features)[9:]
    elif name == "mobilenetv4_conv_small": modules = [m.blocks[-1]]
    else: modules = [m.embeds[3], m.stages[3], m.norm]
    for module in modules: module.requires_grad_(True)
    m.eval()
    names = [k for k,p in m.named_parameters() if p.requires_grad]
    assert names and all(not x.training for x in m.modules())
    return names

def prefix(name, m, x):
    if name == "mobilenet_v3_small": return m.features[:9](x)
    if name == "mobilenetv4_conv_small": return m.blocks[:-1](m.bn1(m.conv_stem(x)))
    for i in range(3): x = m.stages[i](m.embeds[i](x))
    return x

def suffix(name, m, x):
    if name == "mobilenet_v3_small": return m.avgpool(m.features[9:](x)).flatten(1)
    if name == "mobilenetv4_conv_small": return m.blocks[-1](x).mean((-2,-1))
    return m.global_pool(m.norm(m.stages[3](m.embeds[3](x))))

def prepare():
    c = read(U / "EXPERIMENT.json")
    assert c["encoder_candidates"] == NAMES and c["seeds"] == SEEDS
    assert len(c["classes"]) == 45 and len(c["rows"]) == 3359
    rr = {role:[i for i,r in enumerate(c["rows"]) if r["role"] == role] for role in ("fit","validation","test")}
    assert [len(rr[k]) for k in rr] == [2693,284,382]
    assert not set(rr["fit"]) & set(rr["validation"]) and not set(rr["test"]) & (set(rr["fit"]) | set(rr["validation"]))
    active = rr["fit"] + rr["validation"]
    for i in active:
        r = c["rows"][i]
        assert sha(r["image"]) == r["image_sha256"], "Public fitting/validation source changed"
    old = read(PROTOCOL) if PROTOCOL.exists() else None
    original_heads = old["original_heads"] if old else {str(p):sha(p) for p in sorted((U / "models").glob("*.pt")) if p.name.startswith(("probe_","observer_"))}
    assert len(original_heads) == 24 and all(sha(p) == h for p,h in original_heads.items())
    architecture = {}
    for name in NAMES:
        m, full, path = enc.load_encoder(name)
        allowed = configure(name, m)
        sample = torch.stack([enc.crop_tensor(c["rows"][i]["image"]) for i in rr["fit"][:3]])
        sample = enc.normalize(sample, name)
        with torch.no_grad():
            mid = prefix(name,m,sample)
            a,b = full(sample),suffix(name,m,mid)
        err = float((a-b).abs().max())
        assert torch.equal(a,b), f"Full/split numerical discrepancy: {name} {err}"
        initial = state_hashes(m)
        architecture[name] = {"official_weight_path":str(path), "official_weight_sha256":sha(path), "trainable_names":allowed,
                              "trainable_parameters":sum(p.numel() for p in m.parameters() if p.requires_grad),
                              "total_parameters":sum(p.numel() for p in m.parameters()), "feature_dim":int(a.shape[1]),
                              "prefix_shape":list(mid.shape[1:]), "preflight_full_split_exact":True, "max_abs_error":err,
                              "frozen_state_names":[k for k in initial if k not in allowed], "initial_state_hashes":initial}
        del m,full,sample,mid,a,b
        gc.collect()
    protocol = {"schema":"wis_public_last_stage_finetune_v1", "scope":"External physical 45-class handshape adaptation, not KSL correctness",
                "experiment_sha256":sha(U / "EXPERIMENT.json"), "code_sha256":sha(Path(__file__)),
                "encode_code_sha256":sha(U / "encode_public.py"), "metrics_code_sha256":sha(U / "train_observer.py"),
                "architectures":architecture, "seeds":SEEDS, "models":NAMES,
                "split_counts":{k:len(v) for k,v in rr.items()}, "active_indices":active,
                "active_id_sha256":hashlib.sha256("\n".join(c["rows"][i]["id"] for i in active).encode()).hexdigest(),
                "epochs":20, "batch":32, "optimizer":"AdamW", "backbone_lr":.0002, "head_lr":.003,
                "weight_decay":.01, "loss":"unweighted cross entropy, all 45 original labels", "augmentation":False,
                "head":"pooled features L2 normalized, Linear(D,45), random initialization per seed",
                "whole_model_mode":"eval throughout; gradients only to listed last-stage parameters and new head; all buffers frozen",
                "selection_epoch":20, "best_epoch_selection":False,
                "geometry":c["input_geometry"], "device":"CPU", "threads":4, "deterministic_algorithms":True,
                "prefix_cache":"Only fit+validation float32, exact eval-mode frozen-prefix output, full/split identical on 3 fixed fit images",
                "test_prediction_count":0, "test_images_loaded":False,
                "model_selection":"Root compares 3-seed mean supported-class validation macro-F1; within 0.01 choose final full-forward CPU p95",
                "latency":"10 warmup + 40 single-crop full encoder/L2/new-head calls on first fit crop, 4 CPU threads; excludes image decoding/MediaPipe/DTW",
                "protected_before":pins(), "original_heads":original_heads}
    if old: assert old == protocol, "Frozen fine-tuning contract changed"
    else: write(PROTOCOL, protocol)
    return c,protocol,rr

def cached_prefix(name, c, p, images):
    path = U / "data" / ("ft_prefix_" + name + ".pt")
    meta = path.with_suffix(".json")
    if path.exists() or meta.exists():
        assert path.exists() and meta.exists(), "Partial prefix cache; preserve for recovery"
        q = read(meta)
        assert q["protocol_sha256"] == sha(PROTOCOL) and sha(path) == q["sha256"]
        payload = torch.load(path, map_location="cpu", weights_only=True)
        assert payload["indices"] == p["active_indices"]
        return payload["prefix"]
    m,full,source = enc.load_encoder(name)
    configure(name,m)
    values=[]
    with torch.no_grad():
        for start in range(0,len(images),32):
            x = enc.normalize(images[start:start+32],name)
            z = prefix(name,m,x)
            assert torch.isfinite(z).all()
            values.append(z.detach().clone())
            if start % 320 == 0:
                write(STATUS,{"state":"PREFIX_CACHE","model":name,"completed":start+len(z),"total":len(images),"at":now(),"pid":os.getpid()})
                print(name,"prefix",start+len(z),"/",len(images),flush=True)
    x = torch.cat(values).contiguous()
    save_torch(path,{"prefix":x,"indices":p["active_indices"],"protocol_sha256":sha(PROTOCOL)})
    write(meta,{"sha256":sha(path),"protocol_sha256":sha(PROTOCOL),"shape":list(x.shape),"dtype":str(x.dtype),"test_inputs":0})
    del m,full,values
    gc.collect()
    return x

def frozen_check(m, spec):
    after = state_hashes(m)
    bad = [k for k in spec["frozen_state_names"] if after[k] != spec["initial_state_hashes"][k]]
    assert not bad, "Frozen parameters/buffers changed: " + str(bad[:5])
    return {"all_frozen_equal":True,"frozen_state_count":len(spec["frozen_state_names"]),
            "initial":spec["initial_state_hashes"],"final":after,
            "changed_trainable":[k for k in spec["trainable_names"] if after[k] != spec["initial_state_hashes"][k]]}

def fit_one(name, seed, c, p, rr, x, first_image):
    stem = f"ft_{name}_{seed}"
    cp = U / "models" / (stem + ".pt")
    resultpath = U / "models" / (stem + ".json")
    valpath = U / "models" / (stem + "_validation.npz")
    partial = U / "models" / (stem + "_resume.pt")
    if resultpath.exists():
        out=read(resultpath)
        assert out["protocol_sha256"] == sha(PROTOCOL) and sha(cp) == out["checkpoint_sha256"] and sha(valpath) == out["validation_sha256"]
        print("reuse complete",stem,flush=True)
        return out
    m,full,official = enc.load_encoder(name)
    allowed = configure(name,m)
    spec = p["architectures"][name]
    assert allowed == spec["trainable_names"] and state_hashes(m) == spec["initial_state_hashes"]
    torch.manual_seed(seed)
    head = nn.Linear(spec["feature_dim"],45)
    head_initial=state_hashes(head)
    head.eval()
    opt=torch.optim.AdamW([{"params":[v for v in m.parameters() if v.requires_grad],"lr":p["backbone_lr"]},
                           {"params":list(head.parameters()),"lr":p["head_lr"]}],weight_decay=p["weight_decay"])
    generator=torch.Generator().manual_seed(seed)
    y=torch.tensor([c["rows"][i]["target"] for i in p["active_indices"]],dtype=torch.int64)
    nfit=len(rr["fit"])
    trace=[]
    start_epoch=0
    elapsed_saved=0.
    if partial.exists():
        saved=torch.load(partial,map_location="cpu",weights_only=True)
        assert saved["protocol_sha256"] == sha(PROTOCOL) and saved["seed"] == seed and saved["name"] == name
        m.load_state_dict(saved["model"],strict=True);head.load_state_dict(saved["head"],strict=True)
        opt.load_state_dict(saved["optimizer"]);generator.set_state(saved["generator_state"])
        start_epoch=saved["epoch"];trace=saved["trace"];elapsed_saved=saved["elapsed_training_seconds"]
        assert start_epoch == len(trace)
        frozen_check(m,spec)
    elif cp.exists() or valpath.exists():
        raise RuntimeError("Partial final artifact without resume; preserve and inspect")
    t0=time.perf_counter()
    for epoch in range(start_epoch,p["epochs"]):
        m.eval();head.eval()
        assert all(not mod.training for mod in m.modules())
        order=torch.randperm(nfit,generator=generator)
        total=0.;seen=0
        e0=time.perf_counter()
        for start in range(0,nfit,p["batch"]):
            ix=order[start:start+p["batch"]]
            opt.zero_grad(set_to_none=True)
            f=F.normalize(suffix(name,m,x[ix]),dim=1)
            logits=head(f)
            loss=F.cross_entropy(logits,y[ix])
            assert torch.isfinite(loss), "Non-finite fitting loss"
            loss.backward()
            assert all(v.grad is None for k,v in m.named_parameters() if k not in allowed)
            assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in m.parameters() if v.requires_grad)
            opt.step()
            total+=float(loss.detach())*len(ix);seen+=len(ix)
        assert seen == 2693 and len(torch.unique(order)) == 2693
        trace.append({"epoch":epoch+1,"fit_loss":total/seen,"seen_count":seen,"unique_image_count":int(len(torch.unique(order))),
                      "fit_order_indices_sha256":hashlib.sha256(order.numpy().tobytes()).hexdigest(),
                      "epoch_seconds":time.perf_counter()-e0,"test_predictions":0,"validation_predictions":0})
        frozen_check(m,spec)
        save_torch(partial,{"model":m.state_dict(),"head":head.state_dict(),"optimizer":opt.state_dict(),"generator_state":generator.get_state(),
                            "epoch":epoch+1,"trace":trace,"name":name,"seed":seed,"protocol_sha256":sha(PROTOCOL),
                            "elapsed_training_seconds":elapsed_saved+time.perf_counter()-t0})
        write(STATUS,{"state":"FITTING","model":name,"seed":seed,"epoch":epoch+1,"epochs":20,"fit_loss":total/seen,"at":now(),"pid":os.getpid(),"test_predictions":0})
        print(stem,"epoch",epoch+1,"loss",round(total/seen,5),"seconds",round(time.perf_counter()-e0,2),flush=True)
    model_check=frozen_check(m,spec)
    assert model_check["changed_trainable"], "No backbone parameter changed"
    with torch.inference_mode():
        validation=torch.cat([head(F.normalize(suffix(name,m,x[start:start+32]),dim=1)) for start in range(nfit,len(x),32)]).numpy()
        raw=enc.normalize(first_image,name)
        for _ in range(10): head(F.normalize(full(raw),dim=1))
        latencies=[]
        for _ in range(40):
            tt=time.perf_counter();head(F.normalize(full(raw),dim=1));latencies.append((time.perf_counter()-tt)*1000)
    assert validation.shape == (284,45) and np.isfinite(validation).all()
    # A final checkpoint and its predictions are each created once. Resume keeps
    # the fixed epoch20 weights; it never changes the chosen stopping epoch.
    if cp.exists():
        old=torch.load(cp,map_location="cpu",weights_only=True)
        assert old["protocol_sha256"] == sha(PROTOCOL)
        assert all(torch.equal(old["model"][k],v) for k,v in m.state_dict().items())
        assert all(torch.equal(old["head"][k],v) for k,v in head.state_dict().items())
    else:
        save_torch(cp,{"model":m.state_dict(),"head":head.state_dict(),"epoch":20,"model_name":name,"seed":seed,"protocol_sha256":sha(PROTOCOL),"test_used":False})
    if valpath.exists():
        with np.load(valpath,allow_pickle=False) as z: assert np.array_equal(z["logits"],validation)
    else: np.savez_compressed(valpath,indices=np.asarray(rr["validation"]),ids=np.asarray([c["rows"][i]["id"] for i in rr["validation"]]),logits=validation)
    assert sha(official) == spec["official_weight_sha256"]
    out={"name":name,"seed":seed,"protocol_sha256":sha(PROTOCOL),"checkpoint_sha256":sha(cp),"validation_sha256":sha(valpath),
         "epoch":20,"trace":trace,"fit_exposures":sum(t["seen_count"] for t in trace),"test_used":False,"test_predictions":0,
         "validation":metrics(y[nfit:].numpy(),validation,c["classes"]),"frozen_check":model_check,
         "head_initial_hashes":head_initial,"head_final_hashes":state_hashes(head),
         "elapsed_training_seconds":elapsed_saved+time.perf_counter()-t0,"official_weight_unchanged":True,
         "cpu_full_model_ms":{"p50":float(np.percentile(latencies,50)),"p95":float(np.percentile(latencies,95)),"samples":latencies,"threads":4,"batch":1},
         "limitations":["Validation uses two source signers and only supported classes in macro-F1.","External manual shape45 labels are not KSL execution correctness.","Latency excludes crop extraction/MediaPipe/DTW."]}
    write(resultpath,out)
    print("DONE",stem,json.dumps({k:out["validation"][k] for k in ["correct","n","accuracy","macro_f1_supported"]}),flush=True)
    del m,full,head,opt,y
    gc.collect()
    return out

def run(prepare_only=False):
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    assert not LOCK.exists(), "Existing FT execution lock; inspect actual session before duplicate execution"
    write(LOCK,{"pid":os.getpid(),"at":now(),"mode":"prepare" if prepare_only else "run"})
    try:
        c,p,rr=prepare()
        if prepare_only:
            print(json.dumps({"protocol":str(PROTOCOL),"status":"FROZEN_PREFLIGHT_PASS","test_predictions":0}),flush=True)
            return
        images=torch.stack([enc.crop_tensor(c["rows"][i]["image"]) for i in p["active_indices"]])
        outputs=[]
        for name in NAMES:
            x=cached_prefix(name,c,p,images)
            for seed in SEEDS: outputs.append(fit_one(name,seed,c,p,rr,x,images[:1]))
            del x
            gc.collect()
        assert pins() == p["protected_before"]
        assert all(sha(q) == h for q,h in p["original_heads"].items())
        groups=[]
        for name in NAMES:
            rows=[o for o in outputs if o["name"] == name]
            groups.append({"name":name,"seeds":SEEDS,"validation_accuracy_mean":float(np.mean([o["validation"]["accuracy"] for o in rows])),
                           "validation_macro_f1_mean":float(np.mean([o["validation"]["macro_f1_supported"] for o in rows])),
                           "validation_accuracy_range":[min(o["validation"]["accuracy"] for o in rows),max(o["validation"]["accuracy"] for o in rows)],
                           "validation_macro_f1_range":[min(o["validation"]["macro_f1_supported"] for o in rows),max(o["validation"]["macro_f1_supported"] for o in rows)],
                           "cpu_p95_mean_ms":float(np.mean([o["cpu_full_model_ms"]["p95"] for o in rows]))})
        write(U / "FINETUNE_COMPLETE.json",{"status":"NINE_FIXED_EPOCH20_RUNS_COMPLETE_TEST_UNSEEN","protocol_sha256":sha(PROTOCOL),"models":groups,
                                            "checkpoints":{str(U / "models" / f'ft_{o["name"]}_{o["seed"]}.pt'):o["checkpoint_sha256"] for o in outputs},
                                            "protected32_equal":True,"original24_heads_equal":True,"test_predictions":0,"at":now()})
        write(STATUS,{"state":"COMPLETE","runs":9,"at":now(),"test_predictions":0})
    except BaseException as e:
        record={"at":now(),"type":type(e).__name__,"message":str(e),"traceback":traceback.format_exc()}
        if not FIRST_ERROR.exists(): write(FIRST_ERROR,record)
        write(U / "logs" / ("FINETUNE_ERROR_" + str(time.time_ns()) + ".json"),record)
        write(STATUS,{"state":"ERROR","at":now(),"error":str(e),"pid":os.getpid()})
        raise
    finally:
        if LOCK.exists() and read(LOCK).get("pid") == os.getpid(): LOCK.unlink()

if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--prepare-only",action="store_true")
    args=parser.parse_args()
    run(args.prepare_only)
