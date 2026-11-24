"""Fixed, label-independent image encoder comparison; no automatic downloads."""
from pathlib import Path
import sys, json, hashlib, time, os, gc
import numpy as np
import torch
from PIL import Image, ImageOps

U = Path(__file__).resolve().parent
ROOT = U.parents[3]
sys.path.insert(0, str(U / "deps"))
sys.path.insert(0, str(U / "models"))

def read(p): return json.loads(p.read_text(encoding="utf-8-sig"))
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()

def load_encoder(name):
    if name == "mobilenet_v3_small":
        from torchvision.models import mobilenet_v3_small
        m = mobilenet_v3_small(weights=None)
        p = ROOT / "research/front_keyphase_v1/assets/rgb_encoder/mobilenet_v3_small-047dcff4.pth"
        assert sha(p) == "047dcff4addef86ea5bc2eff13c9614dc11f47ab1160d0a71a25e7db994f4e1f"
        m.load_state_dict(torch.load(p, map_location="cpu", weights_only=True), strict=True)
        forward = lambda x: m.avgpool(m.features(x)).flatten(1)
    elif name == "mobilenetv4_conv_small":
        import timm
        m = timm.create_model(name, pretrained=False)
        p = U / "models/mobilenetv4_conv_small.bin"
        assert sha(p) == "28f2109728e2b5b4296397f0ea2aa8fd1896c8dca1e7f5ff217ae5ec22049e39"
        m.load_state_dict(torch.load(p, map_location="cpu", weights_only=True), strict=True)
        forward = lambda x: m.forward_features(x).mean((-2, -1))
    elif name == "tinynext_t":
        from tinynext_adapter import load_tinynext_t
        m = load_tinynext_t()
        p = U / "models/tinynext_t.pth"
        forward = m.forward_embedding
    else:
        raise ValueError(name)
    m.eval().requires_grad_(False)
    return m, forward, p

def crop_tensor(path):
    im = Image.open(path).convert("RGB")
    # Keep the entire crop. Official center-crop geometry could remove fingers.
    im = ImageOps.contain(im, (224,224), Image.Resampling.BICUBIC)
    canvas = Image.new("RGB", (224,224), (128,128,128))
    canvas.paste(im, ((224-im.width)//2, (224-im.height)//2))
    return torch.from_numpy(np.array(canvas, copy=True)).permute(2,0,1).float() / 255

def normalize(x, name):
    if name == "tinynext_t":
        from tinynext_adapter import normalize_rgb01
        return normalize_rgb01(x)
    mean = torch.tensor([.485,.456,.406])[None,:,None,None]
    std = torch.tensor([.229,.224,.225])[None,:,None,None]
    return (x-mean)/std

def run():
    contract = read(U / "EXPERIMENT.json")
    lock = U / "ENCODE_RUNNING.json"
    if lock.exists(): raise RuntimeError("Existing encoder execution; inspect rather than duplicate")
    lock.write_text(json.dumps({"pid":os.getpid(),"started":time.time()}))
    torch.set_num_threads(contract["encoder_threads"])
    try:
        rows = contract["rows"]
        for r in rows: assert sha(Path(r["image"])) == r["image_sha256"]
        images = torch.stack([crop_tensor(r["image"]) for r in rows]).to(torch.float32)
        # Fixed fit images for microbenchmark only; labels are never consulted.
        sample = images[[i for i,r in enumerate(rows) if r["role"] == "fit"][0]:][:1]
        for name in contract["encoder_candidates"]:
            output = U / "data" / (name + "_features.npz")
            meta = U / "data" / (name + "_ENCODER.json")
            if output.exists() or meta.exists():
                assert output.exists() and meta.exists(), "Partial encoder artifact; preserve and investigate"
                old = read(meta)
                assert old["contract_sha256"] == sha(U / "EXPERIMENT.json") and old["features_sha256"] == sha(output)
                print("reusing verified", name, flush=True)
                continue
            m, forward, checkpoint = load_encoder(name)
            start = time.perf_counter()
            all_features = []
            with torch.inference_mode():
                x = normalize(sample, name)
                for _ in range(10): forward(x)
                latencies = []
                for _ in range(40):
                    t = time.perf_counter(); y = forward(x);latencies.append((time.perf_counter()-t)*1000)
                assert y.ndim == 2 and torch.isfinite(y).all()
                for first in range(0,len(rows),contract["encoder_batch"]):
                    x = normalize(images[first:first+contract["encoder_batch"]], name)
                    f = forward(x).numpy().copy()
                    assert f.ndim == 2 and np.isfinite(f).all()
                    all_features.append(f)
                    if first % 320 == 0:
                        print(name, first, "/", len(rows), flush=True)
                        (U / "ENCODE_STATUS.json").write_text(json.dumps({"model":name,"completed":first+len(f),"total":len(rows),"updated":time.time()}))
            features = np.concatenate(all_features)
            np.savez_compressed(output, features=features, ids=np.array([r["id"] for r in rows]))
            result = {"model":name,"rows":len(rows),"feature_dim":features.shape[1],
                      "params":sum(p.numel() for p in m.parameters()),"device":"CPU",
                      "threads":torch.get_num_threads(),"batch":contract["encoder_batch"],
                      "single_crop224_ms_p50":float(np.percentile(latencies,50)),
                      "single_crop224_ms_p95":float(np.percentile(latencies,95)),
                      "latency_samples_ms":latencies,"elapsed_seconds":time.perf_counter()-start,
                      "geometry":contract["input_geometry"],
                      "latency_scope":"warm encoder only; excludes decoding, MediaPipe, DTW and two-hand cost",
                      "contract_sha256":sha(U/"EXPERIMENT.json"),"checkpoint_sha256":sha(checkpoint),
                      "features_sha256":sha(output),"labels_used_for_extraction":False,
                      "test_predictions":False,"encoder_changed":False}
            meta.write_text(json.dumps(result,indent=2),encoding="utf-8")
            print(json.dumps({k:result[k] for k in ["model","feature_dim","single_crop224_ms_p50","single_crop224_ms_p95","elapsed_seconds"]}),flush=True)
            del m, all_features, features;gc.collect()
        (U / "ENCODE_COMPLETE.json").write_text(json.dumps({"models":contract["encoder_candidates"],"contract_sha256":sha(U/"EXPERIMENT.json")}))
    except BaseException as e:
        (U / "logs" / ("ENCODE_ERROR_"+str(time.time_ns())+".json")).write_text(json.dumps({"type":type(e).__name__,"message":str(e)}))
        raise
    finally:
        lock.unlink()

if __name__ == "__main__": run()
