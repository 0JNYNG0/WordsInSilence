"""One isolated, fixed CPU latency replay; no accuracy reevaluation or writes to features."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import os
import platform
import time
import traceback

import numpy as np
import torch

from dino_cpu import load_teacher, extract
from cpu_shubert_adapter import CPUShubertAdapter, CPUShubertConfig, stack_official_features

U = Path(__file__).resolve().parent
ALIASES = ["V01", "V07", "V12"]
RESULT = U / "LATENCY_ISOLATED.json"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def stamp():
    return datetime.now(timezone.utc).isoformat()


def write(data):
    tmp = RESULT.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(RESULT)


def main():
    with RESULT.open("x", encoding="utf-8") as f:
        json.dump({"status": "RUNNING", "pid": os.getpid(), "started_at": stamp(), "fixed_aliases": ALIASES}, f)
    try:
        torch.set_num_threads(4)
        existing = json.loads((U / "PUBLIC_FEATURE_RESULT.json").read_text(encoding="utf-8"))
        assert existing["clips"] == 12
        before = {str(p): sha(p) for alias in ALIASES for p in (U / "features" / alias).iterdir() if p.is_file()}
        sources = {str(p): sha(p) for p in [U/"dino_cpu.py",U/"cpu_shubert_adapter.py",U/"PUBLIC_FEATURE_RESULT.json",U/"inputs/MANIFEST.json"]}
        load_time = {}
        start = time.perf_counter()
        t = time.perf_counter();hands, hands_audit = load_teacher("hands");load_time["hands_teacher_seconds"] = time.perf_counter()-t
        t = time.perf_counter();face, face_audit = load_teacher("face");load_time["face_teacher_seconds"] = time.perf_counter()-t
        t = time.perf_counter()
        checkpoint = torch.load(U/"weights/checkpoint_836_400000.pt", map_location="cpu", weights_only=True, mmap=True)
        model = CPUShubertAdapter(CPUShubertConfig.from_mapping(checkpoint["cfg"]["model"]))
        load_audit = model.load_exact_state_dict(checkpoint["model"])
        model.eval().requires_grad_(False)
        del checkpoint
        load_time["shubert_checkpoint_and_model_seconds"] = time.perf_counter()-t
        load_time["combined_model_load_seconds"] = time.perf_counter()-start
        print(json.dumps({"model_load":load_time}),flush=True)
        rows=[]
        for alias in ALIASES:
            folder = U/"inputs"/alias
            timing={};streams={}
            start=time.perf_counter()
            for name, teacher in [("face",face),("left_hand",hands),("right_hand",hands)]:
                t=time.perf_counter()
                images=np.load(folder/f"{name}_rgb.npy",mmap_mode="r")
                streams[name]=extract(teacher,images,batch_size=8)
                timing[name+"_dino_seconds"]=time.perf_counter()-t
            t=time.perf_counter();streams["body_posture"]=np.load(folder/"body14.npy");timing["cached_body_read_seconds"]=time.perf_counter()-t
            t=time.perf_counter()
            source=[{k:torch.from_numpy(np.ascontiguousarray(v)) for k,v in streams.items()}]
            result=model.extract_features(source,mask=False)
            features=stack_official_features(result)[-1].cpu().numpy()
            timing["shubert_seconds"]=time.perf_counter()-t
            timing["cached_crops_to_features_seconds"]=time.perf_counter()-start
            timing["three_dino_streams_seconds"]=sum(timing[n+"_dino_seconds"] for n in ["face","left_hand","right_hand"])
            with np.load(U/"features"/alias/"streams.npz",allow_pickle=False) as saved:
                differences={k:float(np.max(np.abs(streams[k]-saved[k]))) for k in streams}
            saved_features=np.load(U/"features"/alias/"last_ffn.npy")
            feature_difference=float(np.max(np.abs(features-saved_features)))
            assert features.shape==saved_features.shape and np.isfinite(features).all()
            prior=next(r for r in existing["rows"] if r["alias"]==alias)
            row={"alias":alias,"frames":len(features),"duration_seconds":prior["duration_seconds"],
                 "timing":timing,"saved_stream_max_abs_difference":differences,
                 "saved_last_ffn_max_abs_difference":feature_difference,
                 "prior_possibly_overlapping_primary_seconds":prior["timing"]["primary_feature_seconds"],
                 "warm_model":True,"separate_warmup_before_clip":False}
            rows.append(row);print(json.dumps(row),flush=True)
        after={p:sha(p) for p in before}
        assert before==after,"Saved features changed"
        values=[r["timing"]["cached_crops_to_features_seconds"] for r in rows]
        output={"status":"COMPLETE_ISOLATED_CPU_LATENCY_ONLY","at":stamp(),"fixed_aliases":ALIASES,
            "cpu_threads":4,"batch_size":8,"device":"cpu","dtype":"float32","platform":platform.platform(),
            "torch_version":torch.__version__,"logical_cpus":os.cpu_count(),
            "coordination":"Parent suspended other agent CPU model inference while this fixed3clip replay ran; unrelated OS activity was not controlled.",
            "scope":"Cached224RGB crops and body14 -> three DINO streams -> SHuBERT finalFFN. NOT camera-to-feedback or end-to-end.",
            "included":["cached RGB memmap opening/reading","ImageNet RGB normalization","three DINO stream inference","cached body14 read","SHuBERT inference"],
            "excluded":["Python imports/process startup","original video decode","MediaPipe/face detection","crop preparation","DTW/reference search","feedback sentence generation","UI rendering","feature validation/hash checks outside measured region"],
            "load_policy":"Model loading measured separately once; OS disk cache may already be warm, not a cold-boot deployment guarantee.",
            "reuse_policy":"Reference embeddings can be precomputed; each new user recording still requires its own DINO features.",
            "model_load":load_time,"rows":rows,
            "cached_crops_to_features_seconds":{"min":min(values),"median":float(np.median(values)),"max":max(values)},
            "max_existing_feature_difference":max(r["saved_last_ffn_max_abs_difference"] for r in rows),
            "saved_features_unchanged":True,"accuracy_reevaluated":False,"training":False,"new_models_or_optimization":False,
            "source_pins":sources,"protected_feature_pins":before,"script_sha256":sha(__file__)}
        write(output)
    except BaseException:
        write({"status":"FAILED_ISOLATED_CPU_LATENCY","at":stamp(),"fixed_aliases":ALIASES,"traceback":traceback.format_exc()})
        raise


if __name__=="__main__":
    main()
