"""Static/metadata-only checks. Does not load model or numerical input arrays."""
from pathlib import Path
import ast, json, hashlib
B=Path(__file__).resolve().parent
read=lambda p:json.loads(Path(p).read_text(encoding="utf-8-sig"))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
for name in ("process_head.py","train_process_head.py"):ast.parse((B/name).read_text(encoding="utf-8-sig"))
inputs={r["pixel_view_key"]:r for r in read(B/"PROCESS_INPUT_MANIFEST.json")["rows"]}
targets=read(B/"PROCESS_VIEW_TARGETS.json")["rows"]
points=intervals=0
for r in targets:
    source=inputs[r["pixel_view_key"]];fm=r["ordered_frame_map"]
    assert source["role"]==r["role"]=="fit"
    assert source["clip_id"]==r["clip_id"] and source["person"]==r["recorded_person_id"]
    assert source["source_sha256"]==r["source_sha256"] and r["view_id"] in source["aliases"]
    assert [f["output_index"] for f in fm]==list(range(len(fm)))
    assert source["source_frame_indices"]==[f["source_native_frame"] for f in fm]
    assert source["source_timestamps_us"]==[f["source_native_timestamp_us"] for f in fm]
    assert source["view_timestamps_us"]==[f["view_timestamp_us"] for f in fm]
    assert source["image_available"]==[f["image_available"] for f in fm]
    for v in r["sparse_relation_targets"]:
        f=fm[v["output_index"]]
        assert f["source_native_frame"]==v["source_frame"] and f["image_available"]
        points+=1
    for v in r["fully_retained_motion_interval_targets"]:
        indices=v["output_indices"];native=v["source_frames"]
        assert len(indices)>=2 and indices==list(range(indices[0],indices[-1]+1))
        assert native==[fm[j]["source_native_frame"] for j in indices]
        assert native==list(range(native[0],native[-1]+1))
        assert all(fm[j]["image_available"] for j in indices)
        intervals+=1
result=dict(status="PASS_STATIC_AND_METADATA_ONLY",input_views=len(inputs),query_views=len(targets),points=points,intervals=intervals,syntax_parse=2,real_arrays_read=0,model_forwards=0,training=0,scope="Source/native-time/output-index binding over manifests; runtime NPZ assertions added but not executed here.",pins={str(B/n):sha(B/n) for n in ("process_head.py","train_process_head.py","PROCESS_INPUT_MANIFEST.json","PROCESS_VIEW_TARGETS.json")})
out=B/"PROCESS_TRAIN_BINDING_STATIC_VERIFIED.json"
assert not out.exists()
out.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
print(json.dumps(result))

