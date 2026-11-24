"""Read saved inputs, masks, outputs and path pointers; no extractor/model/solver imports."""
from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime, timezone
import gzip
import hashlib
import json
import numpy as np

HERE = Path(__file__).resolve().parent
OPS = HERE.parent.parent
HOLD = OPS / "word100_hold_invariant_trajectory_20261101"
CONT = OPS / "word100_continuous_trajectory_20261101"
STAGE = OPS / "word100_stage_feedback_cycle_20261031"
TARGET = OPS / "word100_target_known_feedback_20261101"
TARGETS = ["R_index_only", "R_open_four", "L_open_four"]
pins = {}

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def pin(p, expected=None):
    p = str(Path(p).resolve())
    h = sha(p)
    assert expected is None or expected == h, p
    assert p not in pins or pins[p] == h, p
    pins[p] = h
    return h

def read(p):
    p = Path(p); b = p.read_bytes()
    return json.loads(gzip.decompress(b) if p.suffix == ".gz" else b.decode("utf-8-sig"))

def arrhash(a):
    a = np.ascontiguousarray(a)
    return dict(dtype=str(a.dtype), shape=list(a.shape),
                sha256=hashlib.sha256(a.tobytes(order="C")).hexdigest())

def save(name, d):
    with (HERE / name).open("x", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False, indent=2, allow_nan=False)

def main():
    for p in (HOLD/"TASKS.json", HOLD/"review/RESULT_VERIFICATION.json",
              CONT/"RESULT_MANIFEST.json", CONT/"review/RESULT_VERIFICATION.json",
              CONT/"review/FEATURE_PRESERVATION.json", CONT/"NEXT_PLAN_KO.txt",
              STAGE/"LEARNING_INPUTS.json", STAGE/"VISUAL_INDEX.json",
              TARGET/"REFERENCE_PHASES.json", TARGET/"REFERENCE_BANK.json",
              TARGET/"comparator.py", STAGE/"prepare_bundle.py",
              OPS/"word100_joint_stage_alignment_20261101/joint_alignment.py",
              OPS/"word100_alignment_missing_split_20261101/READOUT_ONLY_RESULT.json",
              OPS/"word100_alignment_missing_split_20261101/readout_only.py",
              OPS/"word100_guide_video_feedback_20261028/motion_observer.py",
              OPS/"word100_guide_video_feedback_20261028/GUIDE_CARDS.json",
              HERE/"RGB_CASE_SELECTION.json"):
        pin(p)
    tasks = read(HOLD/"TASKS.json")
    li = read(STAGE/"LEARNING_INPUTS.json")
    previous = read(CONT/"review/FEATURE_PRESERVATION.json")
    assert read(CONT/"review/RESULT_VERIFICATION.json")["status"] == "PASS"
    assert read(HOLD/"review/RESULT_VERIFICATION.json")["status"] == "PASS"
    rows = tasks["rows"]
    index = {r["alias"]: i for i, r in enumerate(li["rows"])}
    target_index = [li["targets"].index(t) for t in TARGETS]
    assert target_index == [0, 1, 2]
    names = rows["P033"]["feature_names"]
    assert len(names) == len(set(names)) == 69
    probabilities = {}
    for s in (42, 43, 44):
        p = STAGE/"runs"/f"pose_plus_signrep_{s}"/"probabilities.npy"
        pin(p, next(d["sha256"] for d in previous["preserved_native_probabilities"] if d["seed"] == s))
        probabilities[s] = np.load(p, allow_pickle=False, mmap_mode="r")
    clips = {}; clip_arrays = {}
    for alias, r in rows.items():
        assert r["feature_names"] == names
        pin(r["input"], r["input_sha256"])
        with np.load(r["input"], allow_pickle=False) as z:
            a = {k:z[k].copy() for k in ("x", "mask", "frame_id", "timestamp_us")}
        n = len(a["x"])
        assert a["x"].shape == a["mask"].shape == (n,69)
        assert a["mask"].dtype == np.bool_
        assert np.array_equal(a["frame_id"],np.arange(n)) and (np.diff(a["timestamp_us"])>0).all()
        assert np.isfinite(a["x"][a["mask"]]).all()
        support = np.empty((n,3),bool)
        for j, target in enumerate(TARGETS):
            side=target[0]; off=0 if side=="L" else 28
            support[:,j]=a["mask"][:,off:off+2].all(1)&(a["mask"][:,off+11:off+25].sum(1)>=10)
        a["support"]=support;clip_arrays[alias]=a
        pr=[]
        for s,p in probabilities.items():
            v=np.asarray(p[index[alias],:n])[:,target_index]
            assert v.shape==(n,3) and np.isfinite(v).all() and ((v>=0)&(v<=1)).all()
            pr.append(dict(seed=s,file=str((STAGE/"runs"/f"pose_plus_signrep_{s}"/"probabilities.npy").resolve()),
                           clip_axis_index=index[alias],native_frame_slice=[0,n],target_axis_indices=target_index,
                           values=arrhash(v),input_support=arrhash(support)))
        clips[alias]=dict(alias=alias,clip_id=r["clip_id"],word_id=r["word_id"],label=r["label"],
                         role=r["role"],person=r["person"],input=r["input"],input_sha256=r["input_sha256"],
                         native_frames=n,arrays={k:arrhash(v) for k,v in a.items()},
                         learned_shape_supported_cells_by_target=support.sum(0).tolist(),
                         native_probability_links=pr,source_video=r["source"],
                         source_video_sha256_as_previously_verified=r["source_sha256"])
    units=[]
    for i,name in enumerate(names):
        unit=("half_turn" if "_angle" in name else
              "unit_direction" if "_axis_" in name or "_motion_direction_" in name else
              "shoulder_width_per_second" if "_velocity_" in name else "shoulder_width")
        units.append(dict(index=i,name=name,unit=unit))
    cm=read(CONT/"RESULT_MANIFEST.json")
    continuous={}
    for item in cm["results"]:
        # Manifest items only contain id/path/hash; natural IDs start with natural_.
        if not item["id"].startswith("natural_"):continue
        pin(item["path"],item["sha256"]);d=read(item["path"])
        assert d["kind"]=="natural"
        continuous[d["query"],d["reference"],d["parent_id"],d["component"]]=item
    # Use source metadata as a fallback if the original frozen ID convention differs.
    if not continuous:
        inp=read(CONT/"INPUTS.json"); natural_ids={r["id"] for r in inp["records"] if r["kind"]=="natural"}
        for item in cm["results"]:
            if item["id"] not in natural_ids:continue
            pin(item["path"],item["sha256"]);d=read(item["path"])
            continuous[d["query"],d["reference"],d["parent_id"],d["component"]]=item
    assert len(continuous)==585
    grouped=defaultdict(list)
    for n in tasks["natural"]:grouped[n["query"],n["reference"]].append(n)
    assert len(grouped)==36 and len(tasks["natural"])==117
    records=[]; seen_readouts=set(); totals=Counter(); parent_seen={}
    for (q,r), ns in grouped.items():
        baseline=ns[0]["baseline"];pin(baseline);bd=read(baseline);path=bd["path"]
        rd=[]
        for p in ns[0]["source_readouts"]:
            pin(p);d=read(p);seen_readouts.add(p)
            assert d["query"]==q and d["reference"]==r and d["path"]==path
            assert d["source"]==baseline and d["source_sha256"]==pins[str(Path(baseline).resolve())]
            assert d["new_alignment"] is False and d["new_model_inference"] is False
            for axis,alias in (("query",q),("reference",r)):
                e=d["evidence"]["groups"]["learned3"][axis];a=clip_arrays[alias]
                assert e["dimension_labels"]==TARGETS
                assert e["frame_id"]==a["frame_id"].tolist() and e["timestamp_us"]==a["timestamp_us"].tolist()
                assert e["own_observed"]==a["support"].tolist()
            rd.append((p,d))
        assert sorted(d["seed"] for _,d in rd)==[42,43,44]
        for n in ns:
            parent=n["parent"];core=n["reference_range"];qr=n["query_range"]
            selected=[i for i,(_,ri) in enumerate(path) if core[0]<=ri<=core[1]]
            assert selected==list(range(min(selected),max(selected)+1))
            cut=path[min(selected):max(selected)+1]
            assert qr==[min(v[0] for v in cut),max(v[0] for v in cut)]
            assert sorted({v[0] for v in cut})==list(range(qr[0],qr[1]+1))
            assert sorted({v[1] for v in cut})==list(range(core[0],core[1]+1))
            assert parent["core"]==core and rows[r]["role"]=="fit"
            parent_seen[r,parent["segment_id"]]=parent
            seeds=[]
            for p,d in rd:
                axes={}
                for axis,alias,rrange in (("query",q,qr),("reference",r,core)):
                    lo,hi=rrange;e=d["evidence"]["groups"]["learned3"][axis]
                    counts=Counter(s for row in e["cell_status"][lo:hi+1] for s in row)
                    axes[axis]=dict(alias=alias,native_indices_inclusive=rrange,
                        frame_ids=e["frame_id"][lo:hi+1],timestamp_us=e["timestamp_us"][lo:hi+1],
                        possible_cells=(hi-lo+1)*3,
                        own_supported_cells=sum(sum(x) for x in e["own_observed"][lo:hi+1]),
                        cell_status_counts=dict(counts),
                        native_probability_pointer=dict(clip_axis_index=index[alias],
                            frame_axis_slice_python=[lo,hi+1],target_axis_indices=target_index),
                        readout_pointer=f"/evidence/groups/learned3/{axis}")
                    totals[axis+"_seed_feature_cells"]+=(hi-lo+1)*3
                seeds.append(dict(seed=d["seed"],readout=p,sha256=pins[str(Path(p).resolve())],axes=axes))
            records.append(dict(query=q,reference=r,role=n["role"],word_id=n["word_id"],
                parent_id=parent["segment_id"],query_range=qr,reference_range=core,
                baseline=baseline,baseline_path_rows_inclusive=[min(selected),max(selected)],
                reference_parent=parent,query_truth_used=False,seed_links=seeds,
                continuous_position_records=[continuous[q,r,parent["segment_id"],c] for c in tasks["component_names"]]))
    assert len(seen_readouts)==108 and len(parent_seen)==39
    cards=read(OPS/"word100_guide_video_feedback_20261028/GUIDE_CARDS.json")
    words={r["word_id"] for r in rows.values()}
    selected_cards=[c for c in cards["cards"] if c["word_id"] in words]
    assert len(selected_cards)==4
    guide=dict(schema="wis_saved_shape_guide_parent_links_v1",status="PASS_EXACT_SAVED_JOIN_NOT_SEMANTIC_TRUTH",
        targets=TARGETS,native_probability_targets=li["targets"],native_target_axis_indices=target_index,
        posterior_semantics="Three independent attribute probabilities; not a mutually exclusive handshape distribution. Low open-four is not a closed-fist label. No palm-front, rotation or wrist-bend output.",
        join_contract="LEARNING_INPUTS row order -> probabilities[clip_index, native_frame_index, target_index]. Frame IDs are verified equal to native ordinals. Readout groups contain residual/support, not raw posterior; preserve both via explicit pointers.",
        masks="Original detector support: hand-wrist XY observed AND >=10 of14 same-hand finger angles observed. Not RGB ground truth, not confidence threshold. Unobserved numeric posterior is not evidence.",
        reference_scope="Saved FIT parent core inclusive; qrange is all query indices in original global path rows with reference inside core. No query truth; keep every original path link and both full axes.",
        denominator=dict(pairs=36,parent_instances=117,unique_fit_parents=39,seeds=3,parent_seed_links=351,
            old_readouts=108,continuous_components=585,**dict(totals)),
        guide_card_status=cards["status"],guide_cards=selected_cards,
        guide_limits=["Source descriptions are expectations, not sample labels.",
            "reference_parent.source_links applies only to listed RGB anchors; no whole-parent automatic truth promotion.",
            "Existing parent descriptions are sampled physical drafts, not mandatory linguistic stages.",
            "Counterpart mapping is the saved algorithmic path, not a newly verified same-event correspondence.",
            "No confidence or metric is used to select parents, references, frames or candidates."],
        records=records)
    audit=dict(schema="wis_independent_hand_input_audit_v1",status="PASS_SOURCE_SCHEMA_SCOPE_JOIN",
        at_utc=datetime.now(timezone.utc).isoformat(),reviewer="Codex semantic_data_audit",
        scope=dict(words=4,clips=16,fit_clips=12,development_clips=4,native_frames=sum(c["native_frames"] for c in clips.values()),
            pairs=36,parent_instances=117,unique_fit_parents=39,parent_component_opportunities=585,
            natural_observed_complete=293,natural_unknown=292,old_readouts=108,seeds=[42,43,44]),
        feature_columns=units,clips=clips,
        coordinate_contract=dict(axes="Source non-mirrored projected XY; image y increases downward; L/R anatomical.",
            positions="(point - same-frame shoulder center) / same-frame shoulder width",
            angles="arccos(clipped dot/(length product))/pi; projected internal angle in half_turn [0,1]; not radians or joint-flexion labels.",
            palm_axis="Unit raw XY direction hand landmark0 ->9; not palm normal. L25/26,R53/54.",
            wrist_bend="No explicit stored wrist-bend angle; elbow_angle is at elbow. Wrist estimator discrepancy is diagnostic.",
            forearm_relation="Old derived along=dot(hand_wrist-other_elbow,other_pose_wrist-other_elbow)/length²; cross=2D determinant/length²; dimensionless. No contact/depth truth.",
            raw_vs_readout="Raw palm_axis unit vector is not divided by2. Old readout L_axis/R_axis divides by2; old derived forearm unit axis also /2. Do not confuse this cost scaling with stored raw units.",
            missing="Use mask AND finite; zero is a value. No interpolation, imputation, smoothing, label filling or inferred visibility.",
            derivatives="Stored translation velocity width/sec; separate normalized translation direction. No angular velocity.",
            shape_rotation="Fourteen finger angles are not wrist angles; rigid in-plane hand rotation may preserve them. Fist axial rotation can leave projected palm axis unchanged."),
        required_preservation=["Original69 values/mask/frame IDs/PTS remain exact; independent channel output appended.",
            "Native3 posterior values and detector support remain seed-specific, no re-inference or closed=1-open.",
            "Position completeness/zero cannot alter shape, axis, forearm relation or missing status.",
            "Direct3D wrist/fist rotation capability remains not assessed; technical transforms are not real rotation labels.",
            "Guide text and query stage truth cannot populate observations or choose runtime counterpart."],
        reusable_prior_feature_audit=str(CONT/"review/FEATURE_PRESERVATION.json"),
        actions=dict(model_inference=0,training=0,extraction=0,path_search=0,old_score_recalculation=0,
            original_writes=0,raw_video_decoding=0,private_access=0),
        pins=pins,protected=tasks["protected"],script_sha256=sha(__file__))
    for p,h in pins.items():assert sha(p)==h,p
    save("INPUT_AUDIT.json",audit);save("SHAPE_GUIDE_LINKS.json",guide)
    print(json.dumps(dict(status=audit["status"],pins=len(pins),scope=audit["scope"],links=guide["denominator"])))

if __name__=="__main__": main()
