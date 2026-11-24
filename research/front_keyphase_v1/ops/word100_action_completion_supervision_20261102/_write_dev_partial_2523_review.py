"""Serialize already-completed direct visual review; validate immutable provenance only.
No model calls, learned rules, inferred dense labels or relabeling by edit kind.
"""
import json, hashlib, sys
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter

sys.stdout.reconfigure(encoding="utf-8")
BASE=Path(__file__).resolve().parent
NOTES=json.loads(r'''[{"person":11,"kind":"prefix","value":0,"s":[50,54,57],"t":[],"motions":[[50,55,0]],"reason":"f46–57 shows raised hands mutually near, with minor posture change and no observed near-to-separated outward chain."},{"person":11,"kind":"suffix","value":0,"s":[],"t":[60,63,69,75],"motions":[[60,63,1],[66,75,0]],"reason":"f60 starts with raised hands already separated. A remaining outward tail is visible at f60–63, then separation is maintained and hands later lower. No mutually-near setup followed by a full observed outward chain survives in this fragment."},{"person":11,"kind":"middle_splice","value":0,"s":[50,54,57],"t":[60,63,69,75],"motions":[[50,55,0],[60,63,1],[66,75,0]],"seam":[57,60],"reason":"The left fragment f46–57 retains near hands; the right fragment f60–79 begins already separated and retains the outward tail f60–63. The connecting f58–59 images are absent across the seam; no other intact near-to-apart occurrence remains."},{"person":11,"kind":"speed_0_8","value":1,"s":[50,54,57],"t":[60,63,69,75],"motions":[[58,63,1],[66,75,0]],"reason":"Adjacent retained images show mutually-near raised hands, visible outward movement f58–63, and separated hands at f60 onward. Relation arrival and continued movement overlap naturally. This is a pixel/order judgment, not an actual speed-performance trial."},{"person":11,"kind":"speed_1_25","value":1,"s":[50,54,57],"t":[60,63,69,75],"motions":[[58,63,1],[66,75,0]],"reason":"The ordered native pixels are identical to the actually viewed speed0.8 material: near hands proceed through f58–63 outward movement to separated hands. Only time-map values differ; no actual speed-performance result is inferred."},{"person":11,"kind":"intact_context_trim","value":1,"s":[50,54,57],"t":[60,63,69,75],"motions":[[58,63,1],[66,75,0]],"reason":"The displayed f50–75 window retains the mutually-near arrangement, all intervening outward images f58–63, and separated positions. Removed context does not remove this directly visible physical chain."},{"person":11,"kind":"missing_rgb","value":null,"s":[50,54,57],"t":[64,69,75],"motions":[[50,55,0],[58,63,null],[66,75,0]],"reason":"Near hands remain through f57 and separated hands reappear at f64, but f58–63 is missing RGB. No other intact near-to-apart process appears in the visible fragments. The hidden transition remains unknown, not an incorrect action."},{"person":12,"kind":"prefix","value":0,"s":[45,50,54,55],"t":[],"motions":[[46,54,0]],"reason":"Every retained image f45–55 preserves a mutually-near raised hand arrangement. No near-to-separated outward process is observed within the shown fragment."},{"person":12,"kind":"suffix","value":0,"s":[],"t":[59,63,70,76],"motions":[[59,63,1],[66,76,0]],"reason":"The retained sequence begins f59 with visibly separated raised hands and continues a partial outward tail to f63. Later relation is held; no new mutually-near setup and outward chain is seen."},{"person":12,"kind":"middle_splice","value":0,"s":[45,50,54,55],"t":[59,63,70,76],"motions":[[46,54,0],[59,63,1],[66,76,0]],"seam":[55,59],"reason":"Near-hand images f45–55 are followed across a seam by already-separated f59. The visible outward tail f59–63 is preserved, but f56–58 connection is absent and there is no second intact near-to-apart chain."},{"person":13,"kind":"prefix","value":0,"s":[56,59,62],"t":[53],"motions":[[52,55,0],[56,61,0]],"reason":"Hands rise while separated at f53, then approach and stay mutually near at f56–62. This is separated-to-near, not the queried near-to-apart process. The early separated relation remains visible evidence without implying later arrival."},{"person":13,"kind":"suffix","value":0,"s":[],"t":[66,70,76,82],"motions":[[66,70,1],[74,80,0]],"reason":"f66 already has visibly separated raised hands. Their remaining outward movement through f70, held separation and later lowering are visible, but no mutually-near starting relation and complete outward connection survive."},{"person":13,"kind":"middle_splice","value":0,"s":[56,59,62],"t":[53,66,70,76,82],"motions":[[52,55,0],[56,61,0],[66,70,1],[74,80,0]],"seam":[62,66],"reason":"The first fragment contains early separated hands at f53 followed by approach and near hands f56–62. The second starts already separated at f66 with a remaining outward tail. The absent f63–65 seam connection cannot be supplied by endpoints; no other near-to-apart process survives."}]''')
OPENED=json.loads(r'''["W:\\WordsInSilence_derived\\word100_action_completion_supervision_20261102\\development_views\\28ab726f84516e0a6731_p1.jpg","W:\\WordsInSilence_derived\\word100_action_completion_supervision_20261102\\development_views\\d688feab66a6aa2e48df_p1.jpg","W:\\WordsInSilence_derived\\word100_action_completion_supervision_20261102\\development_views\\aa69de3a4d4578709d95_p1.jpg","W:\\WordsInSilence_derived\\word100_action_completion_supervision_20261102\\development_views\\aa69de3a4d4578709d95_p2.jpg","W:\\WordsInSilence_derived\\word100_action_completion_supervision_20261102\\development_views\\a2c8573c879583489244_p1.jpg","W:\\WordsInSilence_derived\\word100_action_completion_supervision_20261102\\development_views\\a2c8573c879583489244_p2.jpg","W:\\WordsInSilence_derived\\word100_action_completion_supervision_20261102\\development_views\\e93d0eb6923fa3495589_p1.jpg","W:\\WordsInSilence_derived\\word100_action_completion_supervision_20261102\\development_views\\e93d0eb6923fa3495589_p2.jpg","W:\\WordsInSilence_derived\\word100_action_completion_supervision_20261102\\development_views\\59ed4b9ffd3f6872d8e4_p1.jpg","W:\\WordsInSilence_derived\\word100_action_completion_supervision_20261102\\development_views\\59ed4b9ffd3f6872d8e4_p2.jpg","W:\\WordsInSilence_derived\\word100_action_completion_supervision_20261102\\development_views\\b0f1c222d461afbbae89_p1.jpg","W:\\WordsInSilence_derived\\word100_action_completion_supervision_20261102\\development_views\\fca61ace10ed2d11ac46_p1.jpg","W:\\WordsInSilence_derived\\word100_action_completion_supervision_20261102\\development_views\\890662e652e5b93305f9_p1.jpg","W:\\WordsInSilence_derived\\word100_action_completion_supervision_20261102\\development_views\\890662e652e5b93305f9_p2.jpg","W:\\WordsInSilence_derived\\word100_action_completion_supervision_20261102\\development_views\\cb36a53af9e4a44545b8_p1.jpg","W:\\WordsInSilence_derived\\word100_action_completion_supervision_20261102\\development_views\\09450f68837dbb07ef3b_p1.jpg","W:\\WordsInSilence_derived\\word100_action_completion_supervision_20261102\\development_views\\16b2a4afc08b0ff4f540_p1.jpg","W:\\WordsInSilence_derived\\word100_action_completion_supervision_20261102\\development_views\\16b2a4afc08b0ff4f540_p2.jpg"]''')
MANIFEST=BASE/"DEV_PARTIAL_VIEWS_MANIFEST.json"
def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""): h.update(block)
    return h.hexdigest()
def read(path): return json.loads(Path(path).read_text(encoding="utf-8-sig"))
manifest=read(MANIFEST)
views=[v for v in manifest["views"] if "WORD2523" in v["clip_id"] and v["construction_kind_audit_only"]!="native"]
assert len(views)==len(NOTES)==13
bykey={(v["recorded_person_id"],v["construction_kind_audit_only"]):v for v in views}
assert len(bykey)==13
checks=0
checked_pages={}
checked_frames={}
checked_sources={}
actual_seen=set()
rows=[]

def check(condition, message):
    global checks
    checks+=1
    if not condition: raise AssertionError(message)
def check_hash(path,expected,registry):
    if path not in registry: registry[path]=sha(path)
    check(registry[path]==expected, "SHA mismatch: "+path)

for note in NOTES:
    v=bykey[note["person"],note["kind"]]
    check(v["role"]=="development","role")
    check(v["recorded_person_id"] in (11,12,13),"person scope")
    check(v["physical_query"]=="hands_move_apart","query")
    fm=v["ordered_frame_map"]
    byframe={x["source_native_frame"]:x for x in fm}
    check(len(byframe)==len(fm),"duplicate source frame")
    check([x["output_index"] for x in fm]==list(range(len(fm))),"output indexing")
    check(all(a["view_timestamp_us"]<b["view_timestamp_us"] for a,b in zip(fm,fm[1:])),"view time monotonic")
    check_hash(v["source_path"],v["source_sha256"],checked_sources)
    for x in fm: check_hash(x["source_frame_path"],x["source_frame_sha256"],checked_frames)
    for p in v["review_pages"]:
        check_hash(p["path"],p["sha256"],checked_pages)
        check(p["source_native_frames"]==[fm[i]["source_native_frame"] for i in p["output_indices"]],"page map")
    check([i for p in v["review_pages"] for i in p["output_indices"]]==list(range(len(fm))),"all output cells covered")
    reuse=None
    pages=v["review_pages"]
    not_opened=[]
    if note["kind"]=="speed_1_25":
        ref=bykey[(note["person"],"speed_0_8")]
        fields=("output_index","source_native_frame","source_native_timestamp_us","source_frame_path","source_frame_sha256","pixel_operation","image_available")
        signature=lambda z:[[x[k] for k in fields] for x in z["ordered_frame_map"]]
        check(signature(v)==signature(ref),"speed ordered pixel/source hash identity")
        check(v["query_id"]==ref["query_id"],"speed query identity")
        check(v["source_sha256"]==ref["source_sha256"],"speed source identity")
        check([x["view_timestamp_us"] for x in fm]!=[x["view_timestamp_us"] for x in ref["ordered_frame_map"]],"time maps differ")
        pages=ref["review_pages"]
        not_opened=v["review_pages"]
        reuse={"from_view_id":ref["view_id"],"same_ordered_source_frames_and_hashes":True,
               "same_pixel_operations_and_availability":True,"same_physical_query":True,
               "different_view_timestamps":True,"visually_reviewed_reference_pages":pages,
               "own_artifact_pages_not_opened":not_opened,
               "scope":"Physical image/order observation reused only; timing overlays and actual retimed video performance not visually evaluated."}
    for p in pages:
        check(p["path"] in OPENED,"page not actually opened")
        actual_seen.add(p["path"])
    def point(frame):
        x=byframe[frame]
        check(x["image_available"] and x["pixel_operation"]=="copy","visible relation witness")
        return {"source_frame":frame,"output_index":x["output_index"],
                "source_native_timestamp_us":x["source_native_timestamp_us"],
                "view_timestamp_us":x["view_timestamp_us"],
                "source_frame_path":x["source_frame_path"],"source_frame_sha256":x["source_frame_sha256"]}
    setups=[point(f) for f in note["s"]]
    terminals=[point(f) for f in note["t"]]
    motions=[]
    for lo,hi,label in note["motions"]:
        frames=list(range(lo,hi+1))
        xs=[byframe[f] for f in frames]
        check(all(b["output_index"]==a["output_index"]+1 for a,b in zip(xs,xs[1:])),"motion output contiguity")
        if label is not None: check(all(x["image_available"] and x["pixel_operation"]=="copy" for x in xs),"known motion crosses blackout")
        motions.append({"source_frames":frames,"output_indices":[x["output_index"] for x in xs],
                        "source_native_timestamps_us":[x["source_native_timestamp_us"] for x in xs],
                        "view_timestamps_us":[x["view_timestamp_us"] for x in xs],
                        "source_frame_hashes":[x["source_frame_sha256"] for x in xs],
                        "image_available":[x["image_available"] for x in xs],
                        "label":label,"contiguous_native":True,
                        "scope":("Unobservable queried movement in masked images." if label is None else
                                 "Visible queried outward progress only; does not itself establish the whole process." if label==1 else
                                 "No queried outward progress in this observed interval; small sway or opposite approach is not globally forbidden."),
                        "interval_not_dense_frame_labels":True,"reason":note["reason"]})
    seams=[]
    if note.get("seam"):
        fs=note["seam"]; xs=[byframe[f] for f in fs]
        check(xs[1]["output_index"]==xs[0]["output_index"]+1 and fs[1]>fs[0]+1,"seam")
        seams=[{"source_frames":fs,"output_indices":[x["output_index"] for x in xs],"label":None,
                "source_native_timestamps_us":[x["source_native_timestamp_us"] for x in xs],
                "view_timestamps_us":[x["view_timestamp_us"] for x in xs],
                "reason":"Deleted native frames disconnect the two visible fragments. Endpoints do not supply observed motion.",
                "contiguous_native":False}]
        motions.extend(seams)
    # Split by actual adjacent pixels and missing RGB, not construction kind.
    fragments=[]
    for x in fm:
        if not fragments or x["image_available"]!=fragments[-1][-1]["image_available"] or x["source_native_frame"]!=fragments[-1][-1]["source_native_frame"]+1:
            fragments.append([])
        fragments[-1].append(x)
    fragment_rows=[]
    for group in fragments:
        fs=[x["source_native_frame"] for x in group]; os=[x["output_index"] for x in group]
        fragment_rows.append({"source_frames":fs,"output_indices":os,
                             "source_native_timestamps_us":[x["source_native_timestamp_us"] for x in group],
                             "view_timestamps_us":[x["view_timestamp_us"] for x in group],
                             "image_available":group[0]["image_available"],
                             "near_relation_source_frames":[f for f in note["s"] if f in fs],
                             "separated_relation_source_frames":[f for f in note["t"] if f in fs],
                             "contains_intact_query_chain":None if not group[0]["image_available"] else bool(note["value"]==1),
                             "note":note["reason"] if group[0]["image_available"] else "RGB unavailable here. Physical behavior is not inferred from neighboring frames."})
    rows.append({"view_id":v["view_id"],"query_id":v["query_id"],"clip_id":v["clip_id"],
                 "word_id":"WORD2523","physical_query":v["physical_query"],
                 "pixel_view_key":v["pixel_view_key"],"query_review_key":v["query_review_key"],
                 "person":note["person"],"recorded_person_id":note["person"],"role":"development",
                 "source_path":v["source_path"],"source_sha256":v["source_sha256"],
                 "construction_kind_audit_only":note["kind"],
                 "reviewed_pages":pages,"artifact_pages_verified_not_opened":not_opened,
                 "reviewed_output_indices":[x["output_index"] for x in fm],
                 "reviewed_native_frames":[x["source_native_frame"] for x in fm],
                 "reviewed_all_retained_frames":True,
                 "review_method":"same_ordered_pixels_verified_reuse" if reuse else "direct_view_image_of_every_retained_frame_sheet",
                 "content_reuse_audit":reuse,
                 "observed_process_evidence":note["value"],
                 "visible_chain":"yes" if note["value"]==1 else "partial" if note["value"]==0 else "uncertain",
                 "evidence_reason":note["reason"],"remaining_evidence_by_fragment":fragment_rows,
                 "setup_witnesses":setups,"motion_witnesses":motions,"terminal_witnesses":terminals,
                 "terminal_witness_scope":"Raised-separated physical relation only. An early separated state is not assumed to be arrival of the queried process.",
                 "seam_evidence":seams,"unknown_output_indices":[x["output_index"] for x in fm if not x["image_available"]],
                 "whole_word_correctness":None,"learner_error":None,"required_repetitions":None,
                 "required_extent":None,"word_completion":None,"source_performed_process_absent":None,
                 "exact_boundary_claim":False,"semantic_extent_or_sufficiency_claim":False,
                 "repeated_development":True,"independent_final_test":False,
                 "actual_retimed_video_viewed":False,"speed_performance_evaluated":False,
                 "prior_source_context_known":True,"model_predictions_seen_for_review":False,
                 "reviewer":"Codex /root/egypt_source_binding_review; physical observation annotation, not independent human/KSL-expert gold"})
check(actual_seen==set(OPENED),"actual opened pages exactly accounted")
check(len(OPENED)==len(set(OPENED)),"opened page duplicates")
counts=Counter(str(r["observed_process_evidence"]) for r in rows)
out={"status":"direct_visual_review_complete_with_provenance_self_checks",
     "at":datetime.now(timezone.utc).isoformat(),
     "scope":"WORD2523 public development recorded IDs11-13, nine primary partial views plus four ID11 sentinels. Native three not re-reviewed.",
     "source_manifest_path":str(MANIFEST),"source_manifest_sha256":sha(MANIFEST),
     "source_native_review_path":str(BASE/"REVIEW_DEV_2523.json"),
     "source_native_review_sha256":sha(BASE/"REVIEW_DEV_2523.json"),
     "meaning":{"one":"An intact physical near-to-apart process is observed within a continuous visible fragment.",
                "zero":"The shown visible fragments do not retain an intact queried process; not a claim that the source signer made an error or did not perform it.",
                "null":"Missing relevant observation prevents judgment; not a negative training label."},
     "rows":rows,
     "counts":{"views":len(rows),"direct_views":sum(r["content_reuse_audit"] is None for r in rows),
               "ordered_pixel_reuse_views":sum(r["content_reuse_audit"] is not None for r in rows),
               "actually_opened_unique_pages":len(actual_seen),"own_artifact_pages_hash_verified":len(checked_pages),
               "source_frame_files_hash_verified":len(checked_frames),"source_videos_hash_verified":len(checked_sources),
               "retained_output_slots":sum(len(r["reviewed_output_indices"]) for r in rows),
               "observed_process_labels":counts,"self_checks_passed":checks},
     "provenance":{"actually_opened_page_paths":OPENED,"reviewer":"Codex","human_expert":False,
                   "new_candidate_predictions_seen":False,"prior_context_known":True,
                   "self_check_scope":"Hashes, mappings, coverage and witness continuity. Not independent semantic accuracy verification."},
     "important_preserved_cases":["Partial outward tails in suffix/splice remain motion label1 while the full-chain view label remains0.",
                                 "ID13 f53 separated relation is preserved before the later near setup.",
                                 "ID11 missing RGB f58-63 remains unknown, despite near-before and separated-after endpoints.",
                                 "A separated relation and continuing outward motion may coexist."],
     "tooling_incidents_preserved":["An initial overlarge metadata print was truncated and JSON parsing failed; retried compact metadata, with no scientific output changed.",
                                   "A later schema print hit PowerShell/Python cp949 encoding; stdout UTF-8 fixed the metadata read, with no label or image changed."],
     "unperformed":["No new inference or training","No thresholds tuned","No private/user/mentor/heldout data accessed",
                    "No old labels or model outputs changed","No semantic correctness, contact3D, exact boundary or repetition truth",
                    "No actual retimed video playback or speed-performance test"],
     "repeated_development":True,"independent_final_test":False,"new_inference":False,
     "new_training":False,"old_labels_modified":False,"whole_goal_complete":False}
target=BASE/"REVIEW_DEV_PARTIAL_2523.json"
check(not target.exists(),"Refuse overwrite of an existing review")
target.write_text(json.dumps(out,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
print(json.dumps({"file":str(target),"sha256":sha(target),"counts":out["counts"]},ensure_ascii=False))

