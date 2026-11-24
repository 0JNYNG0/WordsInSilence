"""Record actual same-Codex inspection of all 20 rendered original RGB pages."""
from pathlib import Path
from collections import Counter
import json,hashlib,datetime as dt
import numpy as np
H=Path(__file__).resolve().parent;U=H.parent
def read(p):return json.loads(Path(p).read_text(encoding="utf-8"))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
manifest=read(H/"RECOVERY_RGB_MANIFEST.json")
results=read(U/"RECOVERY_RESULT_MANIFEST.json")
notes={
("P048",72):"추가 L은 화면 오른쪽으로 뻗은 왼팔 끝의 닫힌 손 윤곽과 겹친다. 위쪽의 편 오른손과 공간적으로 구별된다. 접힌 손가락 내부 위치는 RGB만으로 모두 검증할 수 없다.",
("P048",73):"추가 L은 앞쪽 닫힌 왼손에 놓이고 기존 오른손은 어깨 쪽에 따로 있다. 새 반대손 중복은 보이지 않는다. 닫힌 손의 관절별 위치는 미검증이다.",
("P048",74):"원본에서 앞쪽 왼손은 여전히 보인다. crop 후보는 있으나 full 후보에는 해당 손이 없어 추가하지 않았다. 보이는 손이 모두 회복된 것은 아니다.",
("P063",49):"추가 R은 몸 오른팔 끝에서 올라오는 닫힌 손·엄지 윤곽과 겹친다. 낮은 반대손과 구별된다. 움직이는 손의 흐림과 접힘 때문에 관절별 정밀 오차는 판단하지 않는다.",
("P063",51):"추가 R은 오른팔이 올라가는 끝의 엄지를 드러낸 손에 놓인다. 아래의 왼손과 중복되는 새 손은 보이지 않는다. 주먹 내부 손가락 순서·각도는 미검증이다.",
("P063",54):"추가 L은 아래에서 올라오는 왼팔 끝의 손 윤곽에 놓이며 얼굴 앞 오른손과 구별된다. 사선으로 보이는 닫힌 손 내부는 일부 가려진다.",
("P063",56):"추가 L은 아래·앞에서 올라오는 왼팔 끝의 손과 겹친다. 위 오른손에 가까워지지만 두 팔의 연결은 구별 가능하다. 가까운 엄지·접힌 손가락의 정확한 접촉과 깊이는 확정하지 않는다.",
("P063",57):"양손이 얼굴 아래 중앙에서 겹친다. full/crop 후보의 손 윤곽과 관절 위치가 섞여 보이며 새 L은 image_scales_disagree_or_ambiguous로 보류됐다. 가려진 손가락과 정확한 손 경계를 단일 화면에서 확정하지 않는다.",
("P063",63):"추가 R은 아래쪽 오른팔 끝에 놓이고 기존 L은 얼굴 앞 위쪽 손에 놓인다. 두 손의 큰 위치와 좌우 연결을 구별할 수 있으나 접힌 손의 모든 관절은 확인할 수 없다.",
("P063",66):"추가 R은 올라간 오른팔 끝, 기존 L은 가슴 위쪽의 반대팔 끝에 있다. 새 양손 중복이나 큰 좌우 교환은 보이지 않는다. 얼굴과 겹친 주먹 세부는 미검증이다.",
("P063",72):"원본의 두 손이 가까워지며 앞뒤로 겹친다. crop에 아래 손 후보가 보이지만 full에서 해당 손 공통지원이 없어 새 L을 추가하지 않았다. 겹친 관절의 정밀 정체는 미확정이다.",
("P066",57):"두 새 손이 각각 낮은 오른팔 끝과 가슴 앞 왼팔 끝의 서로 떨어진 손 윤곽에 놓인다. 하나의 손을 두 좌우 슬롯에 중복시킨 모습은 보이지 않는다. 오른손의 굽은 손가락 내부는 미검증이다.",
("P066",59):"추가 R은 몸 오른쪽 팔 끝에서 중앙으로 올라오는 손에 놓이고 기존 L은 반대팔 끝에 있다. 손들이 가까우나 큰 좌우 연결은 구별된다. 엄지 및 손끝 접촉은 확정하지 않는다.",
("P066",61):"추가 R은 얼굴 아래로 올라가는 오른손, 기존 L은 아래 가슴 앞 왼손에 각각 놓인다. 닫힌 윤곽·엄지의 큰 위치가 보이나 관절별 좌표 정확도는 미검증이다.",
("P066",71):"얼굴 앞에서 양팔과 손이 교차·겹친다. 새 full 후보는 current_arm_uncertain으로 보류됐고 R을 새로 채우지 않았다. 후보가 어느 물리손의 어느 관절을 따르는지 일부 모호하므로 이 거절을 정답·오답으로 채점하지 않는다.",
("P066",82):"추가 R은 화면 왼쪽에서 올라간 오른팔 끝의 손 윤곽에 놓인다. 아래쪽 반대손과 팔 연결은 구별되지만 두 손이 가까워 세부 관절·깊이는 미확정이다.",
("P069",28):"추가 R은 화면 왼쪽 오른팔 끝에서 벌어진 손가락 윤곽과 겹친다. 기존 L은 반대편에 따로 남는다. 얇게 투영된 손가락 여러 개를 각각 정확히 구별하기 어렵고 기존 손 관측의 정확도도 이번 보존 검산으로 보장되지 않는다.",
("P069",29):"추가 R은 오른팔 끝에서 비스듬히 모인 긴 손가락과 벌어진 엄지의 큰 윤곽을 따른다. 반대손과 공간적으로 구별된다. 측면으로 겹쳐진 손가락별 위치는 미검증이다.",
("P072",70):"두 추가 손은 각 팔 끝의 서로 떨어진 손 윤곽에 놓인다. 손들이 낮은 위치에서 올라오는 중이며 측면으로 겹친 손가락·엄지 세부는 흐리다. 두 이미지 크기에서 지지되지 않은 6개 점은 실제 출력에서도 미관측으로 남았다.",
("P072",72):"추가 R은 오른팔 끝에서 올라오는 비스듬한 편 손 윤곽에 놓이고 왼손은 반대편에 있다. 새 반대손 중복이나 명백한 좌우 교환은 보이지 않으나 긴 손가락들이 측면으로 겹쳐 개별 관절은 미검증이다."
}
assert len(manifest["pages"])==len(notes)==20
rows=[];added_rows=[];counter=Counter();pins={}
for p in manifest["pages"]:
 key=(p["alias"],p["frame_id"]);assert key in notes
 assert sha(p["file"])==p["sha256"]
 assert sha(p["original_rgb"]["image"])==p["original_rgb"]["image_sha256"]
 pins[p["file"]]=p["sha256"];pins[p["original_rgb"]["image"]]=p["original_rgb"]["image_sha256"]
 rec=next(r for r in results["records"] if r["alias"]==p["alias"])
 assert sha(rec["output"])==rec["output_sha256"] and sha(rec["ledger"])==rec["ledger_sha256"]
 pins[rec["output"]]=rec["output_sha256"];pins[rec["ledger"]]=rec["ledger_sha256"]
 z=np.load(rec["output"]);i=int(np.flatnonzero(z["frame_id"]==p["frame_id"])[0])
 slots=[]
 for a in p["additions"]:
  side=a["side"];off=33+21*side
  item={"side":side,"side_name":"L" if side==0 else "R",
    "supported_points":int(z["native_observed"][i,off:off+21].sum()),
    "unsupported_local_joint_ids":np.flatnonzero(~z["native_observed"][i,off:off+21]).tolist(),
    "coarse_hand_identity":"CONSISTENT_WITH_VISIBLE_ARM_CHAIN_IN_THIS_FRAME",
    "new_opposite_hand_duplicate":"NOT_SEEN_IN_THIS_QUALITATIVE_REVIEW",
    "gross_displaced_or_collapsed_hand_overlay":"NOT_SEEN_AT_REVIEWED_BODY_SCALE",
    "fine_finger_landmark_accuracy":"NOT_VALIDATED",
    "educational_correctness":None,"three_dimensional_orientation":None,"contact":None}
  slots.append(item);added_rows.append({"alias":key[0],"frame_id":key[1],**item})
  counter["reviewed_added_hand_frames"]+=1
  counter["reviewed_supported_added_points"]+=item["supported_points"]
  counter["unvalidated_unsupported_joint_slots_retained"]+=len(item["unsupported_local_joint_ids"])
 conflicts=[d for d in p["decisions"] if d["reason"] in [
     "existing_identity_conflict","unresolved_existing_overlap","image_scales_disagree_or_ambiguous",
     "multiple_new_same_side","current_arm_uncertain"]]
 for c in conflicts:counter["conflict_"+c["reason"]]+=1
 counter["executed_frames_reviewed"]+=1;counter["frames_without_addition"]+=int(not slots)
 rows.append({"alias":key[0],"frame_id":key[1],"timestamp_us":p["timestamp_us"],"reviewed_page":p["file"],
   "page_sha256":p["sha256"],"original_rgb":p["original_rgb"],"output":rec["output"],"output_sha256":rec["output_sha256"],
   "ledger":p["ledger"],"ledger_sha256":p["ledger_sha256"],"additions":slots,"explicit_conflicts":conflicts,
   "observation_ko":notes[key],"review_scope":"Original RGB next to overlays; same original fixed frame, no new time window.",
   "gross_anatomical_ground_truth":None,"fine_landmark_ground_truth":None})
assert counter["reviewed_added_hand_frames"]==18 and counter["reviewed_supported_added_points"]==372
assert counter["frames_without_addition"]==4 and counter["unvalidated_unsupported_joint_slots_retained"]==6
for p in [H/"RECOVERY_RGB_MANIFEST.json",H/"RECOVERY_SOURCE_VERIFICATION.json",U/"RECOVERY_RESULT_MANIFEST.json"]:
 pins[str(p)]=sha(p)
result={"schema":"fixed303_recovery_visual_review_v1","status":"COMPLETED_ALL20_EXECUTED_FRAMES_AND18_ADDED_HAND_SLOTS_QUALITATIVE_ONLY",
 "at_utc":dt.datetime.now(dt.timezone.utc).isoformat(),"author":"Same Codex semantic_data_audit; not blind or independent human/KSL expert",
 "counts":dict(counter),"frames":rows,"added_hand_slots":added_rows,"source_pins":pins,"script_sha256":sha(__file__),
 "summary_ko":["고정 303프레임 중 실제 후보를 실행한 20프레임을 원본·기존관측·추가관측·거절후보와 나란히 모두 확인했다.",
  "추가 18손 슬롯은 해당 팔 끝의 보이는 손 위치와 대체로 겹치며, 새 반대손 중복이나 명백한 좌우 교환은 이번 검토에서 보이지 않았다. 이것은 관절 정확도 측정이나 해부학적 정답 검증이 아니다.",
  "P063 f57과 P066 f71의 겹친 손은 후보 불일치·팔 정체 불확실로 보류됐다. P048 f74와 P063 f72 역시 공통 이미지 지원이 없어 미관측으로 남았으며 회복 성공으로 세지 않는다.",
  "P072 f70의 양손은 17점/19점만 저장하고 6점은 미관측으로 남았다. 닫히거나 측면으로 겹친 손가락의 정밀 위치, 접촉, 실제 3D 회전과 교육적 옳고그름은 전부 미검증이다."],
 "limitations":["All coarse identity observations apply only to the listed exact native frames.",
   "Review at fixed body-scale presentation can detect conspicuous displacement/duplication but cannot certify each finger joint.",
   "Agreement of two image scales from the same model is correlated evidence, not independent landmark truth.",
   "Retaining original landmarks does not prove original landmarks accurate.",
   "No acceptance decision, stored coordinate, mask, threshold or old result was changed after visual review."],
 "actions":{"new_model_calls":0,"new_rgb_decoding":0,"new_image_frames":0,"alignment":0,"score_recalculation":0,"label_or_coordinate_changes":0}}
dest=H/"RECOVERY_VISUAL_REVIEW.json";assert not dest.exists()
dest.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
print(json.dumps({"status":result["status"],"counts":dict(counter),"sha256":sha(dest)},ensure_ascii=False))
