"""Record the five-clip source RGB review actually performed via view_image."""
from pathlib import Path
from datetime import datetime, timezone
import json, sys
sys.dont_write_bytecode=True
from evidence_fallback import read,sha
from actual_geometry_review_build import IDS,U

findings={
'NIA_SL_WORD0651_REAL08_F':{
 'facts':['하단838px는 사전 예상과 같음. f22~25(0.733~0.833s)의 왼손 전체 제외는 몸 옆에 내려둔 손이며 오른손은 입 근처에 남음.',
          'f26~27은 왼손을 올리며 경계를 통과하는 부분: mask L2→7점. f29는 왼손 전부 선 위이나 sensor L0, f28은 반대로 native L0/sensor L21. f30부터 양손이 가슴/배 위에 위치한 표본에서 양쪽21점.',
          '검토 표본 f38/46/55/63/71/79의 양손 동작은 선 위에 있음.'],
 'inference':'이 flag의 기하 손실은 주로 내려둔 왼손에서 양손 동작으로 전환하는 상승 초입이다. 관찰한 가슴 앞 동작 또는 받침손을 새 경계가 잘랐다는 증거는 찾지 못했다.',
 'residual':'f29 단일프레임 sensor 누락 등 관측 흔들림은 존재. 표본만으로 수어 의미 전체 보존을 확정하지 않음.'},
'NIA_SL_WORD0772_REAL02_F':{
 'facts':['하단837px는 사전 예상과 같음. f53~54(1.767~1.800s) 오른손은 입/턱 근처, 왼손은 몸 옆 아래. 왼손 전체 제외는 내려둔 손이다.',
          'f55~58 왼손이 올라오며 mask L1→3→6→11점. f59는 왼손21점이 선 위, f60은 RGB 왼손이 선 위지만 native/sensor 모두 L0. f61~62 양손21점.',
          '이후 표본 f70/78/86/94/103/111의 양손 동작은 경계 위.'],
 'inference':'빠른 상승 때문에 moving-near-boundary가 생기지만, 검토한 주동작과 양손 결합 자세는 위쪽에 보존되어 보인다.',
 'residual':'f60의 실제 보이는 손 관측 누락은 경계와 별개 검출 문제. 경계 통과 초입 손가락 일부는 실제 제외되므로 완전 무손실이 아님.'},
'NIA_SL_WORD0185_REAL04_F':{
 'facts':['하단891px 변동 없음. f65~71(2.167~2.367s) RGB 오른손은 어깨/귀 옆에 올라와 있다. native와 sensor 모두 오른손21점 관측이 없고, 어깨·팔꿈치·pose 손목 선은 남아 있다.',
          '같은 구간 mask에 남은 것은 몸 옆 아래 왼손의1점(f65~70) 또는2점(f71)이다. sensor가 이 잔여점도 놓쳐 양손 유효점0이 된다. 완전한 동작손이 새로 잘려0이 된 사례와 다르다.',
          'f72부터 오른손 관측 회복. f74~77 왼손 상승 및 f81/88/96의 양손 자세는 경계 위. f103~108은 두 손이 가슴 앞에서 겹치는 동안 native R21/sensor R0 프레임이 있고 f110에서는 native L21/R0, sensor L0/R0이다.'],
 'inference':'최장0.233초 총0은 기존 오른손 관측 누락과 하단의 약한 왼손 잔여점 불일치가 겹친 결과다. 하단 경계가 올라간 오른손을 잘랐다는 근거가 아니다.',
 'residual':'핵심일 수 있는 올라간 오른손 주먹/손모양의7프레임 관측 누락이 실제로 남는다. 겹친 양손의 sensor 추가누락도 남음. 정보 완전성 PASS 아님.'},
'NIA_SL_WORD1085_REAL08_F':{
 'facts':['하단913px 변동 없음. f36/43 양손 얼굴 앞 자세, f47~50 왼손을 내리면서 오른손은 반대 어깨 앞에 남음. 왼손 f52~53 일부 제외 후 아래에서 쉰다.',
          'f70~81 오른손은 허리 옆 높이에서 선 위21점으로 유지. f82 mask R16점, f83 R9점으로 내려간다.',
          'f84~88(2.800~2.933s) mask R3→2→1→1→1점, sensor R0. RGB에서는 양손이 몸 옆으로 내려오는 종료 전환. f89부터 mask도0.'],
 'inference':'검토한 동시 sensorloss는 내려가는 손의 경계 잔여점 불일치다. 허리에서 머문 이전 자세는 선 위에 남았다.',
 'residual':'내려가는 끝동작의 언어학적 중요성은 미판정. f66처럼 선 위 오른손 native 관측만 사라지는 구간도 있으며 ROI 원인으로 혼동하면 안 됨.'},
'NIA_SL_WORD2029_REAL08_F':{
 'facts':['하단895px 변동 없음. f93~97(3.100~3.233s) 양손을 내려 몸 옆에 두는 종료 전환에서 mask L2/R6→L0/R1으로 잔여점이 줄고 sensor는0이다.',
          'f47/49/64 RGB 손이 선 위인데 sensor 양손 유효점0이 한 프레임씩 있음. f47 native는양손21, f49/64는 native 오른손21·왼손0이다.',
          'f52~87 다수 표본에서 왼손이 오른손 아래 받침 역할처럼 보이나 native와 sensor 모두 왼손21점이 없다. pose 왼팔 선과 일부 겹친 오른손 관측은 남음. 이 손은 하단보다 상당히 위에 있다.',
          'f88~89 양손이 관측되며 내리는 전환, f90~92 부분 경계손실.'],
 'inference':'최장0.167초 총0은 종료 전환의 잔여점 소실이다. 반면 겹친 받침손의 장기 누락은 화면 잘림과 무관한 검출/배정 문제로 보인다.',
 'residual':'받침손의 손모양·접촉 정보를 실제로 관측하지 못한 구간이 있어 모든 필수 정보를 보존했다고 판정할 수 없다. 단일프레임 sensor 추가누락도 존재.'}
}

rows=[];pins={};image_count=0
for cid in IDS:
 p=U/f'actual_geometry_review_{cid}.json';m=read(p);pins[str(p)]=sha(p)
 for ip,h in m['images'].items():assert sha(ip)==h;pins[ip]=h;image_count+=1
 for sp,h in m['source_pins'].items():assert sha(sp)==h;pins[sp]=h
 comparison={}
 for side in ('left','right'):
  actual=m['triage_row']['hands'][side]['native_to_mask_geometry'];legacy=m['legacy_geometry_recomputed'][side]
  keys=['d_moving_hand_near_boundary','newly_absent_core_frames','near_boundary_palm_frames','near_boundary_speed_p75','palm5_drop_pp','robust_core_movement_shoulder_width']
  comparison[side]={k:{'legacy_recomputed':legacy[k],'actual':actual[k],'equal':legacy[k]==actual[k]} for k in keys}
 rows.append({'clip_id':cid,'label':m['label'],'source_sha256':m['source_pins'][m['source']],
              'record_sha256':m['triage_row']['record_sha256'],'bottom_pixels':m['bottom_pixels'],
              'viewed_images':list(m['images']),'viewed_frame_ids':sorted({f['frame_id'] for f in m['frames']}),
              'viewed_frame_timestamp_records':m['frames'],'source_pts_checked':True,
              'viewing_method':'All overview/focus/sensor PNGs listed were opened with view_image and read by roi_data_budget.',
              'legacy_vs_actual_geometry_comparison':comparison,
              'observed_preservation':'검토한 표본에서 새 경계가 선 위 주요 자세 또는 받침손을 물리적으로 잘라낸 명백한 사례는 발견하지 못함.',
              'review_status':'SOURCE_SAMPLE_REVIEWED_WITH_RESIDUAL_OBSERVATION_LOSS_NOT_SEMANTIC_PASS',**findings[cid]})
out={'status':'FIVE_TRAIN_SOURCE_VISUAL_REVIEWS_COMPLETE_NOT_FULL_CORPUS_GATE',
     'created_utc':datetime.now(timezone.utc).isoformat(),'reviewer':'roi_data_budget','clip_count':5,'viewed_image_count':image_count,
     'triage_interpretation_correction':{
        'first_evidence_preserved':str(U/'actual_input_triage_RESULT.json'),
        'affected_clip_ids':IDS[:2],
        'cause':'Legacy triage records only 619 rows selected by forecast.geometry_flags; these two clips are absent. actual_input_triage marked absent legacy row plus current moving flag as newly_moving.',
        'evidence':'Legacy direct hand_info recomputation at the same frozen boundary produces identical moving flags, missing-frame counts, palm5 delta, near-boundary speed and overall movement for both clips.',
        'correct_reading':'Previously unlisted review flags, not demonstrated newly introduced geometric regression.',
        'source_or_condition_changes':0,'automatic_semantic_pass':False},
     'rows':rows,'source_pins':pins,
     'limits':['Only listed source frames were viewed. A sparse overview does not cover every flagged interval of a clip or all899 TRAIN clips.',
               'Manual image interpretation is not a sign-language expert annotation. Returning-to-rest vs meaningful terminal movement is uncertain.',
               'No new obvious physical clipping is different from complete observation preservation: missing active/support hands remain.',
               'No training or inference was run; recognition benefit is unknown.'],
     'new_detector_runs':0,'classifier_runs':0,'training_runs':0,'development_videos_opened':0,'boundary_or_threshold_changes':0}
p=U/'actual_geometry_review_RESULT.json'
with p.open('x',encoding='utf-8') as f:json.dump(out,f,ensure_ascii=False,indent=2,allow_nan=False)
print(json.dumps({'output':str(p),'images_viewed':image_count,'source_frames_viewed':sum(len(r['viewed_frame_ids']) for r in rows)},ensure_ascii=False))
