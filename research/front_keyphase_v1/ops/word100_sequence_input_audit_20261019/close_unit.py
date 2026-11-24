from pathlib import Path
from datetime import datetime,timezone,timedelta
import json,hashlib,shutil,re
U=Path(__file__).resolve().parent;O=U.parent;ROOT=U.parents[3]
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert not (U/'FINAL_AUDIT.json').exists()
scope=read(U/'SCOPE.json');done=read(U/'COMPLETE.json');ind=read(U/'INDEPENDENT.json');summary=read(U/'SUMMARY.json');visual=read(U/'VISUAL_REVIEW.json');manifest=read(U/'VISUAL_MANIFEST.json');plan=read(U/'NEXT_PLAN.json');qa=read(U/'LAYOUT_QA.json')
assert ind['status']==qa['status']=='PASS' and not plan['training_authorized_this_turn']
assert done['model_state_unchanged'] and done['training_runs']==done['user_recordings_read']==0
assert len(done['record_hashes'])==899 and len(visual['reference_reviews'])==100 and len(visual['risk_reviews'])==8
assert ind['summary_sha256']==sha(U/'SUMMARY.json')
for p,h in scope['pins'].items():assert sha(p)==h,p
for cid,h in done['record_hashes'].items():assert sha(U/'records'/f'{cid}.json')==h,cid
for name,h in manifest['files'].items():assert sha(U/'visuals'/name)==h,name
for r in manifest['sources'].values():assert sha(r['path'])==r['sha256'],r['path']
corr=read(U/'LOCAL_ANCHOR_CORRECTION.json');assert corr['zero_wrist_mcp_clips']==31 and corr['frames']==535
assert all(sum(r['whole_left_wrist_mcp_pairs'])==0 and r['wrist_missing']==0 and 1<=r['whole_left_visible_joint_count_max']<=3 for r in corr['rows'])
metric={(x['view'],x['model'],x['condition']):x for x in summary['metrics']}
assert metric['sensor','C','reverse_thirds']['target_top1']==888 and metric['sensor','T3F','reverse_thirds']['target_top1']==648
for x in summary['per_word']:assert x['conditions']['sensor/T3F/original']['target_top1']==x['clips']
assert sum(x['conditions']['sensor/T3F/reverse_thirds']['retained']==x['clips'] for x in summary['per_word'])==46
assert sum(x['conditions']['sensor/T3F/reverse_thirds']['retained']>x['clips']/2 for x in summary['per_word'])==72
report=(U/'RESULT_KO.html').read_text(encoding='utf8')
for href in re.findall(r'(?:src|href)="([^"]+)"',report):
    if not href.startswith(('http:','https:','#')):assert (U/href).exists(),href
docs=[ROOT/'WordsInSilence_PROJECT_CONTEXT.md',ROOT/'NEW_CHAT_HANDOFF.md',O/'STATE.json'];before={str(p):sha(p) for p in docs};backup=U/'before';backup.mkdir(exist_ok=False)
for p in docs:shutil.copy2(p,backup/p.name)
stamp=datetime.now(timezone(timedelta(hours=9))).isoformat();key=U.name
next_action='공개100단어의 순서·입력 감사 결과와 NEXT_PLAN.json을 사용자와 함께 확인한다. 후속 제안은 현재 T3-F 전체 고정+GRU16 순서 보조(새4000변수) 초기동등성/동결/시간·마스크 검증 후 같은TRAIN899/DEV300·3seed 비교다. 이번에는 후보 구현·추가학습 없음. 순서변형 유지율을 오류율·채택지표로 삼지 말고 정상 인식률과 미검출/사람별 회귀를 우선한다. C/T3-F/웹/DTW보존, 사용자·멘토자료 학습/튜닝/선택 금지, ID14~16미사용유지.'
body=f'''### 2026-10-19 공개100단어 순서·입력 보존 감사 완료 — 추가학습 전 결과 검토
갱신 {stamp}. ops/{key}. 사용자 승인 범위는 공개100단어 진단만, 추가학습은 결과 함께 확인 후. TRAIN ID01~09 899영상의 masked/sensor입력과 고정seed20261013 C/T3-F 비교. 8990입력쌍 중 이전18영상180쌍 재사용,8810신규진단쌍;모델17980출력. 원본은양모델양view899/899정답이나TRAIN진단으로일반화성능아님. sensor세구간역순원단어1위 C888/899(98.78%),T3-F648/899(72.08%);T3-F46/100단어전영상유지,72/100단어과반유지. T3-F앞절반580/899(64.52%),뒤절반544/899(60.51%),2회연결899/899. masked역순C892/899,T3-F693/899. 순서변형시최강경쟁대비margin중앙C-1.31,T3-F-10.49로순서반응있음. 위값은정답률/수어오류율아님;정적·반복동작/부분인식은정당할수있음. 프레임시간순서내부삭제0/899양view;주석42869프레임중sensor42687(99.58%),masked42803(99.85%)보존. sensor주석중양손0프레임존재145영상,>=0.3초4영상모두원본장면에서마무리손내림확인. 실제화면안손관절누락잔존:후원REAL01 1.60~1.80초/REAL04 2.17~2.37초,독재자/장애인복지관전환표본등. 100대표800장+입력위험8사례{visual['risk_frames']}장검토;전899영상의전문가의미/손가락정답검증아님. masked31영상535주석프레임은왼손목등1~3점만남아손크기기준불가→local미관측,global보존;초안의손목없음문구는자체수치와모순하여LOCAL_ANCHOR_CORRECTION에정정,최초증거보존. RGB생성 retained_video=None경로오류는VISUAL_FIRST_ERROR보존후원본ZIP으로8위험사례만이어완료,100완료장면재생성/모델재추론없음. 별도코드로1798000logits·1798view입력검산PASS;4513보호근거해시유지,같은Codex산술검산으로독립agent/전문가아님. 신규모델학습0/사용자촬영읽기0/MediaPipe재추출0/운영변경0. 기존C/T3-F유지. 다음제안은기존TF전체고정+작은순서보조3seed비교,아직미구현/미학습. 원본·미검출입력변경과혼합하지않음;단어라벨만으로완수판정보장없음. RESULT_KO.html/SUMMARY/INDEPENDENT/VISUAL_REVIEW/CONCLUSION/NEXT_PLAN/FINAL_AUDIT.

'''
context=docs[0].read_text(encoding='utf-8-sig');h8='## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인\n';h9='## 9. 다음 단일 작업\n'
assert h8 in context and h9 in context and body.splitlines()[0] not in context
context=context.replace(h8,h8+'\n'+body,1).replace(h9,h9+'\n### 2026-10-19 공개100단어 순서·입력 감사 결과 검토\n'+next_action+'\n\n',1)
state=read(docs[2]);state['current_stage']='WORD100_SEQUENCE_INPUT_AUDIT_COMPLETE_PLAN_REVIEW';state['updated_date_kst']=stamp[:10];state['next_single_action']=next_action
state[key]={'status':'PUBLIC100_DIAGNOSIS_COMPLETE_WAIT_REVIEW_NO_TRAINING','at':stamp,'unit':key,
 'scope':{'TRAIN_videos':899,'words':100,'signers':'01~09','views':['masked','sensor'],'frozen_seed':20261013,'new_diagnostic_pairs':8810,'reused_prior_pairs':180},
 'sensor_original_label_top1':{'original':{'C':899,'T3F':899},'reverse_thirds':{'C':888,'T3F':648},'prefix_half':{'C':618,'T3F':580},'suffix_half':{'C':587,'T3F':544},'double_sequence':{'C':899,'T3F':899},'denominator':899},
 'per_word_T3F_sensor_reverse':{'all_samples_retained_words':46,'majority_retained_words':72,'denominator_words':100},
 'input':summary['input'],'visual':{'representative_words':100,'frames':800,'risk_samples':8,'risk_frames':visual['risk_frames'],'semantic_all_frame_expert_validation':False},
 'remaining':'Screen-in hand detection gaps, uncertain fine finger geometry, no completion/error labels. Reordering retention does not alone establish wrong signing or actual user failure cause.',
 'verification':{'logit_values':1798000,'input_views':1798,'source_pins':4513,'operator':'Codex local separate arithmetic code; no independent agent/expert'},
 'training_runs':0,'user_recordings_read':0,'new_detector_runs':0,'production_changes':0,'existing_C_T3F_preserved':True,
 'report':str(U/'RESULT_KO.html'),'next_plan':str(U/'NEXT_PLAN.json'),'training_authorized':False}
handoff=f'''# WordsInSilence 최신 인수인계

갱신: {stamp}

현재 목표: 공개100단어에서도 순서/부분 동작에 둔감한지와 구별동작이 관절입력에 남는지 진단 완료. 사용자 결과 검토 대기.

- 단위: research/front_keyphase_v1/ops/{key}
- 사용자 승인: 공개 자료 진단·보고만. 기존 C/T3-F 유지, 사용자촬영 학습/튜닝 금지. 추가학습은 함께 결과 확인한 뒤.
- 범위: TRAIN899·100단어·ID01~09, masked/sensor2views, 고정seed20261013 C/T3-F. 신규8810입력쌍+이전180쌍재사용. 사용자/멘토파일읽기0, DEV신규평가0, ID14~18원시자료접근0.
- 정상899/899양모델양view정답은TRAIN성적이라새사람성능아님. sensor역순 C888/899(98.78%)→T3-F648/899(72.08%),100단어중46전영상유지/72과반유지. masked역순C892/899,T3-F693/899. T3-F앞절반580/899,뒤절반544/899,전체2회연결899/899. 정답률/수어오류율이아닌원래단어유지민감도. 정적·반복·부분표현은정당한분류일수있음. TFmargin은역순으로중앙-10.49(C-1.31)변해순서전혀못본다는결론금지.
- 입력시간순서·미검출내부프레임삭제없음899/899양view. 주석42869프레임중sensor42687=99.58%,masked42803=99.85%보존;이는관절/의미정확도아님.
- sensor주석에양손미관측1프레임이상145영상;0.3초이상4영상원본확인시모두마무리손내림. 후원REAL01 1.60~1.80초와REAL04 2.17~2.37초는화면안손이있는데관절없음. 독재자/장애인복지관전환표본도누락. 시간유지해도미관측손모양자동복원안됨.
- 대표100단어800장+입력위험8사례{visual['risk_frames']}장전체시트시각확인. 전899영상모든프레임/필수수어단계전문가검증아님. 예전READINESS_VISUAL_REVIEW도제한포함하여재사용. 수치PASS로의미완전성승격금지.
- masked31영상535주석프레임은왼손목등1~3점만보이고MCP없어손크기추정불가→local모양미관측/global손목보존. 초안손목없다는문구는자체수치0과모순,LOCAL_ANCHOR_CORRECTION이최종근거. 최초LOCAL_ANCHOR_AUDIT은오류이력으로보존.
- RGB생성중retained_video=None경로처리오류FIRST_ERROR보존. 완료100장면해시재사용하고위험8개만원본ZIP으로복구. 모델/기완료점수재추론없음.
- 검산1798000logit값·1798view입력·4513원본/모델/웹핀PASS. 같은Codex의별도코드산술검산이며독립agent/수어전문가아님. HTML검색100행/데스크톱·모바일QA완료.
- 제안 NEXT_PLAN: 기존T3-F전체47236고정+GRU16순서보조4000변수. 초기logits동등·BN동결·실제미검출시간보존후같은TRAIN899/DEV300·3seed비교. 정상100단어정확도/top5/사람별·미검출회귀우선. 합성순서변형유지율낮음만으로채택금지. 단어라벨만으로필수단계완수보장없음;그평가에는전문가정상/생략/순서오류자료필요. 아직후보구현·추가학습0.
- C seed20261013 epoch77 SHA283a54bd9866c97959f9d615ebc38d126ee25688821960b208cf9412389eeaab / T3-F동일seed epoch69 SHAb0faa0df5055204295c60b78c596a0b490cd63ef14f30225582a85ea39855846 불변. 웹/DTW불변. ID17/18은과거평가에이미사용;ID14~16정상299개미사용유지.
- 실행종료: audit16173=0,verify28118=0,visual19802첫오류보존→86313복구0,report/Chrome QA완료. 진행중학습없음.
- 보고서: {U/'RESULT_KO.html'}

다음 한 작업: {next_action}
'''
for p,h in before.items():assert sha(p)==h,'Concurrent documentation changes'
docs[0].write_text(context,encoding='utf8');docs[1].write_text(handoff,encoding='utf8');docs[2].write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
files={str(p.relative_to(U)):sha(p) for p in U.iterdir() if p.is_file()}
(U/'FINAL_AUDIT.json').write_text(json.dumps({'status':'PASS','at':stamp,'docs_synced':[str(p) for p in docs],'protected_pins_unchanged':4513,'public_rgb_sources_unchanged':len(manifest['sources']),'report_links_valid':True,'training_runs':0,'user_recordings_read':0,'implementation':'Codex local','verification':'Separate arithmetic/NPZ code; not independent agent or expert','files':files},ensure_ascii=False,indent=2)+'\n',encoding='utf8')
print('PASS: evidence, report, protection and current documents synced; no training')
