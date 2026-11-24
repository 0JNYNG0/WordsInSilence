from pathlib import Path
from datetime import datetime,timezone,timedelta
import json,hashlib,shutil,re
U=Path(__file__).resolve().parent;O=U.parent;ROOT=U.parents[3]
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert not (U/'FINAL_AUDIT.json').exists()
r=read(U/'RESULT.json');v=read(U/'INDEPENDENT.json');s=read(U/'SCOPE.json');q=read(U/'LAYOUT_QA.json');plan=read(U/'NEXT_PLAN.json')
assert v['status']==q['status']=='PASS'
assert r['status']=='PASS_PREFLIGHT_ONLY_NOT_TRAINED'
assert not plan['training_authorized_this_turn'] and not read(U/'IMPLEMENTATION_READINESS.json')['training_allowed']
assert r['training_runs']==r['optimizer_steps']==r['user_recordings_read']==0
assert v['result_sha256']==sha(U/'RESULT.json')
for p,h in s['pins'].items():assert sha(p)==h,p
for p,h in r['files'].items():assert sha(U/p)==h,p
page=(U/'RESULT_KO.html').read_text(encoding='utf8')
for target in re.findall(r'(?:src|href)="([^"]+)"',page):
    if not target.startswith(('http:','https:','#')):assert (U/target).exists(),target
docs=[ROOT/'WordsInSilence_PROJECT_CONTEXT.md',ROOT/'NEW_CHAT_HANDOFF.md',O/'STATE.json']
before={str(p):sha(p) for p in docs};backup=U/'before';backup.mkdir(exist_ok=False)
for p in docs:shutil.copy2(p,backup/p.name)
stamp=datetime.now(timezone(timedelta(hours=9))).isoformat();key=U.name
next_action=plan['next_single_action']+' 이번에는 구현/학습 전 검증까지 완료했고 추가학습은 사용자 결과 확인 후다. 기존 T3-F 전체47236고정+GRU16/출력4000만 학습하는 후보. 같은 C 관측/전체·핵심 일정과 선택규칙 유지, 정상 정확도/top5/사람·미검출회귀 우선. C/T3-F/웹/DTW유지, 사용자·멘토자료 학습/튜닝/선택 금지. ID14~16미사용 유지. IMPLEMENTATION_READINESS는 학습실행승인 아님.'
heading='### 2026-10-19 순서 기억 보조 F-G16 구현·학습 전 검증 완료 — 추가학습 전 결과 확인'
body=f'''{heading}
갱신 {stamp}. ops/{key}. 최신 사용자 요청의 추가학습 전 결과 확인 조건에 따라, 별도 순서 보조 후보 구현과 초기동등성/동결/패딩/미검출/기울기 검증만 수행. 기존T3-F 47236변수·BN전체 고정, C평균 전32차원 특징→단방향GRU16(2400)+bias없는출력층(1600)→기존점수에더함. 새4000/총51236, 출력층0초기화. 공개TRAIN899·100단어·9명만읽음. masked/sensor×whole/context3596입력을3개seed20261013~15에연결하여10788출력(1078800logit)기존TF와bitwise일치. 이전저장1798whole출력도오차0. 실프레임236617/양손미관측6252유지(구간/view중복포함;고유영상프레임수아님). padding/배치순서/영상간기억초기화/잘못된mask·clock거절/보조기울기/저장재읽기PASS. 초기zerohead에서GRU기울기0은예상동작;별도복사본nonzero고정head에서GRU기울기전달확인,학습·optimizer갱신0. 기존및새초기가중치/BN검증전후불변. NumPy수식별도검산max2.22e-16,기존snapshot61tensors일치,보호핀3613불변. 같은Codex별도계산검산이며독립agent/수어전문가아님. 첫읽기cp949인코딩오류는SETUP_NOTE보존후UTF8명시로해결;후보검증실패없음. UNTRAINED_seed20261013.pt는미학습별도증거,운영적용아님. 초기결과동일하므로믿다/사육장등혼동개선미검증. 기존부동결해도학습된보조점수가최종판단을악화시킬수있음. 가려진손모양복원·필수동작완수판정·새사용자개선보장없음. TCN특징은비인과적이므로전체모델실시간인과추론주장금지. 다음제안같은TRAIN899/DEV300·3seed/기존일정으로새4000만학습,저장C/T3F비교/회귀검증;사용자검토후. DEV/보류신규추론0,개인/멘토파일읽기0,웹/DTW변경0. preflight54321종료0/verify종료0/보고서Chrome데스크톱·모바일QA PASS. RESULT.json/INDEPENDENT.json/RESULT_KO.html/NEXT_PLAN.json/FINAL_AUDIT.json.

'''
ctx=docs[0].read_text(encoding='utf-8-sig');h8='## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인\n';h9='## 9. 다음 단일 작업\n'
assert h8 in ctx and h9 in ctx and heading not in ctx
ctx=ctx.replace(h8,h8+'\n'+body,1).replace(h9,h9+'\n'+heading+'\n'+next_action+'\n\n',1)
state=read(docs[2]);state['current_stage']='WORD100_FROZEN_ORDER_AUX_PREFLIGHT_COMPLETE_WAIT_REVIEW';state['updated_date_kst']=stamp[:10];state['next_single_action']=next_action
state[key]={'status':'IMPLEMENTED_PREFLIGHT_PASS_WAIT_REVIEW_NO_TRAINING','at':stamp,'unit':key,
 'candidate':'F-G16 frozen T3-F + GRU16 residual','frozen_parameters':47236,'new_trainable_parameters':4000,'total_parameters':51236,
 'public_videos':899,'words':100,'signers':9,'input_variants':3596,'initial_output_comparisons':10788,'all_initial_outputs_equal':True,
 'saved_original_comparisons':1798,'saved_original_max_abs_error':0,'real_frames_including_repeated_views_and_segments':236617,'both_hand_missing_frames_kept':6252,
 'all_weights_and_buffers_unchanged':True,'independent_status':'PASS same Codex separate NumPy recurrence and saved-score/input recount','training_runs':0,'optimizer_steps':0,'user_recordings_read':0,
 'DEV_and_holdout_new_inference':0,'web_DTW_changed':False,'C_T3F_preserved':True,'training_authorized':False,
 'remaining':'Accuracy gain not measured; existing confusion unchanged initially; missing hand shapes and signing completeness unresolved; new residual may regress.',
 'report':str(U/'RESULT_KO.html'),'next_plan':str(U/'NEXT_PLAN.json'),'running_processes':[]}
handoff=f'''# WordsInSilence 최신 인수인계

갱신: {stamp}

현재 목표: 앞 동작의 순서를 기억하는 보조 후보 F-G16 구현·학습 전 검증 완료. 사용자 결과 확인 후 추가학습.

- 단위: research/front_keyphase_v1/ops/{key}
- 최신 사용자 요청: 작은 순서 보조 부분을 추가해 비교 검증, 사용자촬영 학습/튜닝금지, 추가학습은 함께 결과 확인한 다음. 이에 구현/학습전검증만 완료, 학습0/optimizer0. 이전 대화의 포괄학습승인으로 이번 검토단계를 건너뛰지 않음.
- 구현: frozen_order_aux.py. 기존 T3-F 전체47236변수(phase_weight포함)·BN고정. C프레임별32특징→단방향GRU16(2400)→bias없는head(1600)→기존점수추가. 새4000/총51236. head0초기화. 미래문맥을쓰는기존TCN특징이므로전체모델causalstreaming주장금지.
- 공개TRAIN899·100단어·ID01~09만읽음. masked/sensor×whole/context3596입력,기존3seed에각각연결10788출력/1078800logit비교bitwise동등. 저장seed13whole1798출력오차0. 새학습3회나새영상10788개가아님.
- 실프레임236617/양손미관측6252시간유지(입력view/구간중복포함). 패딩만제외. batch순서·영상간기억초기화·malformedmask/clock거절·초기동등·동결/BN·보조기울기·저장재읽기PASS. zerohead초기GRU기울기0은예상,고정nonzerohead의별도진단복사본에서GRU기울기전달확인. 실제optimizer없음.
- verify.py는candidateforward재호출없이저장출력/입력전수재집계+NumPy수식검산. GRU최대오차2.22e-16. 기존snapshot61tensors일치/보호핀3613불변. 같은Codex별도코드이며독립agent/수어전문가검증아님.
- SETUP_NOTE:첫읽기cp949오류는UTF8명시로복구,원본/조건변경없음. preflight54321종료0,verify종료0,진행중학습없음. 보고서Chrome데스크톱/모바일QA PASS.
- 미학습스냅샷UNTRAINED_seed20261013.pt는격리증거만. 웹에적용하지않음. 모델학습도트레이너도이번에실행하지않음. IMPLEMENTATION_READINESS.json의training_allowed=false.
- 초기결과동일하므로믿다/사육장등혼동그대로. 인식률향상미검증. 보조부학습후기존부고정해도최종점수회귀가능. 가려진손모양복원·모든단계완수판정안됨. 파라미터증가효과와순서효과를분리하려면별도대조필요.
- 다음제안:같은TRAIN899/DEV300,seed20261013/14/15각해당TF체크포인트고정,같은C관측/구간일정·batch32·최대80epoch·patience12·평균DEV CE선택유지. AdamWlr.001/wd.001/clip1로새4000만학습. C/TF저장결과와top1/top5/CE/사람별·미관측회귀/후반변동비교. 정상sensor평균및3seed중2개이상향상·masked/top5/그룹비회귀제안유지. 결과보고후운영교체별도. 아직학습실행승인아님.
- 이전100단어감사근거: ../word100_sequence_input_audit_20261019. TRAIN원본899/899는학습자료성적;역순유지C888/899→TF648/899는수어정오판정아님. 내부순서삭제없으나실화면안손관절누락잔존. 그원본/결과동결보존.
- 사용자/멘토파일읽기0·학습/튜닝/후보선택0. DEV/보류신규평가0. ID14~16정상299개미사용유지,ID17/18기평가자료를새독립검증으로부르지않음. C/T3-F원본·웹·DTW불변. Git/다운로드/외부전송/카메라없음.
- 보고서: {U/'RESULT_KO.html'}

다음 한 작업: {next_action}
'''
for p,h in before.items():assert sha(p)==h,'Concurrent documentation change'
docs[0].write_text(ctx,encoding='utf8');docs[1].write_text(handoff,encoding='utf8');docs[2].write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
for p in U.glob('*.json'):read(p)
assert read(O/'STATE.json')[key]['training_runs']==0
assert all(key in p.read_text(encoding='utf8') for p in docs)
out={'status':'PASS','at':stamp,'files':{p.name:sha(p) for p in U.iterdir() if p.is_file()},'doc_hashes':{str(p):sha(p) for p in docs},'protected_pins_unchanged':len(s['pins']),'docs_synced':True,'report_visual_checked':True,'implementation':'Codex local','verification':'Same Codex separate arithmetic, not independent agent','training_runs':0,'user_recordings_read':0,'training_allowed':False,'running_processes':[]}
(U/'FINAL_AUDIT.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
print('PASS: evidence, protected files and three current documents synchronized; awaiting results review')
