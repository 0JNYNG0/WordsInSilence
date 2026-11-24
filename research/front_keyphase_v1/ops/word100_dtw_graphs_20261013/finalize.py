import json
import hashlib
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path
U=Path(__file__).resolve().parent;B=U.parents[1];R=B.parents[1]
def read(p):return json.loads(Path(p).read_text('utf-8-sig'))
def write(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n','utf-8')
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
assert not (U/'COMPLETION.json').exists()
ui=read(U/'UI_CHECK.json');check=read(U/'GRAPH_CHECK.json');inv=read(U/'BASIS_INVENTORY.json')
assert ui['status']==check['status']=='PASS'
assert not ui['browser_errors']
assert all(not ui[k]['overflowX'] and ui[k]['ranking']==5 for k in ['desktop','narrow'])
saved=read(B/'ops/word100_analysis_20261013/SAVED_ANALYSES.json')
lookup={r['trial_id']:r for r in saved}
for tid,visible in [('tri_cc75cf631a1f',ui['desktop']['charts']),('tri_9171304d585b',ui['first_attempt']['metrics'])]:
    original=lookup[tid]['analysis']['dtw']['metrics']
    assert [v['value'] for v in visible]==[m['value'] for m in original]
guard=read(U/'ORIGINAL_GUARD.json');assert all(sha(p)==h for p,h in guard.items())
assert inv['total_training_clips']==108 and inv['total_other_signer_comparisons_available']==96
assert all(x['observation_hashes_match'] for x in inv['items'])
now=datetime.now(ZoneInfo('Asia/Seoul')).isoformat()
done={'status':'COMPLETE_VERIFIED_DISPLAY_AND_CRITERIA_PLAN','at_kst':now,'implementer':'Codex local',
      'verification':'7 display edge checks + browser QA + original metric equality + hashes; no other agent review',
      'graphs':['2D bend angle0-180degrees','shoulder-relative wrist difference0-100percent','relative wrist trajectory0-100percent'],
      'display_limits_are_pass_thresholds':False,'overflow':'actual value retained and marked','missing':'not zero',
      'tcndtw_backend_and_model_unchanged':True,'user_records_unchanged':True,
      'camera_training_user_inference_executed':False,'server_pid_observed':11368,'server_restarted':False,
      'input_inventory':{'practice_words':12,'train_signers':list(range(1,10)),'clips':108,'future_pairs':96,'hashes_match':True},
      'normal_variation_comparison_executed':False,'correctness_threshold_calibrated':False,
      'criteria_plan':'CRITERIA_PLAN.md','ui':'UI_CHECK.json','checks':'GRAPH_CHECK.json',
      'next':'Training-only cross-person96 descriptive baseline with alignment/form review; then labeled errors and separate-person validation',
      'files':{name:sha(B/'web'/name) for name in ['practice.js','practice.css','practice.html']}}
write(U/'COMPLETION.json',done)
write(U/'FINAL_GUARD.json',{'status':'PASS','all_guarded_files_unchanged':True,'original_dtw_values_unchanged_for_two_attempts':True})
context=R/'WordsInSilence_PROJECT_CONTEXT.md';s=context.read_text('utf-8')
entries={
'## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인':'''### 2026-10-13 DTW 가로 그래프·판정 기준 정리 완료
`ops/word100_dtw_graphs_20261013/COMPLETION.json` COMPLETE_VERIFIED_DISPLAY_AND_CRITERIA_PLAN. 기존2×2/5순위/상위3영상 유지. DTW3축 가로막대 추가: 굽힘각0~180도, 위치/궤적0~100어깨너비%. 같은색/고정눈금/실제수치 보존, 범위넘으면 초과화살표·값 표시, 미관측은0이 아닌측정불가. 그래프끝은합격선아님. 접는기준안내 추가.
정정: 예전18DTW도 최신core/recognize.py에서는 개인시도기반허용범위를사용하지않는 observed_difference_only_v1. 과거합격선이검증되어있고새화면에만빠졌다는해석은틀림. 새화면DTW연결은직전완료, 이번은시각화·기준설계이며손방향/구간별강조등이전세부기능전체이식은아님.
7개표시검사PASS(0/결측/180도/거리/초과/고정눈금), 실제브라우저319×594·1280×800 가로넘침없음, 두감사원수치일치, TCN/DTW백엔드·모델·사용자원본해시불변. 서버11368 재사용, 재시작/카메라/학습/추론재실행0. 구현·검산Codex, 다른에이전트검수아님.
12연습표현의학습자01~09 108개기존관절캐시를동결추출기록·해시확인. 01참고대비8명×12=96개정상변이예비비교가능; 이번에는수치분포계산안함. 최초경로가정오류는before기록보존후권위추출기록으로정정. CRITERIA_PLAN.md에정렬/수형변형검토→오류라벨→사람분리기준검증순서를정리. 사용자자료튜닝/보류자료열기/임계값생성없음.
''',
'## 9. 다음 단일 작업':'''### 2026-10-13 다음 — DTW 정상 변이 예비 조사 준비 완료
웹DTW가로그래프는완료. 다음은CRITERIA_PLAN.md에따라12표현×다른학습자8명=96개를고정참고와비교하고정렬/표현변형을검토하는오프라인단위. 아직실행안함. 정상분포의중앙값·사분위·지원관측만기록하며8개표본으로95%허용선이나정답기준을만들지않음. 사용자촬영/개발11~13/보류10·14~18을기준튜닝에섞지않음.
정답허용범위확정은정상/손모양·위치·움직임오류가확인된자료와사람별분리평가후. 정상자료만으로교정정답·비수어거절을입증하지않음. 현재원본/모델/DTW유지. 서버11368은재개때실제확인. 발표자료유예.
'''}
for heading,entry in entries.items():
    assert entry.split('\n')[0] not in s and heading+'\n' in s
    s=s.replace(heading+'\n',heading+'\n\n'+entry+'\n',1)
context.write_text(s,'utf-8')
sp=B/'ops/STATE.json';state=read(sp)
state['updated_date_kst']=now;state['current_stage']='WORD100_DTW_GRAPHS_AND_CRITERIA_PLAN_READY'
state['next_single_action']='12표현×다른학습자8명=96개 정상변이 예비비교와 정렬·표현변형검토. 아직 실행/허용선 확정 안함.'
state['word100_dtw_graphs_20261013']=done;write(sp,state)
h=R/'NEW_CHAT_HANDOFF.md';text=h.read_text('utf-8')
text=text.replace('## 마지막 완료','## 직전 완료 — DTW 수치 연결',1)
insert=f'''## 마지막 완료 — DTW 가로 그래프와 기준 설계 ({now})
`ops/word100_dtw_graphs_20261013` 완료.3축가로막대(각도0~180°,위치/궤적0~100어깨너비%),초과표시·결측별도·고정눈금·접는기준안내. 7개표시검사+실브라우저319×594/1280×800+두기존감사수치동일+백엔드/모델/사용자파일해시불변. UI_CHECK/GRAPH_CHECK/FINAL_GUARD/CRITERIA_PLAN.md 참고. 서버11368재사용, 카메라/학습/새추론0.
이전18DTW도개인기반허용선을비활성화한차이측정기였음. 새화면에검증된합격선만누락됐다는설명금지. 새100에는이미DTW시간정렬/굽힘각/위치/궤적연결, 이전손방향/상세구간UI전체이식아님.
다음: 학습자01~09의12표현108캐시실재·해시검증완료. 01참고대비96개정상변이예비비교는준비만했고아직계산하지않음. 표본8명분포만으로정답합격선금지. 정렬/같은표현변형검토→정상/오류라벨→기준설정과시험사람분리순서. 사용자자료튜닝/보류시험자동실행없음.

'''
text=text.replace('## 직전 완료 — DTW 수치 연결',insert+'## 직전 완료 — DTW 수치 연결',1)
lines=text.splitlines();lines[1]=f'갱신 {now}. AGENTS/CONTEXT 최신§3·§8·§9/WORKFLOW 우선.'
h.write_text('\n'.join(lines)+'\n','utf-8')
assert read(sp)['current_stage']=='WORD100_DTW_GRAPHS_AND_CRITERIA_PLAN_READY'
print(json.dumps({'status':done['status'],'checks':len(check['checks']),'guard':'PASS','future_pairs':96}))
