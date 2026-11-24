"""Close this comparison milestone, keep the user's larger improvement goal open."""
from pathlib import Path
from collections import Counter
import sys,json,math,hashlib,datetime,html
import numpy as np
import torch
sys.dont_write_bytecode=True;sys.stdout.reconfigure(encoding='utf-8')
R=Path('C:/MyUnityProjects/WordsInSilence');O=R/'research/front_keyphase_v1/ops'
U=O/'word100_boundary_aux_20261026';P=O/'word100_boundary_evidence_20261026';B=O/'word100_body_boundary_20261017';T=O/'word100_frozen_c_temporal_20261019'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):
 with Path(p).open('x',encoding='utf-8') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False)
ctx=read(B/'training_CONTEXT.json');tr=[r for r in ctx['ordered_rows'] if r['role']=='train_candidate'];dv=[r for r in ctx['ordered_rows'] if r['role']=='development_only'];ids={r['clip_id'] for r in tr}
ver=read(U/'VERIFIED.json');assert ver['status']=='PASS'
f=read(U/'FROZEN.json')
for p,h in f['pins'].items():assert sha(p)==h
cf=read(U/'CONTROL_FROZEN.json');tf=read(U/'TRANSFER_FROZEN.json')
for p,h in tf['pins'].items():assert sha(p)==h
controls={v:{k:0 for k in ('clean','start_spike','end_spike','both_spikes')} for v in ('masked','sensor')}
control_rows=0;control_trace_batches=0;control_ce_errors=[];control_epoch=[];transfer={};transfer_rows=0
for seed in f['seeds']:
 c=read(U/f'CONTROL_{seed}_RESULT.json');assert c['protocol_sha256']==sha(U/'CONTROL_FROZEN.json')
 assert c['checkpoint_sha256']==sha(U/f'CONTROL_{seed}_best.pt')
 ck=torch.load(U/f'CONTROL_{seed}_best.pt',map_location='cpu',weights_only=True)
 old=torch.load(T/f'T3F_{seed}_best.pt',map_location='cpu',weights_only=True)
 assert all(torch.equal(v,ck['state_dict'][k]) for k,v in old['state_dict'].items() if k!='phase_weight')
 best=0;ce=c['initial_ce']
 for h in c['history']:
  if h['clean_dev_ce']<ce-1e-6:best=h['epoch'];ce=h['clean_dev_ce']
  assert best==h['selected_epoch']
 assert best==c['selected_epoch']==ck['epoch'] and ce==c['selected_ce'];control_epoch.append(best)
 trace=U/f'CONTROL_{seed}_TRACE.jsonl';assert sha(trace)==c['trace_sha256'];eps={i:[] for i in range(1,41)}
 for line in trace.read_text(encoding='utf-8').splitlines():
  q=json.loads(line);e=q['epoch'];control_trace_batches+=1
  for cid,v,t,kind in q['clips']:
   assert cid in ids and kind=='clean_twice'
   expt='x_context' if (e+int(hashlib.sha256(cid.encode()).hexdigest(),16)%2)%2 else 'x_whole'
   expv='sensor' if ((e-1)//2+int(hashlib.sha256(('sensor_view:'+cid).encode()).hexdigest(),16)%2)%2 else 'masked'
   assert v==expv and t==expt;eps[e].append(cid)
 for e,seen in eps.items():assert seen==[tr[i]['clip_id'] for i in np.random.default_rng(seed+e).permutation(899)]
 with np.load(U/f'features_{seed}.npz',allow_pickle=False) as z:dev=z['dev']
 w=ck['state_dict']['phase_weight'].numpy().astype(float);ces=[]
 for vi,v in enumerate(('masked','sensor')):
  for ki,k in enumerate(('clean','start_spike','end_spike','both_spikes')):
   pred={r['clip_id']:r for r in c['predictions'][v][k]}
   expected=dev[:,vi,0,ki,:100].astype(float)+dev[:,vi,0,ki,100:].astype(float)@w.T
   for i,row in enumerate(dv):
    pp=pred[row['clip_id']];zz=np.array(pp['logits']);assert np.allclose(zz,expected[i],atol=5e-5,rtol=5e-5)
    rank=sorted(range(100),key=lambda j:(-zz[j],j));ok=rank[0]==row['label_index']
    assert ok==pp['correct'] and rank[:5]==pp['top5'];controls[v][k]+=ok;control_rows+=1
    if k=='clean':
     mx=float(max(zz));ces.append(mx+math.log(sum(math.exp(float(x)-mx) for x in zz))-zz[row['label_index']])
 assert abs(sum(ces)/600-c['selected_ce'])<2e-6;control_ce_errors.append(float(abs(sum(ces)/600-c['selected_ce'])))
 tt=read(U/f'TRANSFER_{seed}.json');assert tt['protocol_sha256']==sha(U/'TRANSFER_FROZEN.json')
 assert tt['logits_sha256']==sha(U/f'TRANSFER_{seed}_LOGITS.npz');assert tt['ids']==[r['clip_id'] for r in dv]
 with np.load(U/f'TRANSFER_{seed}_LOGITS.npz',allow_pickle=False) as z:logits=z['logits']
 assert logits.shape==(300,12,3,100) and np.isfinite(logits).all()
 for ci,(v,sec,k) in enumerate(tt['conditions']):
  key=f'{v}/{sec}/{k}';count={m:0 for m in ('original','control','augmented')}
  for i,label in enumerate(tt['labels']):
   for mi,m in enumerate(count):
    rank=sorted(range(100),key=lambda j:(-float(logits[i,ci,mi,j]),j));count[m]+=rank[0]==label;transfer_rows+=1
  assert count==tt['counts'][key]
  target=transfer.setdefault(key,dict(original=0,control=0,augmented=0,n=0))
  for m,x in count.items():target[m]+=x
  target['n']+=300

# Independently recount the causal scale diagnostic; never a deployable oracle.
scale={}
for m in ('C13','T3F13'):
 rr=read(P/f'{m}_SCALE_FIXED_SCORES.json');previous=read(P/f'{m}_STRESS_SCORES.json')
 scale[m]={}
 for v,parts in rr['scores'].items():
  scale[m][v]={}
  for k,pp in parts.items():
   base={r['clip_id']:r for r in previous['scores'][v][k]};g=l=0
   for q in pp:
    order=sorted(range(100),key=lambda j:(-q['logits'][j],j));ok=order[0]==q['label_index'];assert ok==q['correct']
    g+=ok and not base[q['clip_id']]['correct'];l+=not ok and base[q['clip_id']]['correct']
   cc=rr['counts'][v][k];assert (g,l)==(cc['gained'],cc['lost']);scale[m][v][k]=cc

audit=dict(status='PASS',control_rows_verified=control_rows,control_batches_verified=control_trace_batches,
 control_selected_epochs=control_epoch,control_clean_ce_max_difference=max(control_ce_errors),
 transfer_prediction_rows_recounted=transfer_rows,transfer=transfer,control_counts=controls,scale_ablation=scale,
 author='Codex',verification='separate saved arithmetic and checkpoint/cache/trace audit by same Codex, not human ground truth')
write(U/'CONTROL_TRANSFER_VERIFIED.json',audit)

now=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat()
normal_losses=[r for r in ver['losses'] if r['kind']=='clean'];normal_gains=[r for r in ver['gains'] if r['kind']=='clean']
labelmap={r['clip_id']:ctx['labels'][r['label_index']] for r in dv}
for q in normal_losses+normal_gains:q['label']=labelmap[q['clip_id']]
result=dict(status='LIMITED_DEVELOPMENT_ROBUSTNESS_GAIN_NOT_DEPLOYED',at=now,goal_complete=False,
 candidate_acceptance=ver['candidate_acceptance'],normal_and_stress=ver['totals'],clean_continuation=controls,
 transfer=transfer,normal_regressions=normal_losses,normal_gains=normal_gains,
 source_scope='TRAIN899 DEV300 only; same DEV repeated across seed,view,synthetic stress. No additional199/200/user/mentor reads.',
 changed='New 9600-parameter time auxiliary weights only, fixed C tensors. Original files and operating models preserved.',
 progress='Boundary diagnosis,3 augmentation training runs,3 matched clean controls, alternate-length stress, arithmetic verification completed. Larger hand/feedback goal remains active.',
 next='Integrate a research-only evidence contract separating visible position/finger shape/identity/temporal estimates from word classification. Audit existing DTW missing-evidence paths using saved public samples. Keep comparison denominator fixed; never turn missing evidence into a correct-motion or wrong-motion verdict. Do not deploy the new head or retry tuning on its regressions.',
 remaining=['Natural webcam improvement not measured; original DEV average unchanged and individual regressions exist.',
            'Synthetic edge sensitivity is not proved cause of past user failures.',
            'YOLO/HaMeR hand evidence candidates not integrated into production; existing ten identity conflicts remain.',
            'No DTW feedback correctness improvement shown by this classifier experiment.'])
write(U/'RESULT.json',result)
lines=['지속 개선 중간 결과 — 시작·종료 관측 흔들림 보강',
 '전체 목표는 진행 중입니다. 이번 결과는 손 복원이나 피드백 전체 완성이 아닙니다.',
 '', '1. 무엇을 확인했나',
 '공개1199영상×2관측경로의 기존 입력을 정확히 재구성했습니다. 완전히 손이 없는 대기1초를 양쪽에 추가하면 입력이 그대로였습니다.',
 '반면 양끝의 잘못된 한 점을 흉내 내면 동작을 그대로 보존해도 인식이 흔들렸습니다. 이는 합성 시험이며 실제 사용자 오류 원인 확정이 아닙니다.',
 '몸 크기 계산을 원래대로 복원해도 대부분 남았으므로 길어진 시간 범위와 특징 요약의 영향이 함께 있었습니다. 손 크기 기준 자체는 이 시험에서 바뀌지 않았습니다.',
 '', '2. 무엇을 바꿨나',
 '기존 C의 몸·손 인식 특징은 고정하고, T3-F의 작은 시간 분석 부분만 별도 파일로 보강했습니다. 원래 자료와 경계 흔들림 자료를 같은 비중으로 학습했습니다.',
 '정상 손·교차·중간 누락·프레임을 삭제하지 않았습니다. 정상 경계에서도 한두 점만 보일 수 있으므로 희소 관절을 일괄 삭제하지 않았습니다.',
 '공개 TRAIN899만 학습. 세 초기모델 각각40epoch. 단순 추가 학습의 효과를 구분하려고 원래 자료만 더 학습한 대조군3회도 비교했습니다.',
 '', '3. 결과 — 같은 DEV300의 세 모델 평균']
for v,ko in [('sensor','잘린 화면 검출 조건'),('masked','전체 검출 후 영역 제한 조건')]:
 n=ver['totals'][v]['clean'];before=sum(ver['totals'][v][k]['before'] for k in ('start_spike','end_spike','both_spikes'));after=sum(ver['totals'][v][k]['after'] for k in ('start_spike','end_spike','both_spikes'))
 control=sum(controls[v][k] for k in ('start_spike','end_spike','both_spikes'))
 lines.extend([f'{ko}: 원래 영상 {n["before_percent"]:.2f}% → {n["after_percent"]:.2f}%.',
  f'1초 경계 합성시험: {100*before/2700:.2f}% → {100*after/2700:.2f}% ({100*(after-before)/2700:+.2f}%p). 원래 자료만 더 학습한 대조군 {100*control/2700:.2f}%.'])
lines+=['평균은 유지됐지만 원래 영상에서도 개별 정답이 오답으로 바뀐 사례가 있습니다. 세 모델·두 관측 합산 정상 개선3건/퇴보3건이며 서로 독립된 사람 수가 아닙니다.',
 '단순 추가 학습 대조군의 정상 전체검출조건은892/900, 보강 후보는891/900으로 1건 낮습니다. 어느 조건에서나 더 좋은 모델이라는 뜻이 아닙니다.',
 '', '4. 학습 때 쓰지 않은 대기 길이0.5초/1.5초 — 같은 DEV의 합성 조건 확장']
for v,ko in [('sensor','잘린 화면 검출'),('masked','전체 검출 뒤 영역 제한')]:
 for sec in (.5,1.5):
  items=[m for k,m in transfer.items() if k.startswith(f'{v}/{sec}/')];n=sum(m['n'] for m in items)
  vals={k:sum(m[k] for m in items) for k in ('original','control','augmented')}
  lines.append(f'{ko} {sec}초: 기존 {100*vals["original"]/n:.2f}% / 추가학습 대조 {100*vals["control"]/n:.2f}% / 보강 {100*vals["augmented"]/n:.2f}%.')
lines+=['같은 영상의 조건을 바꾼 시험이므로 신규 사용자 정확도나 새 독립 평가로 발표하면 안 됩니다.',
 '', '5. YOLO와 피드백의 현재 위치',
 'MediaPipe 기본 검출→재검출→남은 누락에 선택적 YOLO 보조는 연구 부품으로 시험했습니다. 운영 C·T3-F 웹에서 항상 함께 실행되는 상태는 아닙니다.',
 'HaMeR도 별도 연구 후보입니다. 손 위치·손가락 모양·좌우 신원을 각각 검증하며, 하나를 찾았다고 나머지가 정확하다고 취급하지 않습니다.',
 'TCN은 단어를 구분합니다. DTW는 목표 단어의 여러 참고와 시간축을 맞춰 차이를 계산합니다. 관측 불충분과 실제 동작 차이는 별도로 표시해야 합니다.',
 '', '6. 계속할 일',result['next'],
 '새 후보는 연구용으로만 보존했습니다. 기존 C·T3-F·운영 웹·DTW 변경0, 사용자·멘토 촬영 접근/학습/튜닝0. 이 작은 비교를 끝났다고 전체 목표를 종료하지 않습니다.',
 '구현 및 별도 수치검산: 동일 Codex. 독립 사람·수어 전문가의 정답 검증은 아닙니다.']
text='\n'.join(lines)+'\n';(U/'RESULT_KO.txt').write_text(text,encoding='utf-8')
body=''.join('<p>'+html.escape(line)+'</p>' for line in lines if line)
(U/'RESULT_KO.html').write_text('<!doctype html><html lang="ko"><meta charset="utf-8"><title>경계 관측 보강 중간 결과</title><style>body{max-width:1000px;margin:48px auto;padding:24px;font:18px/1.75 system-ui;background:#f5f7fb;color:#172433}p:first-child{font-size:32px;font-weight:750}p{margin:12px 0}</style><body>'+body+'</body></html>',encoding='utf-8')

# Keep original historical result files untouched; update only latest state.
summary='공개1199×2경로 입력재구성·빈대기불변 검산 완료. 경계 한 점 합성오염에서 인식이 흔들림; 몸크기 복원만으로 대부분 미해결. C를 고정하고 T3F 시간보조만 경계 보강3회(40epoch), clean추가학습대조3회, 0.5/1.5초 합성확장 완료. 검산 PASS. 원래 DEV 평균 sensor99.22%/masked99.00% 유지; 1초합성 평균 sensor97.41→98.00%,masked96.89→97.48%,각+0.59%p. 정상 개별 개선3/퇴보3(3seed×2view 중복); 새 실사용 개선 아님. 연구후보 보존/운영미적용. 손복원10충돌과DTW피드백근거과제는 남음. 전체목표active.'
nextstep=result['next']+' 기존공개저장근거·현재DTW코드를먼저확인하고재사용. 소규모비교완료로목표닫지않음. 사용자/멘토자료와추가199/200접근·튜닝금지.'
context=R/'WordsInSilence_PROJECT_CONTEXT.md';tx=context.read_text(encoding='utf-8-sig')
for prefix,title,content in [('## 8.','### 2026-10-26 경계 보강 연구 후보 검증 — 제한적 견고성 개선·전체목표 진행',summary),('## 9.','### 2026-10-26 지속 개선 — 관측 근거와 DTW 피드백 연구 연결',nextstep)]:
 if title not in tx:
  pos=tx.index(prefix);end=tx.index('\n',pos);tx=tx[:end+1]+'\n'+title+'\n'+content+'\n\n'+tx[end+1:]
context.write_text(tx,encoding='utf-8')
sp=O/'STATE.json';s=read(sp);s.update(updated_at=now,current_stage='AUTONOMOUS_IMPROVEMENT_EVIDENCE_INTEGRATION_NEXT',next_single_action=nextstep)
s['word100_boundary_evidence_20261026'].update(status='VERIFIED_DIAGNOSTIC',execution_session=None,summary='2398입력/3600스트레스예측 검산PASS. 완료; 재실행불필요.')
s['word100_boundary_aux_20261026']={'status':result['status'],'summary':summary,'unit':str(U),'goal_complete':False,'running_processes':[],'report':str(U/'RESULT_KO.html'),'next':nextstep}
sp.write_text(json.dumps(s,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
(R/'NEW_CHAT_HANDOFF.md').write_text('# WordsInSilence 최신 인수인계\n\n갱신 '+now+'\n현재: '+summary+'\n\n다음: '+nextstep+'\n\n근거: '+str(U/'RESULT_KO.txt')+' / VERIFIED.json / CONTROL_TRANSFER_VERIFIED.json. 경계입력 원인검증은 '+str(P)+'. 완료실행33891·74910·80845,현재잠금없음. 기존 C/T3F/웹/DTW 불변. persistent goal active. 과거E3 신규199는 이미노출평가이며튠금지. 원본/W:/sources/읽기전용. 별도heartbeat없음.\n',encoding='utf-8')
plan=read(P/'WORK_PLAN.json');plan['tracks'][0]['status']='VERIFIED';plan['tracks'][1]['status']='LIMITED_RESEARCH_CANDIDATE_VERIFIED';plan['tracks'][2]['status']='NEXT';plan['current_unit']=str(U);plan['execution_session']=None;plan['updated_at']=now;plan['next']=nextstep
write(U/'WORK_PLAN.json',plan)
print('Comparison verified and progress recorded. Larger goal remains active.')
print(json.dumps(dict(transfer=transfer,normal_regressions=normal_losses,normal_gains=normal_gains),ensure_ascii=False))
