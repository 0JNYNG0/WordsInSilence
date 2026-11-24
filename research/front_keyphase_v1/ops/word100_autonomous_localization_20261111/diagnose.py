from localization_core import *
from collections import Counter
import shutil, os
assert not (U/'DIAGNOSTIC_COMPLETE.json').exists()
if not (U/'AUTHORIZATION.json').exists():
 write(U/'AUTHORIZATION.json',dict(user='그럼 검토한 1~4단계 계획대로 모두 진행해줘.',plan=str(Q/'PLAN_KO.txt'),plan_sha256=sha(Q/'PLAN_KO.txt'),public_only=True,no_private=True,no_download=True,no_external=True,no_camera=True,no_git=True,no_default_change=True))
 (U/'before').mkdir(exist_ok=True)
 for p in [ROOT/'WordsInSilence_PROJECT_CONTEXT.md',ROOT/'NEW_CHAT_HANDOFF.md',O/'STATE.json']:shutil.copy2(p,U/'before'/p.name)
 s=read(O/'STATE.json');s.update(current_stage=U.name,next_single_action='공개FIT 후보 소실 진단→시간별 탐색 학습→가이드 연결→고정비교·검산');s[U.name]=dict(status='FIT_DIAGNOSIS',execution=str(U/'CURRENT_EXECUTION.json'));write(O/'STATE.json',s)
 (ROOT/'NEW_CHAT_HANDOFF.md').write_text('# WordsInSilence 최신 인수인계\n\n사용자 4단계 전체 실행 승인. 현재 공개FIT의 후보 생성/선택/연결 진단 진행.\n단위 '+str(U)+'\n100단어899공통/기존 C·T3-F/웹 보존. 사용자·멘토 자료 접근/학습/튜닝/선택,카메라,다운로드,외부전송,Git 금지.\n이후 시간별 탐색 학습→설명 연결→4조건 비교·별도 검산.\n','utf-8')
torch.set_num_threads(4);torch.set_num_interop_threads(1)
write(U/'CURRENT_EXECUTION.json',dict(status='FIT_DIAGNOSIS',pid=os.getpid()))
net=jc.JointNet();net.load_state_dict(torch.load(J/'runs/42/model.pt',weights_only=True,map_location='cpu'))
adapter=torch.load(V/'runs/42/model.pt',weights_only=True,map_location='cpu')
net.quality.load_state_dict(adapter['quality']);net.pair.load_state_dict(adapter['pair']);net.eval()
rows=fitrows();diagnostics=[];truth_rows=[]
for r in rows:
 if not r['annotations']:continue
 z=np.load(B/'cache'/f"{r['clip_id']}.npz");x=prior.physical(r['x_full']);ts=r['timestamp_us'];fh=z['frozen_h_full']
 ob=jc.infer(net,x,ts,fh,[])
 p=U/'fit_diagnostic'/f"{r['clip_id']}.npz";p.parent.mkdir(exist_ok=True);np.savez_compressed(p,rp=ob['rp'],bp=ob['bp'],q=ob['q'])
 cand=ob['candidates'];selected=ob['selected'];truth=[]
 for ar in r['annotations']:
  k=ar['axis']*2+(ar['moving']=='L')
  for e in ar['events']:
   a,b=e['onset'][0],e['offset'][1];same=[v for v in cand if v['axis']==ar['axis'] and v['moving']==ar['moving']];ss=[v for v in selected if v['axis']==ar['axis'] and v['moving']==ar['moving']]
   ov=lambda v:lc.be.ev.iou([a,b],[v['a'],v['b']])
   best=max(same,key=ov) if same else None;sel=max(ss,key=ov) if ss else None
   t=dict(clip_id=r['clip_id'],person=r['person'],axis=ar['axis'],moving=ar['moving'],span=[a,b],direction=e['direction'],candidate_overlap=ov(best) if best else 0.,selected_overlap=ov(sel) if sel else 0.,candidate=best,selected=sel,activity=float(ob['bp'][a:b+1,k,0].mean()),onset_max=float(ob['bp'][e['onset'][0]:e['onset'][1]+1,k,1].max()),offset_max=float(ob['bp'][e['offset'][0]:e['offset'][1]+1,k,2].max()),valid=float((ob['q'][a:b+1,k,-2:].max(-1)>.5).mean()))
   t['failure_stage']='selected' if t['selected_overlap']>=.5 else 'selection' if t['candidate_overlap']>=.5 else 'candidate_generation'
   truth.append(t);truth_rows.append(t)
 diagnostics.append(dict(clip_id=r['clip_id'],source_sha256=r['source_sha256'],arrays=str(p),arrays_sha256=sha(p),candidates=cand,selected=selected,links=ob['links'],truth=truth))
 event('fit_diagnosed',clip=r['clip_id'],truth=len(truth),candidates=len(cand),selected=len(selected))
write(U/'FIT_DIAGNOSTIC.json',dict(rows=diagnostics,summary=dict(Counter(v['failure_stage'] for v in truth_rows)),old_frozen_eval_reruns=0,fit_only=True))
write(U/'DIAGNOSTIC_COMPLETE.json',dict(videos=len(diagnostics),events=len(truth_rows),summary=dict(Counter(v['failure_stage'] for v in truth_rows))))
write(U/'CURRENT_EXECUTION.json',dict(status='FIT_DIAGNOSIS_COMPLETE',pid=None))
