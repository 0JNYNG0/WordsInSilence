from motion_core import *
import shutil
from collections import Counter
assert not (U/'FROZEN.json').exists();ann=read(J/'ALL_FIT_BOUNDARIES.json')['rows'];rr=[r for r in old.rows() if r['role']=='fit'];assert len(rr)==899 and len({r['class_index'] for r in rr})==100
samples=[];audit=[];checks=[]
for r in rr:
 rows=[a for a in ann if a['clip_id']==r['clip_id']]
 for ar in rows:
  assert sha(ar['sheet'])==ar['sheet_sha256'] and ar['source_sha256']==r['source_sha256']
  for typ,items in [('motion',ar['events']),('hold',[dict(onset=[a,a],offset=[b,b],direction=1) for a,b in ar['holds']])]:
   for ei,e in enumerate(items):
    target=e['direction'];meta=dict(clip_id=r['clip_id'],person=r['person'],axis=ar['axis'],moving=ar['moving'],anchor=ar['anchor'],kind=typ,event=ei,onset=e['onset'],offset=e['offset'],target=target,source_sha256=r['source_sha256'])
    a,b=e['onset'][0],e['offset'][1];g,ok,info=prior.geometry(prior.physical(r['x_full']),r['timestamp_us'],a,b,ar['axis'],ar['moving'],ar['anchor']);audit.append(dict(meta,eligible=bool(ok),geometry=info,sign_conflict=bool(ok and target!=1 and np.sign(info['signed_change'])!=target-1)))
    for view in ['full','upper']:
     x=prior.physical(r['x_'+view])
     for a in sorted(set(e['onset'])):
      for b in sorted(set(e['offset'])):
       if b-a<3:continue
       for rev in [False,True]:
        g,ok,info=prior.geometry(x,r['timestamp_us'],a,b,ar['axis'],ar['moving'],ar['anchor'],rev)
        if ok:samples.append(dict(meta,a=a,b=b,view=view,reverse=rev,label=2-target if rev else target,g=g.tolist()))
write(U/'FIT_LABEL_AUDIT.json',dict(rows=audit,disallowed_background_ranges=52,review='Existing direct RGB ranges reused with hashes; same Codex review, not expert ground truth.'))
write(U/'TRAIN_SAMPLES.json',dict(rows=samples,common100_videos_preserved=899,learned_encoder_frozen=True,unique_movement_intervals=39,unique_hold_intervals=22,augmented_examples_not_independent=True))
cfg=dict(seeds=[42,43,44],primary_seed=42,epochs=100,lr=.0005,batch=32,probe_ids=[3,7],rescue_confidence=.90,completion_tail_us=120000,scope='Fine tune geometry motion conv+GRU+head, retain frozen100word encoder/shape/direction baseline. No expected word or desired direction as input.',evaluation='Historical78 + previous16 public originals as development replay; reserve six extra-expression public ID13 before scores. Manual and automatic intervals, direction-only/interval-only/combined separated.',adoption='No new normal/speed/crossing warning; keep old confident directions and shape; improve actual motion without labeling holds as motion; no operational default replacement.',protections=read(J/'FROZEN.json')['protections'],pins={str(p):sha(p) for p in [J/'ALL_FIT_BOUNDARIES.json',J/'FINAL_DECISION.json',J/'RESULT.json',V/'RESULT.json',V/'runs/42/model.pt',old.P/'runs/coupled_42/model.pt',U/'TRAIN_SAMPLES.json']})
write(U/'FROZEN.json',cfg);write(U/'AUTHORIZATION.json',dict(user='Current approval: evidence-based remaining-problem implementation, validation, overall progress and detailed next design.',public_only=True,no_private=True,no_download=True,no_external=True,no_camera=True,no_git=True,no_default_change=True));write(U/'CURRENT_EXECUTION.json',dict(status='PREPARED',pid=None));event('prepared',examples=len(samples),sign_conflicts=sum(r['sign_conflict'] for r in audit))
for p in [ROOT/'WordsInSilence_PROJECT_CONTEXT.md',ROOT/'NEW_CHAT_HANDOFF.md',O/'STATE.json']:(U/'before').mkdir(exist_ok=True);shutil.copy2(p,U/'before'/p.name)
summary='짧은 이동/유지 판정부 미세조정 및 완료근거 분리 준비.100단어899의 공통부/기존모양방향/C·T3F/웹 보존,공개FIT39이동22유지만감독.개인자료0.';nxt='FIT정답·좌표감사→고정3초기값학습→기존/추가공개ID13 비교→별도검산·전체목표진행도/다음계획 보고.'
p=ROOT/'WordsInSilence_PROJECT_CONTEXT.md';t=p.read_text('utf-8')
for sec,body in [('## 8. 현재 상태',summary),('## 9. 다음 단일 작업',nxt)]:
 i=t.index('\n',t.index(sec))+1;t=t[:i]+'\n### 2026-11-11 짧은 움직임·완료 근거 보완 진행\n\n'+body+'\n근거 '+str(U)+'\n'+t[i:]
p.write_text(t,'utf-8');s=read(O/'STATE.json');s.update(current_stage=U.name,next_single_action=nxt);s[U.name]=dict(status='PREPARED',summary=summary,next=nxt,execution=str(U/'CURRENT_EXECUTION.json'));write(O/'STATE.json',s);(ROOT/'NEW_CHAT_HANDOFF.md').write_text('# WordsInSilence 최신 인수인계\n\n'+summary+'\n다음: '+nxt+'\n현재단위 '+str(U)+'\n','utf-8')
