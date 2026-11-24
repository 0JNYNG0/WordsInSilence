"""Exchange finite local-scale denominators only; retain each observation branch."""
from common import *
import argparse,shutil,traceback
from collections import Counter
SEEDS=[20261013,20261014,20261015]
def freeze():
 assert not (U/'FROZEN_PROTOCOL.json').exists()
 for n in ['WordsInSilence_PROJECT_CONTEXT.md','NEW_CHAT_HANDOFF.md','research/front_keyphase_v1/ops/STATE.json']:
  p=U/'before'/Path(n).name;p.parent.mkdir(exist_ok=True);shutil.copy2(ROOT/n,p)
 write(U/'AUTHORIZATION.json',{'at':now(),'user_request':'그래 그럼 손 배정과 손 크기 기준을 분리한 두 조건을 각각 3회씩, 총 6회 비교해줘. 기존 C는 유지하고 내 촬영은 학습이나 튜닝에 쓰지 말고, 어떤 변화가 성능에 영향을 줬는지와 다음 보완 방향을 결과로 정리해줘. 중요한 문제 생기면 바로 알려줘.',
 'executor':'Codex local','authorized':'Input preparation, separate verification, six paired training runs, saved C/Q comparison, report.','excluded':['user/mentor data','production replacement','held-out data','camera','download','external transfer','Git']},new=True)
 p=read(PLAN/'NEXT_PLAN.json');p.update(status='USER_APPROVED_INPUT_GATE_REQUIRED',authorization=str(U/'AUTHORIZATION.json'),arms={'R10':'Q observations, C scale when both finite','R01':'C observations, Q scale when both finite'},seeds=SEEDS,
  training_contract=read(PREV/'FROZEN_PROTOCOL.json')['training_if_later_approved_and_ready'],
  contrasts={'assignment_at_C_scale':'R10-C','assignment_at_Q_scale':'Q-R01','finite_scale_at_C_observation':'R01-C','finite_scale_at_Q_observation':'Q-R10','interaction':'Q-R10-R01+C'},
  semantic_gate='No coordinate routing, mask, timestamp, FOV, temporal interval or missingness changes relative to respective frozen observation branches; prior C/Q semantic reviews reused by hash. Only finite local coordinate denominators change.',
  interpretation='Each cell retrains and selects its own checkpoint by the same frozen CE criterion. Finite-scale effect, not all scale availability; independent real-world generalization remains untested. Off-diagonal arms are diagnostic only, never deployed.')
 paths=[PLAN/'NEXT_PLAN.json',PLAN/'FINAL_AUDIT.json',PREV/'PREP_CONTEXT.json',PREV/'READY.json',PREV/'TRAIN_CONTEXT.json',PREV/'COMPARISON_INDEPENDENT.json',PREV/'TRAIN_VISUAL_READY.json',PREV/'DEV_CHANGE_REVIEW.json',BASE/'training_CONTEXT.json',BASE/'training_core.py',ROOT/'research/sign_model_pilot_v1/src/pilot/model_masked_v1.py',ROOT/'research/sign_model_pilot_v1/src/pilot/data.py']
 paths += [U/n for n in ['common.py','prepare.py','verify_inputs.py','train.py','verify_results.py','AUTHORIZATION.json']]
 for a in ['C','Q']:
  for s in SEEDS:
   prefix=BASE/f'training_C_{s}' if a=='C' else PREV/f'Q_{s}'
   paths.extend([Path(str(prefix)+'_RESULT.json'),Path(str(prefix)+'_best.pt')])
 p['source_and_code_pins']={str(x):sha(x) for x in paths};write(U/'FROZEN_PROTOCOL.json',p,new=True)
 event(stage='PROTOCOL_FROZEN',training_runs=0)
def hs(xy,m,b):
 vals=[]
 for t in range(len(xy)):
  if m[t,b]:
   for k in [5,9,13,17]:
    if m[t,b+k]:vals.append(float(np.linalg.norm(xy[t,b+k]-xy[t,b])))
 s=float(np.median(vals)) if vals else float('nan')
 return s if s>1e-6 else float('nan')
def exchange(x,xy,m,own,other):
 out=x.copy()
 for side,b in enumerate([33,54]):
  if not (np.isfinite(own[side]) and np.isfinite(other[side])) or own[side]==other[side]:continue
  for j in range(21):
   ok=m[:,b]&m[:,b+j];k=side*21+j
   out[ok,102+2*k:104+2*k]=(xy[ok,b+j]-xy[ok,b])/other[side]
 return out
def main():
 protocol=read(U/'FROZEN_PROTOCOL.json');check(protocol['source_and_code_pins']);assert not (U/'PREP_RUNNING.json').exists()
 oldctx=read(PREV/'PREP_CONTEXT.json');rows=oldctx['rows'];assert Counter(r['role'] for r in rows)=={'train_candidate':899,'development_only':300}
 assert [r['clip_id'] for r in rows]==[r['clip_id'] for r in read(BASE/'training_CONTEXT.json')['ordered_rows']]
 expected=oldctx['pins'];pins={};counts=Counter();ledger=[]
 def source(path):
  p=str(Path(path).resolve());h=sha(p)
  if p in expected:assert expected[p]==h
  pins[p]=h;return load(p)
 lock=U/'PREP_RUNNING.json';write(lock,{'at':now(),'pid':os.getpid()},new=True)
 try:
  for i,r in enumerate(rows):
   cid=r['clip_id'];rp=PREV/'records'/f'{cid}.json';rec=read(rp);pins[str(rp)]=sha(rp);expected.update(rec['outputs'])
   out={'clip_id':cid,'role':r['role'],'signer_id':r['signer_id'],'label_index':r['label_index'],'views':{},'outputs':{}}
   for v in VIEWS:
    c=source(r['views'][v]['channels']);co=source(r['views'][v]['observation']);q=source(rec['views'][v]['Q']);branch={a:{} for a in ARMS}
    for t in ['whole','context'] if r['role']=='train_candidate' else ['whole']:
     ids=c[t+'_frame_id'];assert np.array_equal(ids,q[t+'_frame_id']) and np.array_equal(c[t+'_ts'],q[t+'_ts'])
     ci=np.searchsorted(co['frame_id'],ids);qi=np.searchsorted(q['frame_id'],ids);assert np.array_equal(co['frame_id'][ci],ids) and np.array_equal(q['frame_id'][qi],ids)
     cx,cm=co['xy_px'][ci],co['usable'][ci];qx,qm=q['xy_px'][qi],q['usable'][qi];cs=[hs(cx,cm,b) for b in [33,54]];qs=[hs(qx,qm,b) for b in [33,54]]
     for a,x,xy,m,own,other in [('R10',q['x_'+t],qx,qm,qs,cs),('R01',c['x_'+t],cx,cm,cs,qs)]:
      z=exchange(x,xy,m,own,other);assert z.dtype==np.float32 and np.isfinite(z).all() and np.array_equal(z[:,:102],x[:,:102]) and np.array_equal(z[:,186:],x[:,186:])
      branch[a]['x_'+t]=z;branch[a][t+'_frame_id']=ids;branch[a][t+'_ts']=c[t+'_ts'];counts[r['role']+'_tensors']+=1
      changed=not np.array_equal(z,x);counts[a+'_'+r['role']+'_changed_tensors']+=changed
      if changed:ledger.append({'clip':cid,'role':r['role'],'view':v,'temporal':t,'arm':a,'own_scale':[float(s) if np.isfinite(s) else None for s in own],'exchanged_scale':[float(s) if np.isfinite(s) else None for s in other],'changed_values':int(np.count_nonzero(z!=x)),'max_abs_change':float(np.abs(z-x).max())})
    out['views'][v]={}
    for a in ARMS:
     p=U/'inputs'/f'{cid}_{v}_{a}.npz';p.parent.mkdir(exist_ok=True);assert not p.exists();np.savez_compressed(p,**branch[a]);out['views'][v][a]=str(p);out['outputs'][str(p)]=sha(p)
   write(U/'records'/f'{cid}.json',out,new=True);counts[r['role']+'_sources']+=1
   if (i+1)%100==0:event(stage='PREPARING_FINITE_SCALE_INPUTS',completed=i+1,total=1199)
  write(U/'PREP_CONTEXT.json',{'at':now(),'rows':rows,'pins':{**protocol['source_and_code_pins'],**pins,str(U/'FROZEN_PROTOCOL.json'):sha(U/'FROZEN_PROTOCOL.json')},'roles':{'TRAIN':899,'DEV':300},'channels':280,'arms':ARMS},new=True)
  write(U/'PREP_RESULT.json',{'status':'PREPARED_PENDING_VERIFICATION','counts':dict(counts),'changed':ledger,'user_data_used':False},new=True)
  check(pins);lock.unlink();event(stage='INPUT_PREPARATION_COMPLETE_PENDING_VERIFICATION',counts=dict(counts))
 except BaseException:
  if not (U/'FIRST_PREP_ERROR.json').exists():write(U/'FIRST_PREP_ERROR.json',{'at':now(),'traceback':traceback.format_exc()},new=True)
  raise
if __name__=='__main__':
 if '--freeze' in sys.argv:freeze()
 else:main()
