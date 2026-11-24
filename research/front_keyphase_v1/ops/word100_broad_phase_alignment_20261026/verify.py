import study as S
import numpy as np
from collections import Counter,defaultdict
def blank():return dict(cases=0,time_old_sum=0.,time_new_sum=0.,core_frames=0,sequence_old=0,sequence_new=0,lost_support=0,fallback=0,quality_old=0,quality_new=0,groups={g:dict(cells=0,old_sum=0.,new_sum=0.) for g in S.GROUPS})
def phase(x):
 y=x.copy()
 for i in range(2,len(x)-2):
  v=(x[i-2]+4*x[i-1]+6*x[i]+4*x[i+1]+x[i+2])/16
  v=min(max(v,x[1]),x[-2]);y[i]=min(max(v,x[i]-1),x[i]+1)
 return y
def dp_total(cost):
 n,m=cost.shape;w=max(1,round(.2*max(n,m)));acc=np.full((n+1,m+1),np.inf);acc[0,0]=0.
 for i in range(n):
  c=i*(m-1)/(n-1)
  for j in range(max(0,int(np.floor(c-w))),min(m-1,int(np.ceil(c+w)))+1):
   acc[i+1,j+1]=cost[i,j]+min(acc[i,j],acc[i,j+1],acc[i+1,j])
 return acc[-1,-1]
def independent_descriptor(values,valid,first,last,index,domain):
 n=len(index);out=np.full(domain.shape+(5,),np.nan);ok=np.zeros(domain.shape,bool)
 a=np.floor(index).astype(int);b=np.ceil(index).astype(int);w=(index-a)[:,None,None]
 mv=valid[a]&valid[b];v=np.where(mv,values[a]*(1-w)+values[b]*w,np.nan)
 for j in range(2,n-2):
  for h in (0,1):
   use=domain[j-2:j+3,h].all(0)&valid[first[j-2]:last[j+2]+1,h].all(0)&mv[[j-2,j,j+2],h].all(0)
   components=np.stack([v[j-2,h],v[j,h],v[j+2,h],v[j,h]-v[j-2,h],v[j+2,h]-v[j,h]],-1)
   ok[j,h]=use;out[j,h]=np.where(use[:,None],components,np.nan)
 return out,ok
def main():
 f=S.read(S.U/'FROZEN.json');r=S.read(S.U/'RESULT.json');S.check(f['pins']);S.check(f['input_pins']);S.check(f['protected']);S.check(r['pins']);assert len(r['rows'])==4500
 assert not (S.U/'RUNNING.json').exists();by={row['clip_id']:row for row in f['rows']};assert len(by)==300
 allk=set(S.KINDS+S.MISSING);seen=defaultdict(set);totals=defaultdict(blank);reg=[];omissions=[];extra=Counter();normal_clip={}
 subset=set(sorted(by)[:3]);independent=0
 for row in r['rows']:
  cid=row['clip_id'];kind=row['kind'];assert kind not in seen[cid];seen[cid].add(kind);o=S.load(S.U/'cases'/(row['name']+'.npz'));z=S.load(by[cid]['output']);n=len(z['core']);core=z['core'];path=o['path'];ix=o['source_index']
  assert np.array_equal(o['core'],core) and np.array_equal(o['domain'],np.broadcast_to(core[:,None,None],(n,2,23)))
  assert np.array_equal(path[0],[0,0]) and np.array_equal(path[-1],[len(ix)-1,n-1]);delta=np.diff(path,axis=0);assert (delta>=0).all() and (delta<=1).all() and (delta.sum(1)>0).all()
  ids=[path[path[:,1]==j,0] for j in range(n)];base=np.array([np.median(i) for i in ids]);new=phase(base)
  np.testing.assert_allclose(base,o['phase_old'],atol=0,rtol=0);np.testing.assert_allclose(new,o['phase_new'],atol=1e-12,rtol=0)
  oldt=np.abs(np.interp(base,np.arange(len(ix)),ix)-np.arange(n));newt=np.abs(np.interp(new,np.arange(len(ix)),ix)-np.arange(n));np.testing.assert_allclose(oldt,o['old_error']);np.testing.assert_allclose(newt,o['new_error'])
  bo=o['sequence_old_ok'];co=o['sequence_new_ok'];assert not(bo&~co).any();assert np.isfinite(o['sequence_old_delta']).all(-1).sum()==bo.sum();assert np.isfinite(o['sequence_new_delta']).all(-1).sum()==co.sum()
  assert np.isfinite(o['point_delta']).sum()==o['point_ok'].sum();assert not(o['point_ok']&o['missing_original']).any();assert not(o['sequence_quality']&~bo).any();assert not(o['sequence_new_quality']&~co).any()
  src=o['phase_source'];assert np.isin(src,[0,1,2]).all();assert np.array_equal(o['sequence_new_delta'][src==2],o['sequence_old_delta'][src==2],equal_nan=True)
  if kind.startswith('whole'):
   h=0 if kind.endswith('L') else 1
   assert not o['point_ok'][:,h,:17].any() and not co[:,h,:17].any() and not co[:,h,21:].any()
   extra['whole_hand_missing_controls']+=1
  if cid in subset and kind in ['identity','slow15','pace','gapL']:
   qq,qc,qix=S.transformed(z,kind);rr,rc,_=S.transformed(z,'identity');np.testing.assert_array_equal(ix,qix)
   cost,e=S.D.frame_cost_matrix(qq,rr,one_sided_mode='flat',hand_participation=False);np.testing.assert_allclose(cost[tuple(path.T)].sum(),dp_total(cost),atol=1e-9)
   fst=np.array([p.min() for p in ids]);lst=np.array([p.max() for p in ids]);rd,rv=independent_descriptor(rc['values'],rc['valid'],np.arange(n),np.arange(n),np.arange(n,dtype=float),o['domain'])
   od,ov=independent_descriptor(qc['values'],qc['valid'],fst,lst,base,o['domain']);nd,nv=independent_descriptor(qc['values'],qc['valid'],np.minimum(fst,np.floor(new).astype(int)),np.maximum(lst,np.ceil(new).astype(int)),new,o['domain'])
   source=np.where(nv,1,np.where(ov,2,0));assert np.array_equal(source,src);np.testing.assert_array_equal(ov&rv,bo);np.testing.assert_array_equal((nv|ov)&rv,co)
   a=np.where(bo[...,None],abs(od-rd),np.nan).astype(np.float32);b=np.where(co[...,None],abs(np.where(nv[...,None],nd,od)-rd),np.nan).astype(np.float32)
   np.testing.assert_allclose(a,o['sequence_old_delta'],equal_nan=True,rtol=1e-6,atol=1e-6);np.testing.assert_allclose(b,o['sequence_new_delta'],equal_nan=True,rtol=1e-6,atol=1e-6)
   independent+=1
  t=totals['normal' if kind in S.NORMAL else 'omission' if kind.startswith('skip') else 'missing' if kind in S.MISSING else 'identity'];t['cases']+=1;t['time_old_sum']+=float(oldt[core].sum());t['time_new_sum']+=float(newt[core].sum());t['core_frames']+=int(core.sum());t['sequence_old']+=int(bo.sum());t['sequence_new']+=int(co.sum());t['lost_support']+=int((bo&~co).sum());t['fallback']+=int((src==2).sum());t['quality_old']+=int(o['sequence_quality'].sum());t['quality_new']+=int(o['sequence_new_quality'].sum())
  common=bo&co
  for g,(a,b) in S.GROUPS.items():
   m=common[:,:,a:b];old=o['sequence_old_delta'][:,:,a:b][m];newd=o['sequence_new_delta'][:,:,a:b][m];gg=t['groups'][g];gg['cells']+=len(old);gg['old_sum']+=float(old.astype(float).sum());gg['new_sum']+=float(newd.astype(float).sum())
  if kind in S.NORMAL:
   change=float((newt[core]-oldt[core]).mean());reg.append(dict(clip_id=cid,word_code=row['word_code'],kind=kind,delta_frames=change,old=float(oldt[core].mean()),new=float(newt[core].mean())))
  if kind.startswith('skip'):
   removed=np.setdiff1d(np.arange(n),ix.astype(int));assert len(removed)>0
   region=np.arange(max(2,int(removed[0])-2),min(n-2,int(removed[-1])+3));m=common[region];old=o['sequence_old_delta'][region];newd=o['sequence_new_delta'][region]
   info={}
   for g,(a,b) in S.GROUPS.items():
    mm=m[:,:,a:b];aa=old[:,:,a:b][mm];bb=newd[:,:,a:b][mm];info[g]=dict(cells=len(aa),old_max=float(aa.max()) if len(aa) else None,new_max=float(bb.max()) if len(bb) else None,old_mean=float(aa.mean()) if len(aa) else None,new_mean=float(bb.mean()) if len(bb) else None,positive_to_zero=int(((np.max(aa,-1)>1e-6)&(np.max(bb,-1)<=1e-6)).sum()) if len(aa) else 0)
   omissions.append(dict(clip_id=cid,kind=kind,deleted_frames=len(removed),groups=info,point_positive_cells=int((np.nan_to_num(o['point_delta'][region],nan=0)>1e-6).sum()),whole_action_error=None))
 for cid,kinds in seen.items():assert kinds==allk,cid
 assert len(seen)==300 and independent==12 and extra['whole_hand_missing_controls']==600
 for t in totals.values():
  t['time_old_mean_frames']=t['time_old_sum']/t['core_frames'];t['time_new_mean_frames']=t['time_new_sum']/t['core_frames']
  for g in t['groups'].values():
   g['old_mean']=g['old_sum']/(g['cells']*5) if g['cells'] else None;g['new_mean']=g['new_sum']/(g['cells']*5) if g['cells'] else None
 losses=sum(g['positive_to_zero'] for a in omissions for g in a['groups'].values());normal=totals['normal']
 gate=dict(domain_and_input_preserved=True,existing_sequence_support_preserved=all(t['lost_support']==0 for t in totals.values()),normal_timing_lower=normal['time_new_sum']<normal['time_old_sum'],normal_each_group_lower=all(g['new_sum']<g['old_sum'] for g in normal['groups'].values()),no_deleted_window_positive_to_zero=losses==0)
 gate['passed']=all(gate.values());S.check(f['pins']);S.check(f['input_pins']);S.check(f['protected'])
 S.write(S.U/'VERIFIED.json',dict(at=S.now(),status='PASS_SAVED_ARITHMETIC_NOT_SEMANTIC_TRUTH',counts=dict(cases=4500,clips=300,words=100,independent_path_descriptor_checks=independent,**extra),totals=dict(totals),normal_regressions=sorted(reg,key=lambda a:a['delta_frames'],reverse=True),omissions=omissions,deleted_window_positive_to_zero_cells=losses,gate=gate,
  limits='Independent arithmetic within same Codex implementation; no independent anatomical labels. Core is fixed audit grid, not required-hands annotation. No video correctness or recognition accuracy claim. Synthetic transforms, developed DEV only.',pins={str(S.U/n):S.sha(S.U/n) for n in ['FROZEN.json','RESULT.json','verify.py']},production_changed=False))
 print('VERIFIED4500',gate,'normalTime',normal['time_old_mean_frames'],normal['time_new_mean_frames'],'omissionLostCells',losses,flush=True)
if __name__=='__main__':
 try:main()
 except BaseException:
  import traceback
  p=S.U/'VERIFY_FIRST_ERROR.txt'
  if not p.exists():p.write_text(traceback.format_exc(),encoding='utf-8')
  raise
