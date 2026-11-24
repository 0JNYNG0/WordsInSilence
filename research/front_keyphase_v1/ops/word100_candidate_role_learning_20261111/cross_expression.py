from role_core import *
import traceback

def main():
 assert not (U/'CROSS_EXPRESSION_COMPLETE.json').exists();torch.set_num_threads(4);torch.set_num_interop_threads(1)
 sys.modules['motion_core']=mc;me=load('_role_movement_io',M/'evaluate.py');sys.modules['joint_core']=jc;assignmod=load('_role_assign_for_test',J/'proposal_adapter.py')
 anns=read(A/'EVALUATION_LABELS.json')['rows'];raws=[r for r in read(A/'EXTRACTION_COMPLETE.json')['rows'] if r['person']==14];assert len(raws)==6
 write(U/'CROSS_EXPRESSION_FROZEN.json',dict(videos=[r['clip_id'] for r in raws],variants=['normal','whole_reverse','speed075','speed125'],status='Existing previously exposed six non-focus expressions; no fresh test claim',metric='Complete/partial/multiple/background quality on saved localizer proposals with known reviewed targets; unknown candidates excluded and counted',no_default_change=True))
 frozen=frozen_model();base=prior.Network('coupled',read(old.P/'NORMALIZATION.json'));base.load_state_dict(torch.load(old.P/'runs/coupled_42/model.pt',weights_only=True,map_location='cpu'));base.eval();oldadapter=RoleAdapter();nets={}
 for seed in [42,43,44]:
  n=RoleAdapter(seed);n.load_state_dict(torch.load(U/'runs'/str(seed)/'model.pt',weights_only=True,map_location='cpu'));n.eval();nets[seed]=n
 result=[];records=[]
 for r in raws:
  assert sha(r['raw'])==r['raw_sha256'];assert sha(r['source'])==r['source_sha256']
  for kind in ['normal','whole_reverse','speed075','speed125']:
   x,ts=me.convert(r['raw'],kind=='whole_reverse');ts=(np.asarray(ts)/({'speed075':.75,'speed125':1.25}.get(kind,1))).astype(np.int64);fh,_=lc.be.ev.frozen_forward(base,x,-1);ob=features(frozen,x,ts,fh);an=copy.deepcopy([a for a in anns if a['clip_id']==r['clip_id']]);n=len(x)
   if kind=='whole_reverse':
    for a in an:
     a['events']=[dict(e,onset=[n-1-z for z in e['offset'][::-1]],offset=[n-1-z for z in e['onset'][::-1]]) for e in a['events']]
     for k in ['holds','background']:a[k]=[[n-1-b,n-1-c] for c,b in a[k]]
   # Fixed primary localizer candidates are shared across all newly trained
   # adapters, so seed effects are not confounded with different intervals.
   es=read(A/'boundary_integrity_repair/observations/42'/f"{r['clip_id']}_{kind}.json")['events'];ss=[(0,e['a'],e['b'],e['axis']*2+(e['moving']=='L')) for e in es];sf=segment_features(frozen,ob,ss);targets=[assignmod.assign(a,b,k,an)[0] for _,a,b,k in ss]
   with torch.inference_mode():op=oldadapter.quality(torch.tensor(sf)).softmax(-1).numpy();new={s:net.quality(torch.tensor(sf)).softmax(-1).numpy() for s,net in nets.items()}
   for i,(e,t) in enumerate(zip(es,targets)):
    for seed in nets:result.append(dict(clip_id=r['clip_id'],kind=kind,seed=seed,event=e,target=t,before=int(op[i].argmax()),after=int(new[seed][i].argmax()),before_probability=op[i].tolist(),after_probability=new[seed][i].tolist(),known=t>=0))
   records.append(dict(clip_id=r['clip_id'],kind=kind,proposals=len(es),known=sum(t>=0 for t in targets),unknown=sum(t<0 for t in targets)))
 write(U/'CROSS_EXPRESSION_SCORES.json',result);write(U/'CROSS_EXPRESSION_COMPLETE.json',dict(videos=6,conditions=24,rows=records,private_data=0,old_TCN_DTW_runs=0));event('cross_expression_complete',conditions=24)
if __name__=='__main__':
 try:main()
 except BaseException:(U/f'CROSS_FAILURE_{time.time_ns()}.txt').write_text(traceback.format_exc(),'utf-8');raise
