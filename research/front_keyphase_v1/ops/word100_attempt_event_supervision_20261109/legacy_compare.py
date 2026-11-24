"""Fixed external-to-this-unit regression inputs. Never select/tune on outcomes."""
from unit import *
assemble=prior.load_module('_attempt_evaluation',U/'evaluate.py').assemble
def main():
 assert not (U/'LEGACY_COMPARE.json').exists();torch.set_num_threads(4)
 names=['final_normal_only_42','final_attempt_supervised_42'];entries=[r for r in read(P/'EVAL_INPUTS.json')['rows'] if r['old_decision']];assert len(entries)==93
 write(U/'LEGACY_FROZEN.json',dict(models=names,inputs_sha256=sha(P/'EVAL_INPUTS.json'),code_sha256=sha(U/'legacy_compare.py'),scope='All93 previously fixed public controls, includes prior broad-main direction interventions. Additional regression, not fresh test. Weights, rules and thresholds unchanged.',no_selection=True))
 norm=read(U/'NORMALIZATION.json');results={}
 for name in names:
  net=AttemptHead();net.load_state_dict(torch.load(U/'runs'/name/'model.pt',weights_only=True,map_location='cpu'));net.eval();cases=[]
  with torch.inference_mode():
   for e in entries:
    assert sha(e['path'])==e['sha256'];z=np.load(e['path']);g=prior.geometry(z['raw_x']);v=g[:,22:33]>.5;vv=g[:,55:]>.5;g=(g-np.array(norm['mean']))/np.array(norm['std']);g[:,:22]*=np.repeat(v,2,1);g[:,33:55]*=np.repeat(vv,2,1)
    h=torch.tensor(z['h'][None]);p=torch.tensor(z['p'][None]);gg=torch.tensor(g[None],dtype=torch.float32);w=torch.tensor([e['word']]);valid=torch.ones((1,len(g)),dtype=torch.bool)
    ph,d0,pr=net(h,p,gg,w,valid,torch.zeros_like(p));_,d1,_=net(h,p,gg,w,valid,torch.ones_like(p));phase=ph[0].softmax(-1).numpy()
    r=dict(e,p=z['p'],raw=z['raw_x'],timestamp_us=z['timestamp_us'],frame_id=z['frame_id']);out=assemble(r,phase,d0[0].sigmoid().numpy(),d1[0].sigmoid().numpy(),pr[0].sigmoid().numpy())
    target=U/'legacy_decisions'/name/f"{e['clip_id']}_{e['kind']}.json";write(target,out)
    old=read(e['old_decision'])['after']
    cases.append(dict(clip_id=e['clip_id'],kind=e['kind'],word=e['word_id'],before=[c['display_status'] for c in old['cards']],after=[c['display_status'] for c in out['cards']],statuses=[c['status'] for c in out['cards']],required=[c.get('required',True) for c in out['cards']],phase_presence=[all(q['span'] is not None for q in c['phase_evidence']) for c in out['cards']],decision=str(target),sha256=sha(target)))
  summary={}
  for kind in sorted({r['kind'] for r in cases}):
   rr=[r for r in cases if r['kind']==kind];summary[kind]=dict(clips=len(rr),old_difference=sum('difference' in r['before'] for r in rr),new_difference=sum('difference' in r['after'] for r in rr),uncertain=sum('uncertain' in r['after'] for r in rr),all_required_similar=sum(all(s=='similar' for s,req in zip(r['after'],r['required']) if req) for r in rr),all_required_phase_present=sum(all(s for s,req in zip(r['phase_presence'],r['required']) if req) for r in rr))
  write(U/'scores'/f'{name}_legacy_cases.json',cases);results[name]=summary;event('legacy_evaluated',model=name,summary=summary)
 write(U/'LEGACY_COMPARE.json',dict(models=results,no_training_or_tuning=True,all93_included=True))
if __name__=='__main__':main()
