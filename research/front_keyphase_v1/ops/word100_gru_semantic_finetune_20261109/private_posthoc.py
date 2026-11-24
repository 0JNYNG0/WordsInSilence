"""Explicit user-authorized latest recording review AFTER the public decision.
Never imported by training, and never changes weights, thresholds or selection.
"""
from semantic_model import *
import importlib.util

def module(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def main():
 decision_sha=sha(U/'PUBLIC_DECISION.json');verify=read(U/'PUBLIC_VERIFICATION.json');assert verify['status']=='PASS'
 assert verify['public_decision_sha256']==decision_sha
 result=read(U/'PRIVATE_LATEST_RESULT.json');assert result['source']=='user_inference_only' and result['word']=='WORD2523'
 source=Path('C:/Users/local-user/AppData/Local/Temp/wis_action_demo_k8b6c0m8');rawpath=source/'rtmw/observations.npz';raw=np.load(rawpath,allow_pickle=False)
 assert np.array_equal(raw['timestamp_us'],result['timestamp_us']);assert len(raw['xy_px'])==result['recorded_frames']==143
 annotation=dict(scope='latest saved private test only; after public candidate decision; not training or calibration',
  word='WORD2523',frames=[34,44,54,64,74,84,94,104],stages={44:10,54:10,64:11,74:12,84:12},
  direction=[dict(a=54,b=74,axis=3,target=2),dict(a=74,b=84,axis=3,target=1)],order=[dict(a=54,b=84,target=1)],
  shape='Edge-on fingers: no fine handshape truth claimed. Visible gathering and separation only.',reviewer='same Codex RGB review before candidate inference',
  public_decision_sha256=decision_sha,source_sha256=sha(rawpath),video_sha256=sha(source/'recorded.mp4'))
 write(U/'private_posthoc/ANNOTATION.json',annotation)
 sys.path.insert(0,str(O/'word100_observation_guard_20261108'))
 A=module('_posthoc_adapter',O/'word100_rtmw_shared_input_20261108/adapter.py');G=module('_posthoc_guard',O/'word100_observation_guard_20261108/guard.py')
 observed=G.quarantine(raw['xy_px'],raw['score'],raw['timestamp_us'],int(raw['source_width']),int(raw['source_height']))
 a=A.convert(observed['xy_px'],raw['score'],raw['timestamp_us'],raw['frame_id'],int(raw['source_width']),int(raw['source_height']));x=a['tcn280']
 stats=read(V/'FEEDBACK_NORMALIZATION.json');mask=np.concatenate([np.repeat(x[:,186:237]>0,2,axis=1),np.repeat(x[:,237:279]>0,2,axis=1)],1)
 z=x.copy();z[:,:186]=np.where(mask,(x[:,:186]-np.asarray(stats['mean']))/np.asarray(stats['std']),0)
 torch.set_num_threads(4);outputs={}
 for name in ['original','finetune_42']:
  model=SemanticNet()
  if name=='original':model.base.load_state_dict(torch.load(BASE,map_location='cpu',weights_only=True))
  else:model.load_state_dict(torch.load(U/'runs/finetune_42/model.pt',map_location='cpu',weights_only=True))
  model.eval()
  with torch.inference_mode():
   o=model.base(torch.from_numpy(z[None]),torch.tensor([len(z)]),torch.tensor([3]));sp=o['stage'][0].softmax(-1).numpy();h=o['latent'][0]
   anchors=[dict(frame=f,target=g,pred=int(sp[f].argmax()),confidence=float(sp[f].max())) for f,g in annotation['stages'].items()];directions=[];orders=[]
   if name!='original':
    for p in annotation['direction']:
     d,_=model.pair(h,p['a'],p['b'],3);pr=d[p['axis']].softmax(-1).numpy();directions.append(dict(p,pred=int(pr.argmax()),confidence=float(pr.max()),eligible=pair_available(z,p['a'],p['b'],p['axis'])))
    for p in annotation['order']:
     _,order=model.pair(h,p['a'],p['b'],3);prob=float(order.sigmoid());orders.append(dict(p,pred=int(prob>=.5),probability=prob,eligible=order_available(z,p['a'],p['b'],3)))
   else:
    for p in annotation['order']:orders.append(dict(p,pred=int(sp[p['a']].argmax()<sp[p['b']].argmax())))
   outputs[name]=dict(stage_anchors=anchors,correct=sum(q['target']==q['pred'] for q in anchors),n=len(anchors),directions=directions,orders=orders)
  np.savez_compressed(U/'private_posthoc'/f'{name}.npz',stage=sp,attributes=o['attributes'][0].sigmoid().numpy(),frame_id=raw['frame_id'],timestamp_us=raw['timestamp_us'])
 assert sha(U/'PUBLIC_DECISION.json')==decision_sha
 report=dict(status='PRIVATE_POSTHOC_REVIEW_COMPLETE_NO_TUNING',record='latest retained Egypt test, 143 frames, original timestamps matched',
  original_display=dict(tcn_top3=result['tcn_top3'],stop_to_result_seconds=result['stop_to_result_seconds'],cards=[{k:c.get(k) for k in ['id','status','query_frames','message']} for c in result['cards']]),
  comparison=outputs,web_changed=False,learning_used=False,candidate_selection_used=False,public_decision_unchanged=True,
  limitation='One non-independent user recording, five visible stage anchors; not user accuracy, not validation of exact triangular finger configuration or educational correctness.',
  models={str(BASE):sha(BASE),str(U/'runs/finetune_42/model.pt'):sha(U/'runs/finetune_42/model.pt')})
 write(U/'PRIVATE_POSTHOC.json',report);event('private_posthoc_complete',comparison=outputs)
if __name__=='__main__':main()
