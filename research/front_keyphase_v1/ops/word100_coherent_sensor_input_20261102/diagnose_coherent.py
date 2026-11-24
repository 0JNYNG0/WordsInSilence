"""New fixed-model predictions for a changed input contract, no old rescore."""
from pathlib import Path
import sys,time
K=Path(__file__).resolve().parent;O=K.parent;J=O/'word100_sequence_supervision_audit_20261102'
sys.path.insert(0,str(J));from sequence_model import *
from evaluate_witness import observe
import evaluate_episode as prior

def main():
    assert not (K/'DIAGNOSTIC.json').exists();torch.set_num_threads(2)
    plan=read(K/'EXECUTION_FREEZE.json');meta=read(K/'INPUT_MANIFEST.json')['rows'];stats=read(D/'NORMALIZATION.json')
    tar={r['clip_id']:{p['frame']:p['action_family'] for p in r['points'] if p['mask']} for r in read(D/'DEV_ACTION_TARGETS.json')['rows']}
    observer=process.state.StateNet(False);observer.load_state_dict(torch.load(T/'runs/current_42/model.pt',map_location='cpu',weights_only=True));observer.eval()
    rows=[]
    for m in meta:
        assert m['person'] in [11,12,13] and sha(m['input'])==m['input_sha256']
        a=dict(np.load(m['input']));n=len(a['latent']);ids=np.asarray(a['frame_id']);av=a['image_available']
        z,mask=process.old.geometric(a['x'],a['mask'],av)
        x=torch.from_numpy(a['latent'][None]);l=torch.tensor([n]);word=torch.tensor([process.WORDS.index(m['word_id'])])
        with torch.inference_mode():current=observer(x,l,word).softmax(-1)[0].numpy()
        rows.append(dict(m,key=m['clip_id']+'_coherent_native',kind='coherent_native',x=a['latent'],z=z,m=mask,rgb=a['rgb'],
            current=current,available=av,mapping=ids,times=a['timestamp_us']/1e6,y=process.old.target(m['clip_id'],ids),physical=process.state.labels_for(m['clip_id'],ids)))
    episode.attach_targets(rows,tar)
    for r in rows:r['role']=np.where(r['episode']<0,-1,np.where(r['episode']<8,1,2))
    gg=prior.graphs();results=[]
    for tag in plan['conditions']:
        model=WitnessNet();path=J/'runs'/tag/'model.pt';model.load_state_dict(torch.load(path,map_location='cpu',weights_only=True));model.eval();pts=[];raw=[];roles=[];sequences=[]
        for r in rows:
            x,im,l,y,a=batch_j([r],stats)
            with torch.inference_mode():act,role,seq=model(x,im,l);pr=act.softmax(-1)[0].numpy();rp=role.softmax(-1)[0].numpy()
            sq,effective,ro=observe(pr,rp,r,gg[r['word_id']]);sq.update(key=r['key'],word=r['word_id']);sequences.append(sq)
            for f in np.flatnonzero(r['episode']>=0):
                base=dict(key=r['key'],clip_id=r['clip_id'],frame=int(f),y=int(r['episode'][f]))
                pts.append(dict(base,pred=int(effective[f].argmax())));raw.append(dict(base,pred=int(pr[f].argmax())))
                roles.append(dict(key=r['key'],frame=int(f),y=int(r['role'][f]),pred=int(ro[f].argmax())))
            dest=K/'predictions'/tag/f"{r['clip_id']}.npz";dest.parent.mkdir(parents=True,exist_ok=True);assert not dest.exists()
            np.savez_compressed(dest,probability=effective,raw_action_probability=pr,role_probability=ro,activity_probability=rp,current_probability=r['current'],image_available=r['available'],frame_id=r['mapping'])
        result=dict(condition=tag,points=pts,raw_points=raw,role_points=roles,sequences=sequences,
            major=process.state.metrics([p for p in pts if p['y']<8]),raw_major=process.state.metrics([p for p in raw if p['y']<8]),
            rest_known=sum(p['y']==2 for p in roles),rest_correct=sum(p['y']==2 and p['pred']==2 for p in roles),
            supported_order=sum(s['status']==prior.SUPPORTED for s in sequences),conflict_cases=sum(bool(s['rest_action_conflict_frames']) for s in sequences))
        exclusive(K/'evaluations'/f'{tag}.json',result);results.append({k:v for k,v in result.items() if k not in ['points','raw_points','role_points','sequences']})
        print(tag,results[-1],flush=True)
    for p,h in plan['pins'].items():assert sha(p)==h
    exclusive(K/'DIAGNOSTIC.json',dict(at=clock(),status='NEW_INPUT_DIAGNOSTIC_COMPLETE_NOT_PRODUCT_ACCEPTANCE',results=results,new_comparisons=24,
        primary=plan['primary'],new_training=0,old_inference_repeated=False,old_scores_revised=False,
        scope='Whole frontend contract contrast, not isolated crop effect. Same observed public videos, new inputs; no final-test claim or operating model adoption.',whole_goal_complete=False))
if __name__=='__main__':main()
