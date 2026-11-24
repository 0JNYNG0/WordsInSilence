from pathlib import Path
import sys,copy
import numpy as np,torch
U=Path(__file__).resolve().parent;P=U.parent/'word100_clause_finger_diagnostics_20261114'
sys.path.insert(0,str(P));from finger_core import FingerHead,local_sequence
from finger_model import features,windows,SharedFinger
from observation_quality import finger_observability,classification_reasons
FINGERS=['index','middle','ring','little'];KO=['검지','중지','약지','새끼손가락']

def classify(p,available):
    q=np.where(p>=.85,1,np.where(p<=.15,0,-1));q[~available]=-1;return q

def merge_observers(expanded,chain,old):
    a=np.where((expanded==expanded[0]).all(0),expanded[0],-1)
    b=np.where((chain==chain[0]).all(0),chain[0],-1)
    prev=np.where((old==old[0]).all(0),old[0],-1)
    out=np.where((a==b)&(a>=0),a,-1)
    conflict=(out>=0)&(prev>=0)&(out!=prev);out[conflict]=-1
    return out,conflict

class FingerObserver:
    def __init__(self):
        self.models={};self.ready=False
        for kind,path in [('expanded_whole_hand',U/'expanded_whole_hand'),('shared_finger_chain',U/'shared_finger_chain'),('old_v2',P/'finger_models_v2')]:
            mm=[]
            for seed in [42,43,44]:
                ck=torch.load(path/f'{seed}.pt',map_location='cpu',weights_only=False);m=SharedFinger()if kind=='shared_finger_chain'else FingerHead();m.load_state_dict(ck['state_dict']);m.eval();mm.append((m,np.array(ck['mean'],np.float32),np.array(ck['std'],np.float32)))
            self.models[kind]=mm
        self.ready=True
    def infer(self,x,hand):
        f,known=features(x,hand);usable,geometry_reason=finger_observability(x,hand);known=known&usable
        fl,kl=local_sequence(x,hand);n=len(x);xx=windows(f,np.arange(n));xo=windows(np.c_[fl,np.zeros((n,64),np.float32)],np.arange(n));preds={};probs={}
        for kind,models in self.models.items():
            ps=[];qs=[]
            for m,mu,sd in models:
                v=(xx-mu)/sd if kind=='shared_finger_chain'else(xo-mu)/sd
                if kind!='shared_finger_chain':v[:,:,63:]=0
                with torch.inference_mode():p=m(torch.tensor(v,dtype=torch.float32)).sigmoid().numpy()
                if kind!='shared_finger_chain':p=p[:,1:]
                ps.append(p);qs.append(classify(p,known))
            probs[kind]=np.array(ps);preds[kind]=np.array(qs)
        states,conflict=merge_observers(preds['expanded_whole_hand'],preds['shared_finger_chain'],preds['old_v2'])
        reasons=classification_reasons(preds,states,known,geometry_reason)
        return dict(states=states,conflicts=conflict,available=known,probabilities=probs,
                    reasons=reasons,geometry_reasons=geometry_reason)

def refine(result,observations,ts):
    out=copy.deepcopy(result);count=0
    for c in out['joint_cards']:
        c['finger_details']=[]
        if c['state']!='attributable_feature_difference':continue
        for e in c['differences']:
            if e['task']!='shape' or e.get('feature')not in [1,2,3,4] or e.get('expected')!=1:continue
            hand='R'if e['feature']in[1,3]else'L';expected=1 if e['feature']in[1,2]else 0;a,b=e['span'];obs=observations[hand]
            for j,name in enumerate(FINGERS):
                # Report observed local runs, never bridge missing or conflicting frames.
                ids=np.flatnonzero(obs['states'][a:b+1,j]==1-expected)+a
                runs=np.split(ids,np.flatnonzero(np.diff(ids)>1)+1)if len(ids)else[]
                for run in runs:
                    if len(run)<3 or ts[run[-1]]-ts[run[0]]<100000:continue
                    message=f"{'오른손'if hand=='R'else'왼손'} {KO[j]}가 {'굽혀진'if expected else'펴진'} 것으로 관측됐어요."
                    action=f"{'오른손'if hand=='R'else'왼손'} {KO[j]}를 {'펴'if expected else'굽혀'} 보세요."
                    c['finger_details'].append(dict(kind='finger',finger=name,hand=hand,expected='extended'if expected else'flexed',observed='flexed'if expected else'extended',span=[int(run[0]),int(run[-1])],message=message,action=action,coarse_evidence=e,source='compatible temporal whole-hand and shared-finger models; prior conflict guard',whole_sign_correct=None));count+=1
    out['finger_supplement']=dict(enabled=True,detail_count=count,thumb=False,role='Refines existing attributed handshape differences only; does not create error cards or claim completion')
    out['finger_supplement']['observation_guard_version']='numeric-v1-20261116'
    out['finger_supplement']['joint_correctness_verified']=False
    return out
