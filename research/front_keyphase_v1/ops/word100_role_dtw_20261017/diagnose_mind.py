"""Fixed diagnostic cross; never fitting, replacing scores or selecting a candidate."""
import sys,json,hashlib
from pathlib import Path
import numpy as np
import torch
U=Path(__file__).resolve().parent;B=U.parents[1]
sys.path[:0]=[str(B/'src'),str(B.parents[1]/'research/sign_model_pilot_v1/src')]
from wis_lab.service import word100 as W,word100_candidate as C,neural
from wis_lab.core.observation import Observation
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text('utf-8'))
assert not (U/'MIND_DIAGNOSIS.json').exists(),'Use saved diagnostics'
torch.set_num_threads(2)
models={'M0':W._load(),'C':C._load()};out=[];pins={}
for tid in ['tri_8dfa3b4dd027','tri_071e79278225']:
    base=B/'runs/ses_b7aef2518a30/trials'/tid;saved=read(base.with_suffix('.result.json'))
    mp=base.with_suffix('.npz');cp=base.with_name(tid+'.candidate_c.npz')
    m=Observation.load(str(mp));c=Observation.load(str(cp))
    xm=neural._channels(m)[0]['x'];xc=C.build_candidate_channels(c)[0]['x'];assert xm.shape==xc.shape
    cross=[]
    for ik,x in [('M0',xm),('C',xc)]:
        for mk,model in models.items():
            xx=torch.from_numpy(x[None].astype(np.float32));pad=torch.ones(xx.shape[:2])
            with torch.inference_mode():v=model(xx,pad)[0].cpu().numpy()
            rank=W.rank_logits(v,100);target=next(z for z in rank if z['word_id']=='WORD0500')
            if ik==mk:
                old=saved['practice100'] if mk=='C' else saved['comparison']['baseline']
                assert np.allclose(v,old['logits'],atol=3e-5,rtol=1e-5)
                assert target['rank']==old['requested_rank_all100']
            cross.append({'input':ik,'weights':mk,'target_rank':target['rank'],'target_score':target['model_score'],'top1':rank[0],'diagnostic_not_new_candidate':True})
    obs_stats={}
    for name,o in [('M0',m),('C',c)]:
        active=np.flatnonzero(o.usable[:,33:].any(1));sl=slice(active[0],active[-1]+1)
        side=[]
        for hand,offset,pw,other in [('L',33,15,16),('R',54,16,15)]:
            ok=o.usable[:,offset]&o.usable[:,pw]&o.usable[:,other]
            same=np.linalg.norm(o.xy_px[ok,offset]-o.xy_px[ok,pw],axis=1);opp=np.linalg.norm(o.xy_px[ok,offset]-o.xy_px[ok,other],axis=1)
            side.append({'side':hand,'comparable_wrist_frames':int(ok.sum()),'closer_to_opposite_pose_wrist_frames':int((opp<same).sum()),'meaning':'geometric diagnostic only; crossed arms and pose errors can change this'})
        obs_stats[name]={'active_first':int(active[0]),'active_last':int(active[-1]),'quality':o.quality_summary(),'wrist_side_check':side}
    out.append({'trial_id':tid,'cross':cross,'channel_shape':list(xm.shape),'channel_changed_values':int(np.count_nonzero(xm!=xc)),'channel_total_values':int(xm.size),'observations':obs_stats})
    for p in [mp,cp,base.with_suffix('.result.json')]:pins[str(p)]=sha(p)
pub=read(B/'ops/word100_body_boundary_20261017/training_C_20261013_RESULT.json')
dev_summary={v:metrics['by_class']['11'] for v,metrics in pub['metrics'].items()}
report={'status':'COMPLETE_DIAGNOSTIC_ONLY','cases':out,'frozen_public_DEV_WORD0500':dev_summary,'source_sha256':pins,'new_diagnostic_forward_calls':8,'training_steps':0,'tuning':0,'user_results_replaced':False,'limitations':'Two observed takes cannot establish a single cause or a linguistic correctness judgment; cross changes only frozen input-array/weight pairing, not the performer.'}
(U/'MIND_DIAGNOSIS.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),'utf-8')
print(json.dumps({'crosses':[{'trial':x['trial_id'],'rows':x['cross'],'changed':x['channel_changed_values'],'total':x['channel_total_values']} for x in out],'public_dev':dev_summary},ensure_ascii=False,indent=2))
