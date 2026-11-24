"""One-shot public diagnostic of feedback, not new training or learner-error accuracy."""
from pathlib import Path
import sys,json,hashlib
import numpy as np,torch
U=Path(__file__).resolve().parent;O=U.parent;D=O/'word100_action_feedback_demo_20261108';J=O/'word100_rtmw_joint_learning_20261108';G=O/'word100_observation_guard_20261108'
sys.path.insert(0,str(D))
from consumer import FeedbackConsumer,quarantine,A,normalize,assess,dtw
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def aligned_fist(donor,target):
    # Use a previously RGB-reviewed right closed hand; preserve target wrist and
    # match the in-plane palm direction/size. A coordinate intervention, not RGB truth.
    src=donor-donor[0];dst=target-target[0];anchors=[5,9,13,17]
    u,_,vt=np.linalg.svd(src[anchors].T@dst[anchors]);rot=u@vt
    if np.linalg.det(rot)<0:u[:,-1]*=-1;rot=u@vt
    scale=np.median(np.linalg.norm(dst[anchors],axis=1))/max(1e-6,np.median(np.linalg.norm(src[anchors],axis=1)))
    return src@rot*scale+target[0]

def main():
    assert not (U/'AUDIT.json').exists();torch.set_num_threads(4)
    plan=read(G/'INPUT_PLAN.json');rows=[r for r in plan['rows'] if r['word_id']=='WORD1133' and r['person'] in [11,12,13]]
    donor_rec=read(J/'records/NIA_SL_WORD0739_REAL01_F.json');donor=np.load(donor_rec['raw_path'])['xy_px'][68,112:133].copy()
    conditions=['normal_current_guard','right_closed_hand_main','main_reversed','closed_hand_and_reversed']
    pins={str(p):sha(p) for p in [D/'consumer.py',D/'evidence.py',D/'CARDS.json',D/'runs/42/model.pt',J/'feedback_viewfix/feedback_runs/pose_transfer_42/model.pt',G/'guard.py',G/'geometry.py']}
    write(U/'FROZEN.json',dict(clips=[r['clip_id'] for r in rows],conditions=conditions,pins=pins,
       donor=dict(clip='NIA_SL_WORD0739_REAL01_F',frame=68,reason='Previously RGB-reviewed right four-finger closure'),
       interventions='Modify only copied public coordinate arrays: right-hand template in pre-existing broad main interval; reverse same main interval coordinates/scores. Keep all timestamps and outside frames. No user arrays read.',
       endpoint='Report per-card support versus differences/unknown, never retune thresholds or pick models.',
       limitation='Artificial coordinate controls, not independent natural learner-error truth. Original field scores are carried, not validated visibility confidence.',training=False,private_used=False))
    c=FeedbackConsumer();out=[];wi=c.base.f['feedback']['words'].index('WORD1133')
    for r in rows:
        z=np.load(r['cache'] or G/'crops'/(r['clip_id']+'.npz'));ts=z['timestamp_us'];fid=z['frame_id'];lo,hi=r['intervals'][0];ids=np.flatnonzero((ts>=lo*1e6)&(ts<=hi*1e6))
        for kind in conditions:
            xy=z['xy_px'].copy();score=z['score'].copy()
            if kind in ['right_closed_hand_main','closed_hand_and_reversed']:
                for i in ids:xy[i,112:133]=aligned_fist(donor,xy[i,112:133])
            if kind in ['main_reversed','closed_hand_and_reversed']:
                xy[ids]=xy[ids[::-1]];score[ids]=score[ids[::-1]]
            guarded=quarantine(xy,score,ts,640,480);a=A.convert(guarded['xy_px'],score,ts,fid,640,480);x=a['tcn280'];norm=normalize(x,c.base.stats)
            with torch.inference_mode():
                b=c.base.feedback(torch.from_numpy(norm[None]),torch.tensor([len(x)]),torch.tensor([wi]));h=b['latent'];p=b['attributes'].sigmoid();valid=torch.ones((1,len(x)),dtype=torch.bool)
                ph,d0=c.head(h,p,torch.tensor([wi]),valid,torch.zeros_like(p));_,d1=c.head(h,p,torch.tensor([wi]),valid,torch.ones_like(p))
            phase=ph[0].softmax(-1).numpy();d0=d0[0].sigmoid().numpy();d1=d1[0].sigmoid().numpy();attrs=p[0].numpy()
            e=assess('WORD1133',phase,d0,d1,x,ts,fid)
            # Match the deployed consumer's conservative overlap downgrade.
            for card in e['cards']:
                span=card.get('query_indices');sides=[0 if s=='L' else 1 for s in card['hands']]
                if span and guarded['hand_quarantined'][span[0]:span[1]+1,sides].any():card['status']='unknown';card['observation_quarantine_overlap']=True
            # DTW is computed only AFTER all evidence decisions, exactly as runtime.
            before=json.dumps(e,sort_keys=True);proto=c.base.feedback.prototype[wi,:c.base.stats['prototype_lengths'][wi]].detach().numpy();path=dtw(h[0].numpy(),proto)['path'];assert json.dumps(e,sort_keys=True)==before
            rs=[]
            for card in e['cards']:
                span=card.get('query_indices');ix=np.arange(span[0],span[1]+1) if span else ids
                rs.append(dict(id=card['id'],title=card['title'],status=card['status'],span=span,differences=card['differences'],open_right_mean=float(attrs[ix,1].mean()),closed_right_mean=float(attrs[ix,3].mean()),difference_if_right_open_max=float(d1[ix,1].max()),threshold=.65,explicit_feature_indices=list(card['features']),quarantine_overlap=card.get('observation_quarantine_overlap',False)))
            pred=U/(r['clip_id']+'_'+kind+'.npz');np.savez_compressed(pred,phase=phase,different_if0=d0,different_if1=d1,attributes=attrs,path=path,timestamp_us=ts,frame_id=fid)
            out.append(dict(clip_id=r['clip_id'],kind=kind,main_indices=[int(ids[0]),int(ids[-1])],cards=rs,dtw_changed_decision=False,prediction=str(pred),sha256=sha(pred),guarded_hand_frames=guarded['hand_quarantined'].sum(0).tolist(),overall_percent=None,synthetic=kind!='normal_current_guard'))
    summary={k:{cid:sum(next(c for c in r['cards'] if c['id']==cid)['status']=='supported' for r in out if r['kind']==k) for cid in ['brush_arm','reach_wrist']} for k in conditions}
    write(U/'AUDIT.json',dict(rows=out,supported_counts=summary,denominator_per_condition=3,actual_user_video_reviewed=False,model_unchanged=True,training=False,real_error_accuracy=False));print(json.dumps(summary,ensure_ascii=False))
if __name__=='__main__':main()
