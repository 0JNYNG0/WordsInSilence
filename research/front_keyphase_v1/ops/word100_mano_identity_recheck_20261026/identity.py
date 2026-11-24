"""Offline bilateral image continuity evidence; never invents hidden fingers."""
import cv2
import numpy as np
PARAMS=dict(width=640,window_s=.5,arm_gate=.25,arm_margin=.10,
            win=21,levels=3,fb_px=1.5,lk_error=20.,match_shoulder=.12,
            ambiguity_ratio=.5,min_tips=4)
TIPS=[4,8,12,16,20]
def current_arm(hand,pose,good,scale):
    if scale<=1 or not np.isfinite(hand[0]).all():return -1
    d=[float(np.linalg.norm(hand[0]-pose[15+s])/scale) if good[[11+s,13+s,15+s]].all() else np.inf for s in (0,1)]
    side=int(np.argmin(d))
    return side if d[side]<=PARAMS['arm_gate'] and d[1-side]-d[side]>=PARAMS['arm_margin'] else -1

def flow(gray,start,end,xy,ratio):
    p=np.asarray(xy,np.float32)*ratio;valid=np.isfinite(p).all(-1);p[~valid]=0
    steps=[];direction=1 if end>start else -1
    args=dict(winSize=(PARAMS['win'],)*2,maxLevel=PARAMS['levels'],criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,30,.01))
    for t in range(start,end,direction):
        q,st,e=cv2.calcOpticalFlowPyrLK(gray[t],gray[t+direction],p[:,None],None,**args)
        if q is None:return None
        back,bst,_=cv2.calcOpticalFlowPyrLK(gray[t+direction],gray[t],q,None,**args)
        if back is None:return None
        q=q[:,0];back=back[:,0];e=e[:,0];st=st[:,0];bst=bst[:,0]
        fb=np.linalg.norm(back-p,axis=-1)
        good=(st==1)&(bst==1)&np.isfinite(q).all(-1)&np.isfinite(e)&(e<=PARAMS['lk_error'])&(fb<=PARAMS['fb_px'])
        good &= (q[:,0]>=0)&(q[:,0]<gray[t].shape[1])&(q[:,1]>=0)&(q[:,1]<gray[t].shape[0])
        valid &= good
        steps.append(dict(frame_from=t,frame_to=t+direction,previous=p.copy(),forward=q.copy(),backward=back.copy(),status=st.copy(),back_status=bst.copy(),error=e.copy(),valid=valid.copy()))
        p=np.where(np.isfinite(q),q,p)
    return dict(anchor=start,target=end,xy=p/ratio,valid=valid,steps=steps)

def build_anchors(xy,valid,pose,pv,duplicate):
    n=len(xy);anchors=np.zeros((n,2),bool)
    for t in range(n):
        scale=float(np.linalg.norm(pose[t,11]-pose[t,12])) if pv[t,[11,12]].all() else 0
        if scale<=1:continue
        arms=[current_arm(xy[t,s],pose[t],pv[t],scale) if valid[t,s].all() else -1 for s in (0,1)]
        duplicated=bool(valid[t].all() and duplicate(xy[t,0],xy[t,1],valid[t,0],valid[t,1],scale)['hold_new'])
        if duplicated:continue
        for s in (0,1):anchors[t,s]=arms[s]==s and arms[1-s]!=s
    return anchors

def get_tracks(t,gray,xy,anchors,time_s,cache):
    out={}
    for s in (0,1):
        before=[i for i in range(t) if anchors[i,s] and 0<time_s[t]-time_s[i]<=PARAMS['window_s']+1e-8]
        after=[i for i in range(t+1,len(xy)) if anchors[i,s] and 0<time_s[i]-time_s[t]<=PARAMS['window_s']+1e-8]
        out[s]=[]
        for i in ([max(before)] if before else [])+([min(after)] if after else []):
            key=(i,t,s)
            if key not in cache:cache[key]=flow(gray,i,t,xy[i,s],PARAMS['width']/1920.)
            q=cache[key]
            if q is not None:out[s].append(q)
    return out

def assess(hand,pose,pv,scale,tracks):
    arm=current_arm(hand,pose,pv,scale);scores=[]
    for s in (0,1):
        details=[]
        for tr in tracks[s]:
            ids=[i for i in TIPS if tr['valid'][i] and np.isfinite(hand[i]).all()]
            ds=[float(np.linalg.norm(tr['xy'][i]-hand[i])/scale) for i in ids] if scale>1 else []
            score=float(np.median(ds)) if len(ds)>=PARAMS['min_tips'] else None
            wrist=float(np.linalg.norm(tr['xy'][0]-hand[0])/scale) if scale>1 and tr['valid'][0] and np.isfinite(hand[0]).all() else None
            details.append(dict(anchor=tr['anchor'],target=tr['target'],tips=ids,tip_distances=ds,score=score,wrist=wrist))
        both=len(details)==2 and details[0]['anchor']<details[0]['target']<details[1]['anchor']
        qualified=both and all(x['score'] is not None and x['score']<=PARAMS['match_shoulder'] and x['wrist'] is not None and x['wrist']<=PARAMS['match_shoulder'] for x in details)
        cost=max(x['score'] for x in details) if both and all(x['score'] is not None for x in details) else None
        scores.append(dict(side=s,two_sided=both,qualified=bool(qualified),cost=cost,details=details))
    supported=[]
    for s in (0,1):
        q=scores[s];other=scores[1-s]
        unique=q['qualified'] and (other['cost'] is None or q['cost']<PARAMS['ambiguity_ratio']*other['cost'])
        if unique and arm==s:supported.append(s)
    side=supported[0] if len(supported)==1 else -1
    return dict(current_arm=arm,temporal=scores,supported_side=side,
                status='current_arm_and_two_sided_image_support' if side>=0 else 'identity_not_confirmed',
                finger_shape_ground_truth=False)

def choose(candidates,assessments,requested_side,existing_side):
    eligible=[i for i,a in enumerate(assessments) if a['supported_side']==requested_side]
    # Only a uniquely supported CURRENT candidate can replace a duplicate slot.
    # No averaging, no word ID, no carrying the old left/right label as evidence.
    if existing_side==requested_side:return dict(chosen=-1,reason='same_side_source_quality_unresolved')
    if len(eligible)!=1:return dict(chosen=-1,reason='not_unique_current_image_candidate')
    return dict(chosen=eligible[0],reason='unique_current_arm_and_bidirectional_image_candidate')
