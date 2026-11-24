from pathlib import Path
import sys,json,hashlib,time
import numpy as np
import cv2
from identity_v2 import PARAMS,build_anchors,get_tracks,assess,choose
U=Path(__file__).resolve().parent;O=U.parent;OLD=O/'word100_mano_threeway_20261025';G=O/'word100_mano_duplicate_guard_20261025'
sys.path.insert(0,str(G));from candidate_v2 import inspect_pair
def read(p):return json.loads(Path(p).read_text('utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def video_gray(path,ids):
    cap=cv2.VideoCapture(str(path));assert cap.isOpened(),path
    fps=cap.get(cv2.CAP_PROP_FPS);w=cap.get(cv2.CAP_PROP_FRAME_WIDTH);assert w==1920,w
    allframes={};wanted=set(map(int,ids));j=0
    while j<=int(max(ids)):
        ok,img=cap.read();assert ok,(path,j)
        if j in wanted:allframes[j]=cv2.cvtColor(cv2.resize(img,(640,360)),cv2.COLOR_BGR2GRAY)
        j+=1
    cap.release();return [allframes[int(i)] for i in ids],fps
def save_tracks(root,cache):
    root.mkdir();index=[]
    for (i,t,s),tr in sorted(cache.items()):
        if tr is None:index.append({'anchor':i,'target':t,'side':s,'failed':True});continue
        dest=root/f'{i:04d}_{t:04d}_{s}.npz'
        arrays={k:np.stack([q[k] for q in tr['steps']]) for k in ['previous','forward','backward','status','back_status','error','valid']}
        np.savez_compressed(dest,xy=tr['xy'],final_valid=tr['valid'],**arrays)
        index.append({'anchor':i,'target':t,'side':s,'path':str(dest),'sha256':sha(dest),'frames':[[q['frame_from'],q['frame_to']] for q in tr['steps']]})
    return index
def scale(p,v):return float(np.linalg.norm(p[11]-p[12])) if v[[11,12]].all() else 0
def main():
    f=read(U/'FROZEN.json');assert sha(U/'identity.py')==f['candidate_sha256'];assert sha(U/'identity_v2.py')==read(U/'FROZEN_V2.json')['candidate_sha256'];assert not (U/'RESULT_V2.json').exists()
    out=U/'outputs_v2';out.mkdir();td=U/'tracks_v2';td.mkdir();result=[];diagnostics=[];indices={};start=time.perf_counter()
    prior={ (x['clip_index'],x['frame'],x['side']):x for x in read(G/'RESULT_V2.json')['rows']}
    for k,m in enumerate(f['manifest']):
        assert sha(m['prepared'])==m['prepared_sha256'] and sha(m['rgb_path'])==m['rgb_sha256']
        z0=np.load(m['prepared']);z={q:z0[q] for q in z0.files}; g0=np.load(G/'outputs_v2'/(m['clip_id']+'.npz'));g={q:g0[q] for q in g0.files}
        gray,fps=video_gray(m['rgb_path'],z['frame_id']); suppressed=np.zeros(z['full_valid'].shape[:2],bool)
        for cr in read(G/'REQUESTED_DIAGNOSTIC_V2.json'):
            if cr['clip_index']==k and cr['conflict']:
                suppressed[cr['frame'],cr['side']]=True
                for pair in cr['pairs']:
                    if pair['hold_new']:suppressed[cr['frame'],pair['existing_side']]=True
        anchors=build_anchors(z['full_xy'],z['full_valid'],z['pose'],z['pose_valid'],inspect_pair,suppressed);cache={}
        rx=g['G_xy'].copy();rv=g['G_valid'].copy();ri=g['G_inferred'].copy();re=g['G_evidence'].copy();eligible=np.ones((len(rx),2),bool);resolved=np.zeros((len(rx),2),bool)
        for t,s in np.argwhere(z['requested']):
            t=int(t);s=int(s);raw=np.load(OLD/f'raw_predictions/{k:02d}_{t:04d}_{s}.npz')['xy_px']
            tracks=get_tracks(t,gray,z['full_xy'],anchors,z['timestamp_us']/1e6,cache);sc=scale(z['pose'][t],z['pose_valid'][t])
            a=assess(raw,z['pose'][t],z['pose_valid'][t],sc,tracks)
            rec=dict(clip_index=k,clip=m['clip_id'],frame=t,side=s,raw_assessment=a,duplicate_conflict=False)
            p=prior.get((k,t,s))
            if p and p['conflict']:
                rec['duplicate_conflict']=True
                es=[x['existing_side'] for x in p['pairs'] if x['hold_new']]
                eligible[t,s]=False
                for e in es:eligible[t,e]=False
                if len(es)==1:
                    e=es[0];old=z['full_xy'][t,e];oa=assess(old,z['pose'][t],z['pose_valid'][t],sc,tracks);decision=choose([old,raw],[oa,a],s,e)
                    rec.update(existing_side=e,existing_assessment=oa,decision=decision)
                    if decision['chosen']>=0:
                        selected=[old,raw][decision['chosen']]
                        rx[t,e]=np.nan;rv[t,e]=False;ri[t,e]=False;re[t,e]=False
                        rx[t,s]=selected;box=z['roi_box'];valid=np.isfinite(selected).all(-1)&(selected[:,0]>=box[0])&(selected[:,0]<box[2])&(selected[:,1]>=box[1])&(selected[:,1]<box[3])
                        rv[t,s]=valid;ri[t,s]=valid;re[t,s]=False;eligible[t,s]=True;resolved[t,s]=True
                else:rec['decision']={'chosen':-1,'reason':'multiple_existing_hands_conflict'}
                result.append(rec)
            diagnostics.append(rec)
        np.savez_compressed(out/(m['clip_id']+'.npz'),R_xy=rx,R_valid=rv,R_inferred=ri,R_observed=re,identity_eligible=eligible,identity_supported=resolved,
                            raw_G_xy=g['G_xy'],raw_G_valid=g['G_valid'],raw_G_evidence=g['G_evidence'],frame_id=z['frame_id'],timestamp_us=z['timestamp_us'],roi_box=z['roi_box'],anchors=anchors,anchor_suppressed=suppressed)
        indices[m['clip_id']]=save_tracks(td/m['clip_id'],cache)
        print(m['clip_id'],'identity_supported',int(resolved.any(-1).sum()),'tracks',len(cache),flush=True)
    broad=[]
    # Both broad flagged controls, fixed by earlier source review; no new subset selection.
    for oldflag in read(G/'RESULT.json')['broad_proxy']['flags']:
        cid=oldflag['clip_id'];t=oldflag['frame_index'];m=next(x for x in f['controls'] if x['clip_id']==cid)
        assert sha(m['path'])==m['sha256'];video=G/'source_clips'/(cid+'.mp4');assert sha(video)==m['source']['sha256']
        zz=np.load(m['path']);xy=zz['xy_px'];vv=zz['usable'];ids=zz['frame_id'];hx=xy[:,33:75].reshape(-1,2,21,2);hv=vv[:,33:75].reshape(-1,2,21);pose=xy[:,:33];pv=vv[:,:33]
        gray,fps=video_gray(video,ids);anchors=build_anchors(hx,hv,pose,pv,inspect_pair);cache={};tracks=get_tracks(t,gray,hx,anchors,ids/fps,cache)
        assessments=[assess(hx[t,s],pose[t],pv[t],scale(pose[t],pv[t]),tracks) for s in (0,1)]
        pair=inspect_pair(hx[t,0],hx[t,1],hv[t,0],hv[t,1],scale(pose[t],pv[t]))
        broad.append(dict(clip=cid,frame=t,duplicate=pair['hold_new'],assessments=assessments,mutation=False,reason='both_existing_candidates_diagnostic_only_no_arbitrary_winner'))
        indices[cid]=save_tracks(td/cid,cache)
    report={'status':'REPLAY_COMPLETE_REVIEW_REQUIRED','primary_frames':812,'public_clips':7,'missing_slots':57,'saved_HaMeR_requests':len(diagnostics),'duplicate_frames':len(result),'identity_supported_frames':sum(x.get('decision',{}).get('chosen',-1)>=0 for x in result),'rows':result,'broad_diagnostics':broad,'method':'Original full-image RGB flow, nearest clean arm-supported anchors before/after<=0.5s, current arm plus unique tip/wrist agreement. Offline future context, not realtime. Proposed identity not hidden-finger truth.','elapsed_seconds':time.perf_counter()-start,'training':0,'new_neural_inference':0,'user_data_access':0,'production_change':False}
    write(U/'TRACK_INDEX_V2.json',indices);write(U/'DIAGNOSTICS_V2.json',diagnostics);write(U/'RESULT_V2.json',report)
    print(json.dumps({k:v for k,v in report.items() if k not in ['rows','broad_diagnostics']},ensure_ascii=False))
if __name__=='__main__':main()
