from probe import *
import av,hashlib,copy

def states(q):return {c['id']:c['status'] for c in q['lesson_acceptance']['criteria']}
def unknowns(q):return sum(v=='unknown' for v in states(q).values())
def retained(a,b):return all(v=='unknown' or states(b).get(k)==v for k,v in states(a).items()) and a['tcn_top3']==b['tcn_top3']

def main():
    torch.set_num_threads(2);cv2.setNumThreads(2)
    records=read(N/'INPUTS.json')['rows'];checks=[];normal_rows=[];natural=[]
    for r in records:
        if 'video'not in r:r['video']=r['path']
    pins=read(P/'web/ADMISSION.json')['pins']
    sources=[r for r in records if r['person']in[11,12,16,17]]
    write(D/'VERIFICATION_PLAN.json',dict(sources=[r['id']for r in sources],natural_sources=[r['id']for r in sources if r['person']==11],frozen_candidate_sha256=hashlib.sha256((D/'selective_retry.py').read_bytes()).hexdigest(),thresholds_changed_after_results=False,private_data=False))
    net=FeedbackConsumer();engine=Engine()
    try:
        for r in records:
            require_front(r);raw=dict(np.load(r['raw']));initial=net.raw(raw,r['word'])
            q,_=retry(raw,initial,VideoFrames(r['video'],raw),engine,net.raw,r['word'])
            saved=read(P/'web'/('normal_'+r['id']+'.json'))
            row=dict(id=r['id'],passed=states(q)==states(saved) and q['tcn_top3']==saved['tcn_top3'] and q['selective_retry']['pose_calls']==0,retry=q['selective_retry'])
            normal_rows.append(row)
            write(D/'NORMAL_PRESERVATION.json',normal_rows)
        print('NORMAL',sum(x['passed']for x in normal_rows),len(normal_rows),flush=True)
        for r in sources:
            z=dict(np.load(r['raw']));normal=net.raw(z,r['word'])
            guide=next(c for c in normal['cards']if c['id']==('right_shape'if r['word']=='WORD1290'else'index_shape'))
            a,b=guide['query_spans'][0];mid=(a+b)//2;lo=mid-2;hi=mid+2
            for kind in ['missing_one_finger','missing_hand','blank_hand_pixels','long_gap','outside_action']:
                raw={k:v.copy()for k,v in z.items()};joint=slice(125,129)if kind!='missing_hand'else slice(112,133)
                u,v=(lo,hi)if kind not in['long_gap','outside_action']else(max(0,mid-6),min(len(z['xy_px'])-1,mid+6))if kind=='long_gap'else(3,7)
                raw['xy_px'][u:v+1,joint]=np.nan;raw['score'][u:v+1,joint]=0
                initial=net.raw(raw,r['word'])
                class Pixels(VideoFrames):
                    def read(self,ids,deadline):
                        frames=super().read(ids,deadline)
                        if kind=='blank_hand_pixels':
                            for i,im in frames.items():
                                if lo<=i<=hi:
                                    hand=z['xy_px'][i,112:133];low=np.maximum(np.floor(hand.min(0)-15),0).astype(int);high=np.minimum(np.ceil(hand.max(0)+16),[im.shape[1],im.shape[0]]).astype(int)
                                    im[low[1]:high[1],low[0]:high[0]]=0
                        return frames
                out,fixed=retry(raw,initial,Pixels(r['video'],raw),engine,net.raw,r['word'])
                no_wrong=not any(v=='difference' and states(initial)[k]!='difference' for k,v in states(out).items())
                protected=retained(initial,out)
                if kind in ['blank_hand_pixels','long_gap']:passed=protected and not out['selective_retry']['accepted']
                elif kind=='outside_action':passed=protected and out['selective_retry']['pose_calls']==0
                else:passed=protected and no_wrong
                # Original arrays remain unchanged. Any replacement must occur in
                # the one declared missing-hand span; no other body's points move.
                if out['selective_retry']['accepted']:
                    delta=~np.isclose(raw['xy_px'],fixed['xy_px'],equal_nan=True).all(-1)
                    allowed=np.zeros(delta.shape,bool);s=91 if out['selective_retry']['hand']=='L' else 112;aa,bb=out['selective_retry']['span'];allowed[aa:bb+1,s:s+21]=~np.isfinite(raw['xy_px'][aa:bb+1,s:s+21]).all(-1)
                    passed=passed and not np.any(delta&~allowed)
                row=dict(id=r['id'],role=r['role'],kind=kind,before=states(initial),after=states(out),unknown_before=unknowns(initial),unknown_after=unknowns(out),passed=bool(passed),retry=out['selective_retry'])
                checks.append(row);write(D/'checks'/f"{r['id']}_{kind}.json",row)
                print('CASE',r['id'],kind,row['unknown_before'],row['unknown_after'],row['passed'],out['selective_retry']['reason'],round(out['selective_retry']['seconds'],3),flush=True)
        # Existing erroneous joint-pattern controls are preserved even if another
        # hand becomes unavailable. They are observation edits, not real learner videos.
        r=next(r for r in records if r['word']=='WORD1290'and r['person']==13);z=dict(np.load(r['raw']));normal=net.raw(z,r['word']);ev=normal['lesson_acceptance']['events'];a=ev[0]['start'];b=ev[-1]['end']
        donor=next(r for r in read(O/'demo_handshape_learning_20261113/INPUTS.json')['rows']if r['clip_id']=='NIA_SL_WORD2633_REAL01_F');require_front(donor);dr=dict(np.load(donor['raw']));i=donor['anchors'][len(donor['anchors'])//2];hand=dr['xy_px'][i,112:133];hand=(hand-hand[0])/np.linalg.norm(hand[9]-hand[0])
        for kind in ['known_error','known_error_other_hand_missing','partial','out_of_order']:
            raw={k:v.copy()for k,v in z.items()}
            if kind.startswith('known_error'):
                size=np.linalg.norm(raw['xy_px'][:,121]-raw['xy_px'][:,112],axis=1);raw['xy_px'][a:b+1,112:133]=raw['xy_px'][a:b+1,112:113]+hand[None]*size[a:b+1,None,None]
                if kind.endswith('missing'):raw['xy_px'][a+1:a+6,104:108]=np.nan
            elif kind=='partial':
                cut=ev[-1]['arrival_index']-1
                for k in ['xy_px','score','timestamp_us','frame_id']:raw[k]=raw[k][:cut]
            else:
                raw['xy_px'][a:b+1]=raw['xy_px'][a:b+1][::-1];raw['score'][a:b+1]=raw['score'][a:b+1][::-1]
            initial=net.raw(raw,r['word']);out,_=retry(raw,initial,VideoFrames(r['video'],raw),engine,net.raw,r['word'])
            passed=retained(initial,out) and (not out['overall_pass'] if kind!='out_of_order' else states(out)==states(initial))
            row=dict(id=r['id'],kind=kind,before=states(initial),after=states(out),passed=passed,retry=out['selective_retry'],synthetic_observation_control=True,semantic_error_gold=kind!='out_of_order')
            checks.append(row);print('ERROR',kind,passed,flush=True)
        # Natural detection at webcam-sized resolution: no dropped observations.
        # Use two unchanged public front videos, full-frame letterbox stress and
        # the existing upper-body fixture. These are not newly unseen people.
        for r in sources:
            if r['person']!=11:continue
            for mode in ['letterbox','upper_body']:
                def transform(im):
                    h,w=im.shape[:2]
                    if mode=='upper_body':return cv2.resize(im[int(h*.08):int(h*.75),int(w*.25):int(w*.75)],(640,480))
                    scale=min(640/w,480/h);nw,nh=round(w*scale),round(h*scale);out=np.zeros((480,640,3),np.uint8);out[(480-nh)//2:(480-nh)//2+nh,(640-nw)//2:(640-nw)//2+nw]=cv2.resize(im,(nw,nh));return out
                coordinates=[];scores=[];times=[]
                with av.open(r['video'])as container:
                    for fr in container.decode(video=0):
                        ans=engine.infer(transform(fr.to_ndarray(format='bgr24')));coordinates.append(ans['xy']);scores.append(ans['score']);times.append(round(float(fr.pts*fr.time_base)*1e6))
                raw=dict(xy_px=np.array(coordinates),score=np.array(scores),timestamp_us=np.array(times),frame_id=np.arange(len(times)),source_width=np.array(640),source_height=np.array(480))
                initial=net.raw(raw,r['word']);out,_=retry(raw,initial,VideoFrames(r['video'],raw,transform),engine,net.raw,r['word'])
                row=dict(id=r['id'],mode=mode,before=states(initial),after=states(out),unknown_before=unknowns(initial),unknown_after=unknowns(out),retained=retained(initial,out),retry=out['selective_retry'],source_pixels_reframed=True,synthetic_dropout=False)
                natural.append(row);print('NATURAL',r['id'],mode,row['unknown_before'],row['unknown_after'],flush=True)
                write(D/'natural'/f"{r['id']}_{mode}.json",row)
        hashes=all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==sha for p,sha in pins.items())
        recovered=[x for x in checks if x['kind']in['missing_one_finger','missing_hand']and x['unknown_after']<x['unknown_before']]
        passed=all(x['passed']for x in normal_rows+checks)and all(x['retained']for x in natural)and bool(recovered)and hashes
        write(D/'PUBLIC_VALIDATION.json',dict(passed=passed,normal=normal_rows,controls=checks,natural=natural,recovered_cases=len(recovered),prior_pins_unchanged=hashes,prior_pins=len(pins),devices=engine.devices,private_data=False,training_runs=0,scope='Known public F sources, controlled observation losses and 4 natural re-extractions; not physical webcam or general accuracy'))
        print('VALIDATION',passed,'recovered',len(recovered),flush=True)
    finally:close(net)
if __name__=='__main__':main()
