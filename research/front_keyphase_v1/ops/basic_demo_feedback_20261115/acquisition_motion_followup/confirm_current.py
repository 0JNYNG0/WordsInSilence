from motion_model import *
from check_motion_bridge import bridge,Features,RelationObserver,adapter
import sys,time
sys.path.insert(0,str(U));from shape_consensus import ShapeConsensus

def main():
    V=D/'confirmation';assert not(V/'RESULTS.json').exists();torch.set_num_threads(2)
    records=read(V/'EXTRACTED.json')['rows']
    # Source RGB reviewed before opening this unit's predictions.
    truth=dict(gratitude=dict(person=18,arrivals=[[57,60],[65,67]],first_only_end=60,
        motion_pairs=[dict(start=54,end=58,label=1),dict(start=58,end=62,label=2),dict(start=62,end=66,label=1),dict(start=66,end=72,label=2),dict(start=72,end=74,label=0)]),
        apple=[dict(person=14,index_phase_anchors=[33,37,41,45],later_interaction_anchors=[57,65,73,81],right_long_fingers=[1,0,0,0]),
               dict(person=15,index_phase_anchors=[48,52,56,60,64],later_interaction_anchors=[72,80,88,96],right_long_fingers=[1,0,0,0])],
        review='Native sequential RGB crops; Codex visual review, no independent KSL expert. Coarse long-finger extension only, thumb and depth contact not labelled.',
        full_feedback_correctness=None,apple_later_state='Visible two-hand interaction only; no invented per-finger truth for curved/occluded fingers.',
        provenance=[dict(clip_id=r['clip_id'],sha256=r['sha256'])for r in records])
    write(V/'SOURCE_TRUTH.json',truth)
    factory=Features();base=RelationObserver();motion=Observer('mlp32');shape=ShapeConsensus();rr=[];motion_rows=[];finger_rows=[]
    pilot=U.parents[3]/'research/sign_model_pilot_v1/src';sys.path.insert(0,str(pilot))
    from pilot.model_masked_v1 import MaskedSignTCN
    ac=torch.load(U/'apple_classifier.pt',map_location='cpu',weights_only=False)
    assert sha(ac['base_checkpoint'])==ac['base_sha256'];ck=torch.load(ac['base_checkpoint'],map_location='cpu',weights_only=True)
    net=MaskedSignTCN(**ck['model_config']);net.load_state_dict(ck['state_dict']);net.eval()
    extra=nn.Linear(ac['state_dict']['weight'].shape[1],1);extra.load_state_dict(ac['state_dict']);extra.eval()
    for r in records:
        ch=dict(np.load(r['channel_path']));raw=dict(np.load(r['raw_path']));n=len(ch['frame_id'])
        if r['word']=='WORD1290':
            for kind in ['normal','first_only','constant','missing','fast_timestamps','slow_timestamps','scale_translation']:
                d={k:v.copy()for k,v in raw.items()}
                if kind=='first_only':
                    for k in ['xy_px','score','timestamp_us','frame_id']:d[k]=d[k][:61]
                if kind=='constant':d['xy_px']=np.repeat(d['xy_px'][58:59],n,axis=0)
                if kind=='missing':d['xy_px'][:,91:]=np.nan
                if kind=='fast_timestamps':d['timestamp_us']=(d['timestamp_us']*.65).astype(np.int64)
                if kind=='slow_timestamps':d['timestamp_us']=(d['timestamp_us']*1.5).astype(np.int64)
                if kind=='scale_translation':d['xy_px']=d['xy_px']*.75+np.array([90,40],np.float32)
                z=adapter.convert(d['xy_px'],d['score'],d['timestamp_us'],d['frame_id'],int(d['source_width']),int(d['source_height']))
                t=time.perf_counter();f,q,_=factory(z);s,old=base(f,q,z['timestamp_us']);m=motion.sequence(z);new=bridge(s,m);elapsed=time.perf_counter()-t
                expected=2 if kind in ['normal','fast_timestamps','slow_timestamps','scale_translation']else 0 if kind in ['constant','missing'] else None
                good=len(new)<=1 if expected is None else len(new)==expected
                native_match=all(a<=e['end']<=b for e,(a,b)in zip(new,truth['gratitude']['arrivals'])) if kind=='normal' and len(new)==2 else None
                rr.append(dict(clip_id=r['clip_id'],kind=kind,old_events=old,new_events=new,passed=good,arrival_native_match=native_match,seconds=elapsed))
            for a in truth['gratitude']['motion_pairs']:
                motion_rows.append(dict(**a,pred=motion.predict(feature(ch,a['start'],a['end']))))
        else:
            # New normal recordings only. Word name recognition is separate from correction.
            h=[];hook=net.head.register_forward_pre_hook(lambda module,args:h.append(args[0].detach()))
            with torch.inference_mode():
                x=torch.tensor(ch['tcn280'][None]);old=net(x,torch.ones((1,n)));logits=torch.cat([old,extra(h.pop())],dim=1)[0].numpy()
            hook.remove();rank=np.argsort(-logits);top=rank[:3].tolist()
            rr.append(dict(clip_id=r['clip_id'],kind='apple_recognition',top3=top,apple_rank=int(np.where(rank==100)[0][0])+1,whole_word_correct=None))
            so=shape.sequence(ch['tcn280'],'R');a=next(x for x in truth['apple']if x['person']==r['person'])
            for i in a['index_phase_anchors']:
                for finger,y,p in zip(['index','middle','ring','little'],a['right_long_fingers'],so['fingers'][i]):
                    finger_rows.append(dict(clip_id=r['clip_id'],frame=i,finger=finger,label=y,pred=int(p)))
    summary=dict(gratitude_checks=sum(r.get('passed',False)for r in rr),gratitude_total=sum('passed'in r for r in rr),
        motion_anchor=metrics(motion_rows,'pred'),apple_top1=sum(r.get('apple_rank')==1 for r in rr),apple_videos=2,
        apple_finger_anchors=metrics(finger_rows,'pred'))
    write(V/'RESULTS.json',dict(summary=summary,rows=rr,motion_anchors=motion_rows,finger_anchors=finger_rows,
        final_whole_word_success=False,web_admitted=False,training=False,
        limit='One gratitude and two apple recordings held from current model selection. Not general user feedback accuracy; no natural learner error recordings.',
        timing='Motion observation after extracted RTMW; no camera/RTMW/DTW/UI timing.'))
    print(json.dumps(summary,ensure_ascii=False))
if __name__=='__main__':main()
