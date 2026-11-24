"""Compare masked observations and input channels, never model predictions."""
from pathlib import Path
from collections import Counter
import sys,json,time
import numpy as np
sys.dont_write_bytecode=True
sys.stdout.reconfigure(encoding='utf-8')
U=Path(__file__).resolve().parent;sys.path.insert(0,str(U))
from run_pair import read,write,pins_ok,event,Observation,sha256_file
from roi_adapter import apply_roi,build_roi_channels
PALM=np.array([0,5,9,13,17])

def flags(ob,side):
    v=ob.usable[:,33+21*side:54+21*side]
    return {'any':v.any(1),'palm5':v[:,PALM].all(1),'all21':v.all(1)}

def longest_false(v):
    longest=run=0
    for a in v:
        run=0 if a else run+1;longest=max(longest,run)
    return longest

def metrics(ob):
    out={'frames':ob.n_frames,'usable_body_points':int(ob.usable[:,:33].sum()),'shoulders_both_frames':int(ob.usable[:,[11,12]].all(1).sum())}
    for side,label in enumerate(('left','right')):
        f=flags(ob,side);out[label]={k:int(v.sum()) for k,v in f.items()}
        out[label]['usable_points']=int(ob.usable[:,33+21*side:54+21*side].sum())
        out[label]['longest_no_hand_frames']=longest_false(f['any'])
    return out

def pair_detail(full,sensor):
    assert np.array_equal(full.timestamp_us,sensor.timestamp_us) and np.array_equal(full.frame_id,sensor.frame_id)
    out={};events=[]
    for side,label in enumerate(('left','right')):
        a,b=flags(full,side),flags(sensor,side);stats={}
        for key in a:
            stats[key]={'both':int((a[key]&b[key]).sum()),'full_only':int((a[key]&~b[key]).sum()),'sensor_only':int((~a[key]&b[key]).sum()),'neither':int((~a[key]&~b[key]).sum())}
        out[label]=stats
        # Event source is whole-frame reference, not hand ground-truth or measured physical entry.
        v=a['palm5'];missing=0
        for f,on in enumerate(v):
            if on and missing>=3:
                hi=min(f+6,len(v));win=b['palm5'][f:hi];found=np.flatnonzero(win)
                events.append({'side':label,'frame':f,'timestamp_us':int(full.timestamp_us[f]),'preceding_reference_missing_frames':missing,
                    'window_frames':hi-f,'full_palm5_frames':int(v[f:hi].sum()),'sensor_palm5_frames':int(win.sum()),
                    'sensor_has_palm5_at_reference_entry':bool(win[0]),'sensor_first_palm5_offset_within_window':int(found[0]) if len(found) else None})
            missing=0 if on else missing+1
    out['reference_entry_events']=events
    return out

def main():
    assert not (U/'COMPARISON.json').exists(),'Reuse completed result'
    plan=read(U/'PLAN.json');pins_ok(plan['pins']);capture=read(U/'CAPTURE_RESULT.json')
    for r in capture['rows']:pins_ok(r['outputs'])
    write(U/'COMPARISON_CONTEXT.json',{'plan_sha256':sha256_file(U/'PLAN.json'),'capture_sha256':sha256_file(U/'CAPTURE_RESULT.json'),
        'code':{str(p):sha256_file(p) for p in (Path(__file__),U/'roi_adapter.py')},'scope':'fixed4conditions20clips; no classifier'})
    rows=[];aggregate={k:Counter() for k in ('legacy','bottom')};entries={k:[] for k in ('legacy','bottom')}
    for it in plan['items']:
        cid=it['clip_id'];native=Observation.load(it['cached']['native']['observation'])
        crop=Observation.load(it['cached']['crop']['observation']);bottom=Observation.load(str(U/'sensor_crop'/cid/'candidate.npz'))
        variants={
            'full_then_legacy_mask':(native,it['box'],(0,0)),
            'legacy_crop_then_mask':(crop,it['box'],tuple(it['box'][:2])),
            'full_then_bottom_mask':(native,it['bottom_box'],(0,0)),
            'bottom_crop_then_mask':(bottom,it['bottom_box'],(0,0)),
        }
        masked={};rm={};outputpins={}
        for name,(source,box,origin) in variants.items():
            ob=apply_roi(source,box,source_origin=origin);ch=build_roi_channels(ob)
            x=ch['x'];assert x.shape==(source.n_frames,280) and np.isfinite(x).all()
            d=U/'outputs'/cid/name;d.mkdir(parents=True)
            ob.save(str(d/'model_input.npz'))
            np.savez_compressed(d/'channels.npz',x=x,frame_id=ob.frame_id,timestamp_us=ob.timestamp_us)
            rm[name]=metrics(ob);masked[name]=ob
            outputpins.update({str(p):sha256_file(p) for p in d.iterdir()})
        pairs={}
        for key in ('legacy','bottom'):
            detail=pair_detail(masked['full_then_'+key+'_mask'],masked[key+'_crop_then_mask'])
            pairs[key]=detail
            for side in ('left','right'):
                for measure in ('any','palm5','all21'):
                    for flag,value in detail[side][measure].items():aggregate[key][measure+'_'+flag]+=value
            entries[key].extend(dict(clip_id=cid,label=it['label'],**e) for e in detail['reference_entry_events'])
        rows.append({'clip_id':cid,'label':it['label'],'frames':native.n_frames,'metrics':rm,'pairs':pairs,'outputs':outputpins})
        event(stage='compared',clip=cid)
    pins_ok(plan['pins'])
    write(U/'COMPARISON.json',{'status':'COMPUTED_PENDING_INDEPENDENT_AND_VISUAL_REVIEW','clips':20,'conditions':80,
        'unique_video_frames':sum(r['frames'] for r in rows),'condition_frames':4*sum(r['frames'] for r in rows),
        'aggregate':{k:dict(v) for k,v in aggregate.items()},'entry_events':entries,'rows':rows,
        'interpretation':'Frame-side evidence coverage, not coordinate ground truth or recognition accuracy. Fullframe context can be privileged versus a physical narrow sensor. Each ROI pair matched, legacy versus bottom ROI comparisons not single-variable detector comparisons.',
        'training':0,'classifier_inference':0,'production_changes':0})

if __name__=='__main__':main()
