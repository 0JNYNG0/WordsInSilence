"""Independent recomputation without importing proposal/crop/gate implementations."""
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
import av
import numpy as np
from PIL import Image, ImageDraw

UNIT=Path(__file__).resolve().parent
BASE=UNIT.parents[1]
OUT=BASE/'runs/rgb_temporal_roi_v1'

def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def load(p):
    with np.load(p,allow_pickle=False) as z: return {k:z[k] for k in z.files}
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        while block:=f.read(1048576): h.update(block)
    return h.hexdigest()

def independent_gate(xy,mask,side):
    palms=[np.array([33,38,42,46,50]),np.array([54,59,63,67,71])]
    spans=[np.arange(33,54),np.arange(54,75)]
    other=1-side
    def palm(t,s):
        if not mask[t,spans[s]].any(): return 'absent',None
        if not mask[t,palms[s]].all() or not np.isfinite(xy[t,palms[s]]).all(): return 'bad',None
        return 'complete',xy[t,palms[s]].astype(float)
    def nearer(q,s,o):
        ds=np.mean(np.sqrt(np.sum((q-s)**2,axis=-1)))
        do=np.mean(np.sqrt(np.sum((q-o)**2,axis=-1)))
        return bool(np.isfinite(ds) and np.isfinite(do) and ds<do)
    t0,t1=palm(0,side)[1],palm(2,side)[1]
    state0,o0=palm(0,other); state1,o1=palm(2,other)
    if 'bad' in (state0,state1): return False
    for t,tar in [(0,t0),(2,t1)]:
        pw=[15+side,15+other]
        if not mask[t,pw].all() or not np.isfinite(xy[t,pw]).all(): return False
        if not nearer(tar[:1],xy[t,pw[0]][None],xy[t,pw[1]][None]): return False
    if o1 is not None and not nearer(t0,t1,o1): return False
    if o0 is not None and not nearer(t1,t0,o0): return False
    ms,mo=palm(1,other)
    if ms=='bad': return False
    if ms=='complete':
        if o0 is None or o1 is None: return False
        if not nearer(mo,o0,t0) or not nearer(mo,o1,t1): return False
    return True


def main():
    manifest=read(BASE/'ops/rgb_pose_readiness_20261010/PAIRED_INPUT_MANIFEST.json')
    by_id={r['sample_id']:r for r in manifest}
    hidden=read(OUT/'HIDDEN_EVENTS.json'); natural=read(OUT/'NATURAL_EVENTS.json')
    stored={(e['sample_id'],e['frame_id'],e['side']):e for e in hidden+natural}
    assert len(stored)==len(hidden)+len(natural)
    obs_cache={}; expected_hidden=set(); expected_natural=set(); metric_rows=[]
    for r in manifest:
        z=load(r['output_path']); p=r['source']['observation_path']
        if p not in obs_cache: obs_cache[p]=load(p)
        obs=obs_cache[p]
        ids=z['frame_id']; ts=z['timestamp_us']
        for j in range(1,len(ids)-1):
            if ids[j-1]+1!=ids[j] or ids[j]+1!=ids[j+1]: continue
            if not 0<int(ts[j+1]-ts[j-1])<=66668 or not ts[j-1]<ts[j]<ts[j+1]: continue
            for s,side in enumerate(('left','right')):
                if not z['crop_valid'][j-1,s] or not z['crop_valid'][j+1,s]: continue
                hand=slice(33+21*s,54+21*s)
                ishidden=bool(z['crop_valid'][j,s])
                if not ishidden and (obs['usable'][ids[j],hand].any() or obs['observed'][ids[j],hand].any()): continue
                key=(r['sample_id'],int(ids[j]),side)
                (expected_hidden if ishidden else expected_natural).add(key)
                event=stored[key]
                xy=obs['xy_px'][ids[j-1:j+2]].copy(); mask=obs['usable'][ids[j-1:j+2]].copy()
                xy[1,hand]=np.nan; mask[1,hand]=False
                accepted=independent_gate(xy,mask,s)
                assert accepted==event['gate']['accepted'],('gate',key)
                before=z['bbox_xyxy'][j-1,s].astype(float); after=z['bbox_xyxy'][j+1,s].astype(float)
                fraction=(int(ts[j])-int(ts[j-1]))/(int(ts[j+1])-int(ts[j-1]))
                length=math.ceil(before[2]-before[0]+fraction*((after[2]-after[0])-(before[2]-before[0])))
                cx=(before[0]+before[2])/2+fraction*((after[0]+after[2]-before[0]-before[2])/2)
                cy=(before[1]+before[3])/2+fraction*((after[1]+after[3]-before[1]-before[3])/2)
                box=[math.floor(cx-length/2),math.floor(cy-length/2)]
                box += [box[0]+length,box[1]+length]
                assert box==event['proposed_box'],('proposal',key,box,event['proposed_box'])
                if not ishidden: continue
                truth=z['bbox_xyxy'][j,s].astype(float); points=obs['xy_px'][ids[j],hand].astype(float)
                for method,rect in [('proposal',box),('previous',before)]:
                    x0,y0,x1,y1=map(float,rect); a0,b0,a1,b1=truth
                    hits=sum(x0<=x<x1 and y0<=y<y1 for x,y in points)
                    inter=max(0,min(x1,a1)-max(x0,a0))*max(0,min(y1,b1)-max(y0,b0))
                    ar=(x1-x0)*(y1-y0); at=(a1-a0)*(b1-b0)
                    values={'all21':hits==21,'points_inside':int(hits),'iou':inter/(ar+at-inter),
                            'normalized_center_error':math.hypot((x0+x1-a0-a1)/2,(y0+y1-b0-b1)/2)/(a1-a0),
                            'area_ratio':ar/at}
                    for k,v in values.items():
                        if k=='all21': assert bool(v)==event[method+'_metrics'][k],(key,method,k)
                        else: assert abs(v-event[method+'_metrics'][k])<1e-10,(key,method,k)
                    metric_rows.append((accepted,method,values))
    assert expected_hidden=={(e['sample_id'],e['frame_id'],e['side']) for e in hidden}
    assert expected_natural=={(e['sample_id'],e['frame_id'],e['side']) for e in natural}
    result=read(OUT/'GEOMETRY_RESULT.json')
    for accepted_only,name in [(False,'hidden_ungated'),(True,'hidden_accepted')]:
        for method in ('proposal','previous'):
            rows=[v for ok,m,v in metric_rows if m==method and (ok or not accepted_only)]
            assert len(rows)==result[name]['n']
            reported=result[name][method]
            assert sum(v['all21'] for v in rows)==reported['all21']
            assert abs(100*sum(v['points_inside'] for v in rows)/(21*len(rows))-reported['point_containment_percent'])<1e-9
            for k in ('iou','normalized_center_error','area_ratio'):
                assert abs(sum(v[k] for v in rows)/len(rows)-reported['mean_'+k])<1e-10
    prior=read(BASE/'ops/natural_handgap_audit_20261004/REVIEW.json')['events']
    prior_by_key={(e['sample_id'],e['frame_ids'][1],e['side']):e for e in prior}
    for e in natural:
        old=prior_by_key[(e['sample_id'],e['frame_id'],e['side'])]
        assert (e['prior_event'],e['prior_classification'])==(old['event'],old['classification'])
    records=read(OUT/'PIXEL_RECORDS.json')
    assert {(e['sample_id'],e['frame_id'],e['side']) for e in records}=={(e['sample_id'],e['frame_id'],e['side']) for e in natural if e['gate']['accepted']}
    # Re-decode every output frame and use PIL's independent outside-image padding.
    tasks=defaultdict(list)
    for e in records: tasks[e['source_path']].append(('natural',e))
    regressions=[e for e in hidden if e['previous_metrics']['all21'] and not e['proposal_metrics']['all21']]
    selected=sorted(regressions,key=lambda e:-e['proposal_metrics']['normalized_center_error'])[:5]
    for e in regressions[:5]:
        if e not in selected: selected.append(e)
    for n,e in enumerate(selected):
        e=dict(e,visual_index=n)
        tasks[by_id[e['sample_id']]['source']['raw_path']].append(('hidden',e))
    checked=0; timing_errors=[]; hidden_panels=[]
    for raw,items in sorted(tasks.items()):
        needed=defaultdict(list)
        for kind,e in items: needed[e['frame_id']].append((kind,e))
        found=set()
        with av.open(raw) as con:
            stream=con.streams.video[0]
            for ordinal,frame in enumerate(con.decode(stream)):
                if ordinal in needed:
                    rgb=frame.to_ndarray(format='rgb24'); image=Image.fromarray(rgb)
                    us=round(frame.pts*stream.time_base*1000000)
                    for kind,e in needed[ordinal]:
                        assert abs(us-e['timestamp_us'])<=1
                        timing_errors.append(abs(int(us)-e['timestamp_us']))
                        expected=image.crop(e['proposed_box']).resize((128,128),Image.Resampling.BILINEAR)
                        if kind=='natural':
                            z=load(e['path'])
                            assert sha(e['path'])==e['sha256']
                            assert hashlib.sha256(rgb.tobytes()).hexdigest()==e['raw_rgb_sha256']
                            assert np.array_equal(z['rgb'],np.asarray(expected))
                            white=Image.new('L',image.size,255).crop(e['proposed_box']).resize((128,128),Image.Resampling.NEAREST)
                            assert np.array_equal(z['pixel_valid'],np.asarray(white)>0)
                            assert z['region_estimated'] and not z['hand_landmarks_observed']
                            assert z['frame_id']==ordinal and z['timestamp_us']==e['timestamp_us']
                            assert np.array_equal(z['bbox_xyxy'],e['proposed_box'])
                            checked+=1
                        else:
                            panel=Image.new('RGB',(900,390),'white'); d=ImageDraw.Draw(panel)
                            d.text((8,5),e['sample_id'],fill='black')
                            d.text((8,23),f"{e['side']} frame={ordinal} gate={e['gate']['accepted']} contained={e['proposal_metrics']['points_inside']}/21",fill='black')
                            thumb=image.resize((576,324)); td=ImageDraw.Draw(thumb)
                            for b,color in [(e['before_box'],'blue'),(e['truth_box'],'green'),(e['proposed_box'],'red')]:
                                td.rectangle([v*576/image.width for v in b],outline=color,width=2)
                            panel.paste(thumb,(8,60)); enlarged=expected.resize((288,288))
                            s=0 if e['side']=='left' else 1
                            obs=obs_cache[by_id[e['sample_id']]['source']['observation_path']]
                            pts=obs['xy_px'][ordinal,33+21*s:54+21*s]
                            x0,y0,x1,y1=e['proposed_box']; ed=ImageDraw.Draw(enlarged)
                            for x,y in pts:
                                xx=(x-x0)*288/(x1-x0); yy=(y-y0)*288/(y1-y0)
                                ed.ellipse((xx-2,yy-2,xx+2,yy+2),fill='red')
                            panel.paste(enlarged,(600,66)); hidden_panels.append((e['visual_index'],panel))
                    found.add(ordinal)
                if found==set(needed): break
            assert found==set(needed)
    hidden_panels.sort(key=lambda x:x[0])
    for start in range(0,len(hidden_panels),6):
        page=Image.new('RGB',(1800,1170),'white')
        for i,(_,panel) in enumerate(hidden_panels[start:start+6]): page.paste(panel,((i%2)*900,(i//2)*390))
        page.save(OUT/'visual'/f'hidden_regressions_{start//6+1:02d}.jpg',quality=94)
    pins=read(OUT/'SOURCE_PINS.json')
    assert all(sha(p)==s for p,s in pins.items())
    report={'status':'PASS','independent_implementation':True,'hidden_events':len(hidden),
            'natural_events':len(natural),'all_gate_and_box_decisions_recomputed':len(stored),
            'metric_rows_recomputed':len(metric_rows),'actual_rgb_crops_redecoded':checked,
            'maximum_pts_error_us':max(timing_errors,default=0),'source_files_unchanged':len(pins),
            'hidden_regression_visual_events':len(selected),'verifier_sha256':sha(__file__),
            'limits':'No anatomical ground truth; repeated source events are not independent subjects.'}
    with (UNIT/'INDEPENDENT_CHECK.json').open('x',encoding='utf-8') as f: json.dump(report,f,indent=2)
    print(json.dumps(report),flush=True)

if __name__=='__main__': main()
