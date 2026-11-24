"""Bounded public-input preparation. Original files and prior runs remain read-only."""
from pathlib import Path
import sys, json, hashlib, importlib.util, os, datetime, traceback, random, time
sys.dont_write_bytecode = True
U = Path(__file__).resolve().parent
O = U.parent
P = O / 'word100_visible_temporal_targets_20261101'

def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576), b''): h.update(b)
    return h.hexdigest()
def now(): return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat()
def write(p, data, replace=False):
    p=Path(p).resolve(); assert p.is_relative_to(U)
    p.parent.mkdir(exist_ok=True,parents=True)
    if p.exists() and not replace: raise FileExistsError(p)
    tmp=p.with_name(p.name+'.part')
    with tmp.open('x',encoding='utf8') as f: json.dump(data,f,ensure_ascii=False,indent=2,allow_nan=False)
    os.replace(tmp,p)
def event(**d):
    d={'at':now(),**d}
    with (U/'PREP_EVENTS.jsonl').open('a',encoding='utf8') as f: f.write(json.dumps(d,ensure_ascii=False)+'\n')
    print(json.dumps(d,ensure_ascii=False),flush=True)
def helpers():
    s=importlib.util.spec_from_file_location('preserved_native_preparer',P/'setup_prepare.py')
    m=importlib.util.module_from_spec(s); s.loader.exec_module(m); m.U=U
    return m,m.adapters()

def select():
    if (U/'SELECTION.json').exists():
        d=read(U/'SELECTION.json')
        for p,h in d['pins'].items(): assert sha(p)==h,p
        return d
    candidates=read(U/'review/CANDIDATES.json')
    new=candidates['rows']; assert len(new)==48
    old=read(P/'SELECTION.json'); old_inputs={r['clip_id']:r for r in read(P/'INPUTS.json')['rows']}
    rows=[]
    for r in old['rows']:
        if r['role'] not in ('fit','development'): continue
        a=old_inputs[r['clip_id']]
        assert sha(a['input'])==a['input_sha256']
        rows.append(dict(r,existing_source=a['source'],reuse_all=True,centers=a['centers']))
    assert len(rows)==48
    new=sorted(new,key=lambda r:r['clip_id']); random.Random(2026110102).shuffle(new)
    for i,r in enumerate(new,1):
        rec=read(r['record']); source=rec['source']
        rows.append(dict(clip_id=r['clip_id'],word_id=r['word_id'],label=r.get('label',''),person=r['person'],role=r['role'],
            intervals=r['intervals'],record=r['record'],record_sha256=sha(r['record']),
            observation=rec['views']['native']['observation'],observation_sha256=sha(rec['views']['native']['observation']),
            archive_source=source,box=rec['calibration']['box_source_pixels'],frames=rec['frames'],
            existing_source=r.get('existing_source'),centers=None,alias=f'N{i:03}',reused_input=False,reuse_all=False))
    assert len(rows)==len({r['clip_id'] for r in rows})==len({r['archive_source']['sha256'] for r in rows})==96
    from collections import Counter
    roles=dict(Counter(r['role'] for r in rows))
    protected=old['protected']; assert all(sha(p)==h for p,h in protected.items())
    pins=[Path(__file__),U/'AUTHORIZATION.json',U/'PLAN_KO.txt',U/'review/CANDIDATES.json',P/'setup_prepare.py',P/'SELECTION.json',P/'INPUTS.json',P/'DATASET.json',P/'RUBRIC.json']
    result=dict(status='FIXED_BEFORE_NEW_VISUAL_LABELS_OR_SCORES',at=now(),rows=rows,roles=roles,
        history=candidates.get('history',candidates.get('scope')),protected=protected,pins={str(p):sha(p) for p in pins},
        selection='Existing fit/dev48 only, score-blind deterministic eight new fit/dev expressions and eight further evaluation expressions. No consumed8 relabelled fresh.',
        windows='Three13-native-frame windows at fixed25/50/75-percent original interval positions; official interval is temporal context, not frame-wise semantic importance truth.',
        inference='Existing48 input arrays and visual labels reused by SHA. New48 frozen SignRep extraction only. No new detector or backbone weights.',
        research_truth='Codex direct RGB research labels, not expert educational correctness.')
    write(U/'SELECTION.json',result)
    rubric=read(P/'RUBRIC.json')
    rubric.update(status='FROZEN_BEFORE_NEW_RGB_LABELS',truth='Same-Codex direct physical labels; official word is not every-frame shape truth',
        temporal_evidence='Visible before/after shape may be recorded as contextual support. Do not convert it into direct-center label or label unseen intervals unimportant.',
        phase_policy='Still holds, brief changes and boundaries may be core. Whole-clip sequence retained as context; no activity-based semantic labels.',
        evaluation_support='Per target report both-class counts by clip/person/word. At least3 distinct positive and3 negative clips across2 people and2 words per class needed for a broader target-level claim; otherwise descriptive only.',
        fixed_comparison_targets=read(P/'DATASET.json')['selected_targets'])
    write(U/'RUBRIC.json',rubric)
    event(stage='selection_frozen',clips=96,new=48,roles=roles)
    return result

def overview(row, source, obs):
    import av, numpy as np
    from PIL import Image,ImageDraw,ImageFont
    ids=np.unique(np.linspace(0,row['frames']-1,19).round().astype(int)); frames={}
    with av.open(str(source)) as v:
        for i,f in enumerate(v.decode(video=0)):
            if i in ids: frames[i]=f.to_image()
    assert set(frames)==set(ids)
    sheet=Image.new('RGB',(1800,1500),'#20252b'); draw=ImageDraw.Draw(sheet)
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',20)
    for k,i in enumerate(ids):
        x=(k%5)*360; y=(k//5)*375; im=frames[i].copy(); im.thumbnail((354,332))
        sheet.paste(im,(x+(360-im.width)//2,y+36));draw.text((x+5,y+7),f"{row['alias']} f{i} {int(obs['timestamp_us'][i])/1e6:.3f}s",font=font,fill='white')
    p=U/'visuals'/row['alias']/'whole_context.jpg';sheet.save(p,quality=92)
    return dict(file=str(p),frame_ids=[int(i) for i in ids],purpose='Sparse full-clip context only; never dense phase/shape truth')

def run():
    import numpy as np
    if (U/'PREP_RUNNING.json').exists(): raise RuntimeError('Inspect active session and logs before resuming; do not clear by PID alone')
    selected=select(); write(U/'PREP_RUNNING.json',dict(pid=os.getpid(),at=now()))
    m,mod=helpers(); comp=None; inputs=[]; visuals=[]
    prior_inputs={r['clip_id']:r for r in read(P/'INPUTS.json')['rows']}
    prior_visual={r['alias']:r for r in read(P/'VISUAL_INDEX.json')['rows']}
    for row in selected['rows']:
        alias=row['alias']; cp=U/'prepared'/(alias+'.json')
        if cp.exists():
            saved=read(cp)
            for p,h in saved['pins'].items(): assert sha(p)==h,p
            inp=saved['input']; vis=saved['visual']
        elif row['reuse_all']:
            inp=prior_inputs[row['clip_id']];vis=prior_visual[alias]
            assert sha(inp['input'])==inp['input_sha256']
            write(cp,dict(input=inp,visual=vis,pins={inp['input']:inp['input_sha256']},reuse_prior=True))
        else:
            assert sha(row['record'])==row['record_sha256'] and sha(row['observation'])==row['observation_sha256']
            source,method=m.source_file(row);obs=mod.read_observation(row['observation'])
            centers=m.centers_for(row,obs);vis,image_paths=m.make_visuals(row,source,obs,centers)
            vis['whole_context']=overview(row,source,obs);image_paths.append(Path(vis['whole_context']['file']))
            # Annotators can inspect native RGB before the frozen feature encoder finishes.
            write(U/'visual_ready'/(alias+'.json'),vis)
            event(stage='images_ready',alias=alias)
            b=row['box'];xy=obs['xy_px'];obs['usable'] &= (xy[:,:,0]>=b[0])&(xy[:,:,0]<b[2])&(xy[:,:,1]>=b[1])&(xy[:,:,1]<b[3])
            features=mod.extract_features(obs); x=np.nan_to_num(features['x']).astype(np.float32)
            if comp is None: event(stage='gpu_compile');comp=mod.compile_gpu()
            wf,ts,starts,seconds=mod.encode(source,b,len(x),comp)
            assert x.shape==(row['frames'],69) and wf.shape==(len(starts),768)
            assert np.isfinite(wf).all() and np.max(abs(ts-obs['timestamp_us']))<=1
            mid=(ts[np.asarray(starts)]+ts[np.asarray(starts)+15])/2
            rgb=np.stack([np.interp(ts,mid,wf[:,j]) for j in range(768)],1).astype(np.float32)
            dest=U/'inputs'/(alias+'.npz');dest.parent.mkdir(exist_ok=True)
            if dest.exists(): raise RuntimeError('Uncheckpointed input: inspect and preserve before recovery')
            temp=dest.with_suffix('.npz.part')
            with temp.open('xb') as f:np.savez_compressed(f,x=x,mask=features['mask'],rgb=rgb,window_features=wf,window_starts=starts,window_center_us=mid,frame_id=obs['frame_id'],timestamp_us=ts)
            os.replace(temp,dest)
            inp={k:row[k] for k in ['alias','clip_id','word_id','label','person','role','record','record_sha256','observation','observation_sha256','box','frames']}
            inp.update(source=str(source),source_sha256=sha(source),source_method=method,input=str(dest),input_sha256=sha(dest),centers=centers,windows=len(starts),rgb_seconds=seconds,feature_names=features['feature_names'],reused_input=False)
            pins={str(p):sha(p) for p in [dest,source,Path(row['record']),Path(row['observation'])]+image_paths}
            write(cp,dict(input=inp,visual=vis,pins=pins))
            event(stage='clip_complete',alias=alias,seconds=seconds)
        inputs.append(inp);visuals.append(vis)
        write(U/'VISUAL_INDEX.json',dict(status='COMPLETE' if len(inputs)==96 else 'PARTIAL',rows=visuals,samples=[w for v in visuals for w in v['windows']],clips_complete=len(inputs),windows_complete=3*len(inputs),expected_clips=96,metadata_blind=True,no_model_predictions=True),replace=True)
    assert len(inputs)==96 and all(sha(p)==h for p,h in selected['protected'].items())
    write(U/'INPUTS.json',dict(status='PREPARED',rows=inputs,roles=selected['roles'],feature_names=inputs[0]['feature_names']))
    write(U/'PREP_COMPLETED.json',dict(at=now(),clips=96,windows=288,reused_inputs=48,new_fixed_signrep_inputs=48,protected_sha_verified=32))
    (U/'PREP_RUNNING.json').unlink();event(stage='complete',clips=96,windows=288)

if __name__=='__main__':
    try: {'select':select,'run':run}[sys.argv[1]]()
    except BaseException:
        p=U/'PREP_FIRST_FAILURE.txt'
        if not p.exists(): p.write_text(traceback.format_exc(),encoding='utf8')
        raise
