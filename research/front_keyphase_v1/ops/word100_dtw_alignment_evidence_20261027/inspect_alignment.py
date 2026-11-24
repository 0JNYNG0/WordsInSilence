"""Read-only reconstruction of saved webcam DTW. No fitting or classifier inference."""
from pathlib import Path
import sys, json, hashlib, zipfile
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np

U=Path(__file__).resolve().parent
LAB=U.parents[1]; REPO=LAB.parents[1]
sys.path[:0]=[str(LAB/'src'),str(REPO/'research/sign_model_pilot_v1/src')]
sys.dont_write_bytecode=True
from wis_lab.core.observation import Observation
from wis_lab.service import word100_analysis as A, word100_role_dtw as R

SESSION=LAB/'runs/ses_067e0348ff32'
DETAIL_WORDS=['WORD1290','WORD1139','WORD0949']
def read(p):return json.loads(Path(p).read_text('utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(name,value):
    with (U/name).open('x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False)
def now():return datetime.now(ZoneInfo('Asia/Seoul')).isoformat()
def check(pins):
    for p,h in pins.items():assert sha(p)==h,p

def freeze():
    assert not (U/'FROZEN.json').exists()
    old=read(LAB/'ops/word100_visible_coordinate_reference_20261026/FROZEN.json')
    protected=old['protected'];check(protected)
    session=read(SESSION/'session.json');assert session['state']=='CLOSED' and not session['camera']
    refs={x['word_id']:x for x in read(LAB/'config/word100_analysis_references.json')['items']}
    rolecfg=read(LAB/'config/word100_role_dtw.json');assert sha(rolecfg['bank_path'])==rolecfg['bank_sha256']
    bank=read(rolecfg['bank_path']); rows=[];pins={}
    for tid in session['trial_ids']:
        base=SESSION/'trials'/tid;res=read(base.with_suffix('.result.json'));word=res['requested_word_id'];ref=refs[word]
        assert res['reference_binding']['demonstration_clip_id']==ref['clip_id']
        assert ref['source_role']=='train_candidate' and 1<=ref['signer_id']<=9
        assert sha(ref['observation'])==ref['observation_sha256']
        assert sha(ref['video_path'])==ref['source_sha256']
        assert sha(base.with_suffix('.npz'))==res['observation_sha256']
        stored=base.with_name(tid+'.role_dtw_'+rolecfg['bank_sha256'][:12]+'.json');d=read(stored)
        assert d['query_observation_sha256']==res['observation_sha256']
        paths=[base.with_suffix('.result.json'),base.with_suffix('.npz'),base.with_suffix('.sidecar.json'),stored,Path(ref['observation']),Path(ref['observation']).with_suffix('.sidecar.json'),Path(ref['video_path'])]
        if word in DETAIL_WORDS:paths.append(base.with_suffix('.capture.zip'))
        br=next(x for x in bank['words'][word]['references'] if x['clip_id']==ref['clip_id'])
        paths.append(Path(br['sample_path']))
        for p in paths:pins[str(p)]=sha(p)
        rows.append(dict(trial_id=tid,word_id=word,label=ref['label'],base=str(base),reference=ref,
                         required=d['required_sides'],stored=str(stored),bank_sample=br['sample_path'],detail=word in DETAIL_WORDS))
    assert len(rows)==12
    for p in [Path(__file__),Path(A.__file__),Path(R.__file__),LAB/'src/wis_lab/core/features.py',LAB/'src/wis_lab/core/dtw.py',LAB/'src/wis_lab/core/observation.py']:
        pins[str(p)]=sha(p)
    write('FROZEN.json',dict(at=now(),purpose='Explain existing DTW on explicitly authorized previous user recordings; never training/tuning/selection',
        selection='All12 saved takes, bound demonstrated TRAIN reference, not lowest-distance reference. Detailed views preselected 감사/믿다/멀다.',
        operation='Restore same sampled features and path; compare against same-progress nearest frames; verify against saved metrics',
        pipeline='Original stored MediaPipe observation for DTW, not C/T3F observation; no re-detection/YOLO/HaMeR/AIHub coordinate repair',
        sampling_budget=A.BUDGET,band=A.BAND,group_weights=A.F.GROUPS,rows=rows,sources=pins,protected=protected,
        no_training=True,no_classifier_inference=True,no_source_write=True,no_network=True,no_camera=True,
        limits=['User footage only for this explanation/evaluation. No new quality thresholds.',
                'DTW minimizes combined feature cost, not semantic correctness. No alignment ground truth.',
                'One displayed reference per word, not full-bank aggregate.',
                'Saved RGB is at most10fps; nearest actual RGB time gap displayed. No fabricated in-between RGB.',
                'Before/after have different pair multiplicity/support. Lower cost alone is not accuracy.']))
    print('FROZEN12',flush=True)

def metrics(q,r,qi,ri,required):
    rows=[]
    for side in ('L','R'):
        for name in ('handpos','bend'):
            x,xv=A._group(q,name+'_'+side);y,yv=A._group(r,name+'_'+side)
            valid=xv[qi]&yv[ri]; a=x[qi[valid]];b=y[ri[valid]]
            if name=='handpos':err=np.linalg.norm(a-b,axis=1)*100
            else:err=np.abs(a[:,:15]-b[:,:15]).mean(axis=1)*180
            rows.append(dict(side=side,group=name,required=side in required,paired=int(valid.sum()),pairs=len(qi),
                             support=float(valid.mean()),difference=float(err.mean()) if len(err) else None,
                             unit='shoulder_width_percent' if name=='handpos' else '2D_degrees'))
    return rows

def run():
    f=read(U/'FROZEN.json');check(f['sources']);check(f['protected']);results=[]
    assert not (U/'RESULT.json').exists()
    (U/'arrays').mkdir(exist_ok=True)
    for row in f['rows']:
        obs=Observation.load(row['base']+'.npz');refobs=Observation.load(row['reference']['observation'])
        q=A._sample(obs);r=A._sample(refobs)
        with np.load(row['bank_sample'],allow_pickle=False) as z:
            assert np.array_equal(r['features'],z['features'],equal_nan=True)
            assert np.array_equal(r['present'],z['present'])
        weights=None if len(row['required'])==2 else {g:w if g.endswith('_'+row['required'][0]) else 0. for g,(_,w) in A.F.GROUPS.items()}
        cost,evidence=A.frame_cost_matrix(q,r,group_weights=weights,hand_participation=False,one_sided_mode='flat')
        path=np.array(A._optimal_path(cost,A.BAND),dtype=np.int64);assert path.ndim==2 and len(path)
        qi,ri=path.T;n=max(cost.shape);lq=np.rint(np.linspace(0,cost.shape[0]-1,n)).astype(int);lr=np.rint(np.linspace(0,cost.shape[1]-1,n)).astype(int)
        saved=next(p['result'] for p in read(row['stored'])['comparisons'] if p['reference_clip']==row['reference']['clip_id'])
        live=R.compare_samples(q,r,row['required'])
        assert live['status']==saved['status']=='OK'
        for k in ['path_pairs','alignment_evidence_fraction','dtw_distance']:
            assert abs(live[k]-saved[k])<1e-7,(row['label'],k,live[k],saved[k])
        after=metrics(q,r,qi,ri,row['required']);before=metrics(q,r,lq,lr,row['required'])
        for m in after:
            if not m['required']:continue
            key='location' if m['group']=='handpos' else 'handshape'
            prev=next(z for z in saved['metrics'] if z['key']==key)
            prev=next(z for z in prev['sides'] if z['side']==m['side'])
            assert m['paired']==prev['compared_pairs']
            assert (m['difference'] is None and prev['value'] is None) or np.isclose(m['difference'],prev['value'],atol=2e-5)
        out=dict(**row,query_frames=obs.n_frames,reference_frames=refobs.n_frames,query_sampled=len(q['features']),reference_sampled=len(r['features']),
            query_span=[float(q['timestamp_us'][0]/1e6),float(q['timestamp_us'][-1]/1e6)],reference_span=[float(r['timestamp_us'][0]/1e6),float(r['timestamp_us'][-1]/1e6)],
            path_pairs=len(path),dtw_sum=float(cost[qi,ri].sum()),linear_sum=float(cost[lq,lr].sum()),dtw_mean=float(cost[qi,ri].mean()),linear_mean=float(cost[lq,lr].mean()),
            evidence=float(evidence[qi,ri].mean()),linear_evidence=float(evidence[lq,lr].mean()),before=before,after=after,
            repeat_query_steps=int((np.diff(qi)==0).sum()),repeat_reference_steps=int((np.diff(ri)==0).sum()),
            all_query_samples_covered=len(set(qi))==len(q['features']),all_reference_samples_covered=len(set(ri))==len(r['features']),
            saved_match=True,semantic_alignment_verified=False)
        arrays={}
        for prefix,s in [('q',q),('r',r)]:
            for key in ['features','present','timestamp_us','frame_id']:arrays[prefix+'_'+key]=s[key]
        arrays.update(cost=cost,evidence=evidence,path=path,linear=np.column_stack([lq,lr]))
        target=U/'arrays'/f"{row['word_id']}.npz";assert not target.exists();np.savez_compressed(target,**arrays)
        out['array_sha256']=sha(target);results.append(out)
        print(row['label'],round(out['linear_mean'],3),round(out['dtw_mean'],3),round(out['evidence'],3),flush=True)
    check(f['sources']);check(f['protected'])
    write('RESULT.json',dict(at=now(),rows=results,status='RECONSTRUCTED12_SAVED_METRICS_MATCH',training=0,tuning=0,classifier_inference=0,detector_inference=0,operating_changes=0))

if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    {'freeze':freeze,'run':run}[sys.argv[1]]()
