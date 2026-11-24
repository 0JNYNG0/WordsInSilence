"""Frozen DEV-to-TRAIN descriptive DTW replay. No detector, classifier, or training."""
from pathlib import Path
import os, sys, json, hashlib, datetime, importlib.util, time, traceback
from collections import defaultdict, Counter
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'): os.environ.setdefault(key,'1')
sys.dont_write_bytecode=True
sys.stdout.reconfigure(encoding='utf8')
import numpy as np
U=Path(__file__).resolve().parent; O=U.parent
B=O/'word100_train_reference_bank_20261026'; D=O/'word100_dev_image_feedback_20261026'; Q=O/'word100_recognition_feedback_candidate_20261026'; T=O/'word100_repeat_sample_alignment_20261026'
sys.path.insert(0,str(B)); import prepare as R
M=R.mod('dev_train_alignment',O/'word100_multipeople_feedback_integration_20261026/run.py')
G=R.mod('dev_train_ranges',B/'range.py'); C=R.mod('dev_train_repeat',T/'candidate.py')
read=R.read; load=R.load; sha=R.sha; write=R.write; check=R.check; now=R.now
GROUPS=(('angle',0,15),('position',15,23))

def frozen_inputs(f):
    samples={r['clip_id']:load(r['output']) for r in f['train']+f['dev']}
    for z in samples.values(): z['quality_eligible']=np.zeros_like(z['quality_eligible'])
    return samples

def counts_of(z):
    c=Counter(pairs=1)
    for mode in ('point','sequence'):
        for name,a,b in GROUPS:
            c[f'{mode}_{name}_domain']=int(z['domain'][:,:,a:b].sum())
            c[f'{mode}_{name}_raw']=int(z[mode+'_ok'][:,:,a:b].sum())
            for k in range(1,6): c[f'{mode}_{name}_state{k}']=int((z[mode+'_status'][:,:,a:b]==k).sum())
    return c

def freeze():
    assert not(U/'FROZEN.json').exists()
    tr=read(B/'PREPARED.json'); dv=read(D/'PREPARED.json'); ad=read(B/'ALIGNMENT_DONE.json'); prior=read(Q/'MANIFEST.json')
    assert read(B/'INPUTS_VERIFIED.json')['counts']['clips']==899
    assert read(B/'ALIGNMENT_VERIFIED.json')['counts']['pairs']==7184
    assert read(D/'FEEDBACK_VERIFIED.json')['queries']==300
    assert read(T/'PREFLIGHT.json')['tests']==8
    train=[{k:r[k] for k in ('clip_id','word_code','label','signer_id','output')} for r in tr['rows']]
    dev=[dict(clip_id=r['clip_id'],word_code=r['source']['word_code'],label=r['source']['label'],signer_id=r['source']['signer_id'],output=r['output']) for r in dv['rows']]
    assert len(train)==899 and len(dev)==300
    assert {r['signer_id'] for r in train}==set(range(1,10)) and {r['signer_id'] for r in dev}=={11,12,13}
    assert not ({r['clip_id'] for r in train}&{r['clip_id'] for r in dev})
    by=defaultdict(list)
    for r in train: by[r['word_code']].append(r)
    pairs=[dict(query=q['clip_id'],reference=r['clip_id'],word_code=q['word_code']) for q in sorted(dev,key=lambda x:x['clip_id']) for r in sorted(by[q['word_code']],key=lambda x:x['clip_id'])]
    assert len(pairs)==2697 and len(by)==100
    paths=[Path(__file__),U/'AUTHORIZATION.json',B/'PREPARED.json',B/'INPUTS_VERIFIED.json',B/'ALIGNMENT_DONE.json',B/'ALIGNMENT_VERIFIED.json',B/'RANGE_PROTOCOL.json',B/'RANGE_VERIFIED.json',D/'PREPARED.json',D/'FEEDBACK_VERIFIED.json',Q/'MANIFEST.json',Q/'VERIFIED.json',T/'candidate.py',T/'VERIFIED.json',T/'PREFLIGHT.json',Path(M.__file__),Path(M.D.__file__),Path(M.F.__file__),Path(M.P.__file__),Path(G.__file__),Path(R.__file__)]
    pins={str(p):sha(p) for p in paths}; pins.update(tr['outputs']); pins.update(dv['pins']); pins.update(ad['pins'])
    check(pins); protected=read(B/'FROZEN.json')['protected']; check(protected)
    inventory={}; observations=defaultdict(set)
    for row in train+dev:
        z=load(row['output']); h=hashlib.sha256(z['xy_px'].tobytes()+z['usable'].tobytes()+z['frame_id'].tobytes()).hexdigest(); observations[h].add('dev' if row['signer_id']>9 else 'train')
        z['quality_eligible']=np.zeros_like(z['quality_eligible']); starts,ends,ids=C.groups(z)
        inventory[row['clip_id']]=dict(frames=len(ids),exact_repeated_rows=len(ids)-len(starts),observation_sha256=h)
    assert all(len(v)==1 for v in observations.values()),'Exact observation duplicated across split'
    # Existing archive metadata is an additional duplicate screen, not biological identity proof.
    bp=O/'word100_body_boundary_20261017/PREP_CONTEXT.json'; pins[str(bp)]=sha(bp)
    meta={r['clip_id']:r for r in read(bp)['ordered_rows']}; archive_keys=defaultdict(set)
    for row in train+dev:
        for entry in meta[row['clip_id']]['video_entries']: archive_keys[(entry['bytes'],entry['crc32'])].add('dev' if row['signer_id']>9 else 'train')
    assert all(len(v)==1 for v in archive_keys.values()),'Archive size/CRC duplicate across split needs inspection'
    protocol=dict(
      hypothesis='Can fixed TRAIN-only named-reference bands separate insufficient observation from measured DEV differences without changing any model, raw coordinates, assignment, reference domain or score?',
      calibration='Each TRAIN reference retains all its7/8 other TRAIN recordedIDs as calibrators. >=5supported people, unchanged maximum absolute residual rule. No DEV in bound, no mean skeleton, no best reference selection.',
      alignment='Existing flat1.5/band.2 grouped DTW and bounded1frame fractional sequence. Research copies set quality_eligible0 consistently with TRAIN calibration; originals preserved. Exact observation repeats collapsed only for path search using frozenT; all source evidence windows restored. This is not claimed identical to previous DEV-to-DEV D paths.',
      baseline='Raw point/sequence measurements on the same named-reference path are retained. New states add image corroboration and fixed empirical band. Prior D twoDEV references are a different reference population, not a causal accuracy baseline. No lower residual or within-band percentage is labelled improvement.',
      interpretation='0 outside reference domain,1 raw missing,2 image unconfirmed,3 fewer than5 supported TRAIN peers,4 within observed band,5 beyond observed band. 4 is not correct sign;5 is not user error. Finger evidence is sameMediaPipe full/crop corroboration, not independent anatomical truth.',
      controls='One query perword chosen by smallest ID,first named TRAIN reference. At fixed observed path: one-angle missing and whole-hand missing preserve other angles/hand/arm and domain. End-to-end controls additionally test masks change alignment on deterministic10words, including short3frame gaps. Hidden values cannot silently count as normal. No semantic essential-motion groundtruth claimed.',
      gates='2697complete; split/duplicate checks; all raw inputs protected; independent reconstruction of every saved difference/support/band/status; fixed denominator; unknown never within; all original people/references retained. Report support losses and out-of-band alongside coverage, no runtime correction command.',
      scope='Development-only public100words: not unseen final user test, not3000class expansion. No camera/download/training/detector/TCN inference/personal/mentor/extra199/200 input or operating replacement.')
    write(U/'FROZEN.json',dict(at=now(),train=train,dev=dev,pairs=pairs,calibration_pairs=ad['rows'],inventory=inventory,pins=pins,protected=protected,recognition=prior['rows'],protocol=protocol,minimum_peers=5,new_training=0,new_inference=0))
    for name in ('bounds','bound_records','pairs','records','starts','controls','queries'): (U/name).mkdir()
    write(U/'PREFLIGHT.json',dict(at=now(),train=899,dev=300,pairs=2697,split_ids_disjoint=True,cross_split_exact_observation_duplicates=0,cross_split_archive_size_crc_duplicates=0,actual_person_identity_certified=False,exact_repeated_rows=sum(x['exact_repeated_rows'] for x in inventory.values())))
    print('FROZEN1199_INPUTS_2697_PAIRS',flush=True)

def calibrate():
    f=read(U/'FROZEN.json'); check(f['pins']); check(f['protected']); samples=frozen_inputs(f); by=defaultdict(list)
    for p in f['calibration_pairs']: by[p['reference']].append(p)
    rows=[]; pins={}; start=time.perf_counter()
    for rid,pp in sorted(by.items()):
        rec=U/'bound_records'/(rid+'.json')
        if rec.exists():
            old=read(rec); assert old['context']==sha(U/'FROZEN.json'); check(old['pins']); rows.append(old['row']); pins.update(old['pins']); continue
        ds=[]; ss=[]; pc=[]; sc=[]
        for p in pp:
            z=load(p['pair']); _,_,a,b=G.support(z,samples[p['query']],samples[rid]); ds.append(z['point_delta']); ss.append(z['sequence_delta']); pc.append(a); sc.append(b)
        pb,pn=G.bound_from(np.array(ds),np.array(pc)); sb,sn=G.bound_from(np.array(ss),np.array(sc))
        output=U/'bounds'/(rid+'.npz'); np.savez_compressed(output,point_bound=pb,point_peer_count=pn,sequence_bound=sb,sequence_peer_count=sn)
        row=dict(reference=rid,calibration_ids=sorted(p['query'] for p in pp),output=str(output)); pin={str(output):sha(output)}
        write(rec,dict(context=sha(U/'FROZEN.json'),row=row,pins=pin)); rows.append(row); pins.update(pin)
        if len(rows)%100==0: print('TRAIN_ONLY_BOUNDS',len(rows),899,round(time.perf_counter()-start,1),flush=True)
    assert len(rows)==899; check(f['protected']); write(U/'BOUNDS_DONE.json',dict(at=now(),rows=rows,pins=pins,seconds=time.perf_counter()-start))

def annotate(z,q,r,b):
    po,se,_,_=G.support(z,q,r)
    for mode,e in [('point',po),('sequence',se)]:
        z[mode+'_evidence']=e; z[mode+'_status']=G.classify(z['domain'],z[mode+'_ok'],e,z[mode+'_delta'],b[mode+'_bound'],b[mode+'_peer_count'])
    return z

def run():
    f=read(U/'FROZEN.json'); bounds=read(U/'BOUNDS_DONE.json'); check(f['pins']); check(bounds['pins']); check(f['protected'])
    assert not(U/'RUNNING.json').exists(); write(U/'RUNNING.json',dict(at=now(),pid=os.getpid()))
    samples=frozen_inputs(f); bm={r['reference']:r for r in bounds['rows']}; bc={}; rows=[]; pins={}; total=Counter(); start=time.perf_counter()
    for i,p in enumerate(f['pairs']):
        name=p['query']+'__'+p['reference']; rec=U/'records'/(name+'.json'); sp=U/'starts'/(name+'.json')
        if rec.exists():
            old=read(rec); assert old['context']==sha(U/'FROZEN.json'); check(old['pins']); rows.append(old['row']); pins.update(old['pins']); total.update(old['counts']); continue
        assert not sp.exists(),'Partial run preserved: inspect before recovery'; write(sp,dict(at=now()))
        q=samples[p['query']]; r=samples[p['reference']]; repeat=any(f['inventory'][cid]['exact_repeated_rows'] for cid in [p['query'],p['reference']])
        startpair=time.perf_counter(); z=C.pair(q,r,M) if repeat else M.pair(q,r); seconds=time.perf_counter()-startpair
        if p['reference'] not in bc:bc[p['reference']]=load(bm[p['reference']]['output'])
        annotate(z,q,r,bc[p['reference']]); output=U/'pairs'/(name+'.npz'); np.savez_compressed(output,**z)
        count=counts_of(z); count['exact_repeat_realignment']=int(bool(repeat)); row=dict(p,output=str(output),seconds=seconds,repeat=bool(repeat),counts=dict(count)); pin={str(output):sha(output)}
        write(rec,dict(at=now(),context=sha(U/'FROZEN.json'),row=row,counts=dict(count),pins=pin)); rows.append(row); pins.update(pin); total.update(count)
        if (i+1)%100==0:print('DEV_TRAIN_PAIRS',i+1,2697,'seconds',round(time.perf_counter()-start,1),flush=True)
    assert len(rows)==2697; check(f['protected']); write(U/'DONE.json',dict(at=now(),rows=rows,pins=pins,counts=dict(total),seconds=time.perf_counter()-start)); (U/'RUNNING.json').unlink()
    print('DONE2697',dict(total),flush=True)

if __name__=='__main__':
    try: globals()[sys.argv[1]]()
    except BaseException:
        p=U/('FIRST_ERROR_'+sys.argv[1]+'.txt')
        if not p.exists():p.write_text(traceback.format_exc(),encoding='utf8')
        raise
