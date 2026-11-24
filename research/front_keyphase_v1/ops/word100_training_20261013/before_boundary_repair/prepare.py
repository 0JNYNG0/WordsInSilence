"""Restartable one-worker preparation; same production detector/adapter, no classifier."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import traceback
import zipfile

from common import U,BASE,ROOT,OUT,read,write,sha,pin_check
import numpy as np
from wis_lab.capture.landmarker import Landmarker
from wis_lab.capture.video_source import extract
from wis_lab.core.observation import Observation
from wis_lab.service.neural import _channels
from pilot.web_input import to_native_like,active_span
from pilot.data import build_channels

def features(obs,row):
    obs.validate()
    assert obs.n_frames>0 and obs.sidecar['dropped_non_monotonic_pts']==0
    assert np.array_equal(obs.frame_id,np.arange(obs.n_frames))
    assert np.all(np.diff(obs.timestamp_us)>0)
    full,reason,support=_channels(obs)
    if full is None: return None,dict(reason=reason,support=support)
    nat,pr=to_native_like(obs.xy_px,obs.usable); lo,hi=active_span(pr)
    # Separate reconstruction checks production adapter vs exact original row slice.
    ref=build_channels(nat[lo:hi+1],pr[lo:hi+1],obs.timestamp_us[lo:hi+1])
    np.testing.assert_array_equal(ref['x'],full['x'])
    arrays=dict(x_whole=full['x'],whole_frame_id=obs.frame_id[lo:hi+1],whole_ts=obs.timestamp_us[lo:hi+1])
    info=dict(reason=None,support=support,whole_span=[lo,hi])
    if row['role']=='train_candidate':
        a,b=row['intervals'][0]; seconds=obs.timestamp_us/1e6
        core=(seconds>=a)&(seconds<=b)
        assert core.any(), 'Annotation core not in decoded timeline'
        ids=np.flatnonzero((seconds>=a-.3)&(seconds<=b+.3))
        cl=max(lo,int(ids[0])); ch=min(hi,int(ids[-1]))
        assert ch>=cl and pr[cl:ch+1,25:67].any(1).sum()>=4, 'Core context lacks enough observations'
        ctx=build_channels(nat[cl:ch+1],pr[cl:ch+1],obs.timestamp_us[cl:ch+1])
        arrays.update(x_context=ctx['x'],context_frame_id=obs.frame_id[cl:ch+1],context_ts=obs.timestamp_us[cl:ch+1])
        info.update(context_span=[cl,ch],core_frames=int(core.sum()))
    for key in ['x_whole','x_context']:
        if key not in arrays: continue
        x=arrays[key]; assert x.shape[1]==280 and np.isfinite(x).all()
        assert x[0,-1]==0
        assert np.isin(x[:,186:279],[0.,1.]).all()
        bmask=x[:,186:237].astype(bool); hmask=x[:,237:279].astype(bool)
        assert (x[:,:102].reshape(-1,51,2)[~bmask]==0).all()
        assert (x[:,102:186].reshape(-1,42,2)[~hmask]==0).all()
    return arrays,info

def main(limit,max_seconds):
    f=pin_check(); manifest=read(U/'MANIFEST.json')
    pilot=read(BASE/'ops/word_pilot12_inputs_20261013/SCOPE.json')
    compatible=all(sha(p)==v for p,v in pilot['source_sha256'].items())
    old=read('W:/WordsInSilence/data/AIHub_103/pilot12_inputs_20261013/RESULT.json')
    old={r['clip_id']:r for r in old['clips']} if compatible else {}
    for d in ['observations','channels','records','temporary']: (OUT/d).mkdir(parents=True,exist_ok=True)
    contract=read(BASE/'config/contract.json'); start=time.perf_counter(); done=0; archives={}
    try:
        for i,r in enumerate(manifest):
            cid=r['clip_id']; rp=OUT/'records'/f'{cid}.json'
            if rp.exists():
                record=read(rp)
                assert record['manifest_sha256']==f['source_sha256'][str(U/'MANIFEST.json')]
                assert sha(record['observation'])==record['observation_sha256']
                if record['channel_file']: assert sha(record['channel_file'])==record['channel_sha256']
                continue
            if done>=limit or time.perf_counter()-start>max_seconds: break
            assert shutil.disk_usage(OUT).free>500_000_000_000
            e=r['video_entries'][0]; st=time.perf_counter(); temp=None
            reused=cid in old
            if reused:
                prev=old[cid]; assert sha(prev['observation'])==prev['observation_sha256']
                assert sha(prev['source'])==prev['source_sha256']
                obs=Observation.load(prev['observation']); op=Path(prev['observation']); source_hash=prev['source_sha256']
            else:
                archive=archives.get(e['archive'])
                if archive is None: archive=archives[e['archive']]=zipfile.ZipFile(e['archive'])
                zi=archive.getinfo(e['member']); assert zi.CRC==e['crc32'] and zi.file_size==e['bytes']
                assert e['bytes']<200_000_000
                temp=OUT/'temporary'/f'{cid}.mp4'
                with archive.open(zi) as src,temp.open('wb') as dst: shutil.copyfileobj(src,dst,1<<20)
                assert temp.stat().st_size==e['bytes']
                source_hash=sha(temp)
                with Landmarker(contract) as lm: obs=extract(str(temp),lm,cid)
                obs.sidecar.update(archive=e['archive'],archive_member=e['member'],archive_crc32=e['crc32'],source_temporary_file_removed_after_commit=True)
                assert obs.sidecar['source_sha256']==source_hash
                op=OUT/'observations'/f'{cid}.npz';obs.save(str(op))
            arrays,info=features(obs,r)
            cp=None
            if arrays is not None:
                cp=OUT/'channels'/f'{cid}.npz'; np.savez_compressed(cp,**arrays)
            record=dict(clip_id=cid,role=r['role'],signer_id=r['signer_id'],label_index=r['label_index'],
                manifest_sha256=f['source_sha256'][str(U/'MANIFEST.json')],
                observation=str(op),observation_sha256=sha(op),channel_file=str(cp) if cp else None,
                channel_sha256=sha(cp) if cp else None,source_sha256=source_hash,source_bytes=e['bytes'],
                frames=obs.n_frames,quality=obs.quality_summary(),reused_pilot=reused,
                elapsed_seconds=time.perf_counter()-st,feature_check=info,extractor_id=obs.sidecar['extractor_id'])
            write(rp,record)
            if temp:
                assert temp.resolve().parent==(OUT/'temporary').resolve()
                temp.unlink() # only this newly created temporary member, never the original ZIP
            done+=1
            progress=dict(status='PREPARING',pid=os.getpid(),completed=i+1,total=len(manifest),new_this_chunk=done,
                last_clip=cid,last_role=r['role'],elapsed_chunk_seconds=time.perf_counter()-start,last_seconds=record['elapsed_seconds'])
            write(OUT/'progress.json',progress)
            print(json.dumps(progress,ensure_ascii=False),flush=True)
        count=len(list((OUT/'records').glob('*.json')))
        pin_check()
        result=dict(status='PREPARED_PENDING_INDEPENDENT_CHECK' if count==len(manifest) else 'CHUNK_COMPLETE',completed=count,total=len(manifest),new_this_chunk=done,seconds=time.perf_counter()-start)
        write(OUT/'progress.json',result);print(json.dumps(result),flush=True)
    except Exception as exc:
        if not (OUT/'FIRST_FAILURE.json').exists():write(OUT/'FIRST_FAILURE.json',dict(type=type(exc).__name__,message=str(exc),clip=cid if 'cid' in locals() else None,trace=traceback.format_exc()))
        raise
    finally:
        for z in archives.values():z.close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--limit',type=int,default=300);p.add_argument('--max-seconds',type=int,default=7200);args=p.parse_args();main(args.limit,args.max_seconds)
