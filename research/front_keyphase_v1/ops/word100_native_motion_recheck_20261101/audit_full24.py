"""Apply unchanged24 gates to native readings: reuse9 + newly viewed15, never support12."""
from pathlib import Path
import sys,json,hashlib,importlib.util,copy
sys.dont_write_bytecode=True
N=Path(__file__).resolve().parent;F=N/'full24_native';W=N.parent/'word100_interval_motion_20261101'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):
    with Path(p).open('x',encoding='utf8') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False)

def main():
    freeze=read(F/'AUDIT_FROZEN.json')
    for p,h in freeze['pins'].items():assert sha(p)==h,p
    verified=read(N/'review/REMAINING_EXPORT_VERIFICATION.json')
    assert verified['status'].startswith('PASS') and not verified['failures']
    spec=importlib.util.spec_from_file_location('fixed_presence_audit',W/'audit_presence.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    manifest=copy.deepcopy(read(W/'PILOT_MANIFEST.json'))
    original={r['sample_id']:r for r in manifest['rows']};assert len(original)==24
    native=read(N/'MANIFEST.json')['rows']+read(F/'MANIFEST.json')['rows']
    sources={r['sample_id']:r for r in native};assert len(native)==len(sources)==24 and sources.keys()==original.keys()
    normalized=[];pins=dict(freeze['pins']);rawpaths=[]
    for who in 'AB':
        rows=[]
        for base,count,name in [(N,9,'review/NATIVE_'),(F,15,'REVIEW_')]:
            path=base/f'{name}{who}.json';d=read(path);pins[str(path)]=sha(path);rawpaths.append(str(path))
            assert d['status']=='COMPLETE' and len(d['rows'])==count
            assert d['rubric_sha256']==sha(W/'RUBRIC.json') and d['native_manifest_sha256']==sha(base/'MANIFEST.json')
            local={r['sample_id']:r for r in d['rows']};expected={r['sample_id'] for r in read(base/'MANIFEST.json')['rows']}
            assert len(local)==count and local.keys()==expected
            for sid,raw in local.items():
                src=sources[sid];frames=[f['frame_id'] for f in src['frames']]
                assert frames==original[sid]['frame_ids']
                assert [f['timestamp_us'] for f in src['frames']]==original[sid]['timestamps_us']
                assert raw['alias']==src['alias'] and raw['actual_image_viewed'] is True
                allowed={p['file']:(p['sha256'],p['frame_ids']) for p in src['pairs']}
                for f in src['frames']:
                    for k in ('full','fixed'):allowed[f[k]]=(f[k+'_sha256'],[f['frame_id']])
                seen=set()
                for im in raw['images_viewed']:
                    h,ids=allowed[im['file']];assert h==im['sha256']==sha(im['file']);seen.update(ids)
                assert seen==set(frames)
                proof=hashlib.sha256(json.dumps(src['frames'],sort_keys=True).encode()).hexdigest()
                r=copy.deepcopy(raw);r['image_sha256']=proof;rows.append(r)
                # Transparent metadata-only adapter after native evidence verification.
                # Retains source24 centre, strata, person, clip and frame IDs. No raw label edit.
                original[sid]['alias']=src['alias'];original[sid]['image_sha256']=proof
        assert len(rows)==24 and len({r['sample_id'] for r in rows})==24
        normalized.append(dict(status='COMPLETE',rows=rows))
    result=m.calculate(manifest,normalized,read(W/'PILOT_GATES.json'))
    result.update(review_scope='Same24 retrospective native-view reassessment; raw9 exact reuse +15 new views.',
        source_pins=pins,raw_review_paths=rawpaths,native_frames=312,reused_frames=117,new_frames=195,
        support12_excluded_from_every_gate=True,original24_failed_audit_unchanged=True,
        training_runs=0,new_inference_runs=0,
        independent_human_truth=False,implementation='Codex root; separate Codex source/audit review.')
    write(F/'FULL24_AUDIT.json',result)
    print(json.dumps({k:result[k] for k in ('status','checks','metrics','contract_failures')},ensure_ascii=False))
if __name__=='__main__':main()
