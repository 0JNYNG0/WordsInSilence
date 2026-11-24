from pathlib import Path
import json, urllib.request, hashlib

U=Path(__file__).resolve().parent
names=['repvit_m1_0.dist_450e_in1k','mobilenetv4_hybrid_medium.ix_e550_r256_in1k']
records=[]
for name in names:
    p=U/'models'/(name+'.safetensors')
    info=U/'models'/(name+'.source.json')
    if p.exists() or info.exists():
        assert p.exists() and info.exists(), 'Partial acquisition; inspect without overwriting'
        r=json.loads(info.read_text());assert hashlib.sha256(p.read_bytes()).hexdigest()==r['sha256']
        records.append(r);continue
    with urllib.request.urlopen('https://huggingface.co/api/models/timm/'+name,timeout=40) as response:
        meta=json.load(response)
    assert not meta.get('gated') and meta.get('cardData',{}).get('license')=='apache-2.0'
    revision=meta['sha']
    url='https://huggingface.co/timm/'+name+'/resolve/'+revision+'/model.safetensors'
    part=p.with_suffix(p.suffix+'.partial');assert not part.exists()
    with urllib.request.urlopen(url,timeout=60) as src,part.open('xb') as dst:
        while chunk:=src.read(1024*1024):dst.write(chunk)
    assert 1000000<part.stat().st_size<60000000
    part.rename(p)
    r={'name':name,'revision':revision,'url':url,'file':str(p),'bytes':p.stat().st_size,
       'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'license_from_card':'apache-2.0','gated':False}
    info.write_text(json.dumps(r,indent=2),encoding='utf-8');records.append(r)
    print(json.dumps(r),flush=True)
(U/'MODEL_ACQUISITION.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
