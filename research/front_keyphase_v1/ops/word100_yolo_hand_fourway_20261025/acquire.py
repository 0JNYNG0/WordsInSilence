"""Acquire one user-authorized public checkpoint; never transmit project data."""
from pathlib import Path
import urllib.request,json,hashlib,datetime,zipfile,pickletools
U=Path(__file__).resolve().parent
def get(url):
    req=urllib.request.Request(url,headers={'User-Agent':'WordsInSilence-public-model-evaluation'})
    with urllib.request.urlopen(req,timeout=60) as r:return r.read()
def main():
    assert not (U/'MODEL_SOURCE.json').exists()
    meta=json.loads(get('https://api.github.com/repos/NeuralVulture/yolo-hand-pose/commits/main'))
    commit=meta['sha'];base=f'https://raw.githubusercontent.com/NeuralVulture/yolo-hand-pose/{commit}/'
    files={}
    for src,dst in [('README.md','upstream_README.md'),('train/config.yaml','upstream_config.yaml'),('inference/live.py','upstream_live.txt'),('LICENSE','upstream_LICENSE.txt'),('model/best.pt','yolo_hand_best.pt')]:
        data=get(base+src)
        with (U/dst).open('xb') as f:f.write(data)
        files[dst]={'url':base+src,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}
    # Non-executing pickle opcode inspection; loading is separately restricted.
    with zipfile.ZipFile(U/'yolo_hand_best.pt') as z:
        pickles=[n for n in z.namelist() if n.endswith('data.pkl')]
        assert len(pickles)==1
        globals_=[arg for op,arg,pos in pickletools.genops(z.read(pickles[0])) if op.name=='GLOBAL']
    out={'repository':'NeuralVulture/yolo-hand-pose','commit':commit,'at':datetime.datetime.now().astimezone().isoformat(),'files':files,'pickle_globals':sorted(set(globals_)), 'purpose':'Local four-condition public hand detector evaluation; no training or production replacement','outbound_project_data':False}
    (U/'MODEL_SOURCE.json').write_text(json.dumps(out,indent=2),encoding='utf8')
    print(json.dumps(out,indent=2))
if __name__=='__main__':main()
