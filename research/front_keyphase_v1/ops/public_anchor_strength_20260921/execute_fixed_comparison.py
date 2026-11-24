"""Operational launcher: two predeclared sequential pairs, exclusive output paths."""
from pathlib import Path
import datetime
import hashlib
import json
import subprocess
import sys
import time

OPS=Path(__file__).resolve().parent;WEB=OPS.parent.parent
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
policy=read(OPS/'comparison_policy.json')
assert policy['coefficients']==[.1,.01] and policy['planned_training_runs']==4
keys=['lambda_0p1','lambda_0p01']
for key in keys:
    ready=read(OPS/key/'execution_ready.json')
    assert ready['status']=='READY_FIXED_PUBLIC_ONLY_PAIR'
    assert not Path(ready['out']).exists(), 'Any existing output blocks automatic restart'
    assert not (OPS/key/'training_started.json').exists()
for key in keys:
    folder=OPS/key;ready=read(folder/'execution_ready.json')
    for name,expected in ready['frozen_execution_inputs'].items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==expected,name
    assert not Path(ready['out']).exists()
    argv=[sys.executable,'-u','-B','-X','utf8',str(WEB/'tools_public_anchor_strength_pair.py'),
          '--plan',str(folder/'plan.json'),'--out',ready['out'],'--execute']
    started=time.time()
    with (folder/'training_console.txt').open('x',encoding='utf-8') as log:
        process=subprocess.Popen(argv,cwd=WEB,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='strict')
        with (folder/'training_started.json').open('x',encoding='utf-8') as f:
            json.dump({'pid':process.pid,'argv':argv,'at':datetime.datetime.now().astimezone().isoformat(),'pins_verified_before_Popen':True},f,indent=2)
        for line in process.stdout:
            log.write(line);log.flush();print(key+': '+line,end='',flush=True)
        code=process.wait()
    with (folder/'training_exit.json').open('x',encoding='utf-8') as f:
        json.dump({'exit_code':code,'elapsed_s':time.time()-started,'pid':process.pid},f,indent=2)
    if code:raise SystemExit(code)
print('Both fixed public-only pairs completed.',flush=True)
