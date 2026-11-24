"""Materialize a reusable, isolated inference adapter from the frozen G4 code.
No source labels or saved predictions are used by the generated runtime.
"""
from pathlib import Path
import json, hashlib
U=Path(__file__).resolve().parent;O=U.parent
src=(O/'word100_public_g4_20261114/automatic.py').read_text('utf-8-sig')
imports=src[:src.index('def sources():')]
imports=imports.replace('from prepare import U,O,read,sha,write,load,Path',
'''from pathlib import Path
import sys
O=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(O/'word100_public_g4_20261114'))
from prepare import read,sha,write,load
from conflict_guard import qualify''')
init=src[src.index(' net=rc.frozen_observer()'):src.index(" (U/'auto').mkdir")]
init=init.replace(' net=', ' net=',1)
init='def initialize():\n'+init+' return net,base,rn,traits,heads,caps\n'
body=src[src.index("  x=ch['tcn280']"):src.index('  write(p,dict(rows=variants')]
body=body.replace("specs=[s for s in rc.FAMILIES[sr['word']]", "specs=[s for s in rc.FAMILIES[word]")
body=body.replace("sr['word']",'word')
body=body.replace("for condition in ['old_local_full','current_g3']:","for condition in ['current_g3']:")
needle='    order=core_order(cards);association='
body=body.replace(needle,'''    lookup={c['card']:c for c in cards};cards=[lookup[s['id']] for s in specs]
    allowed={tuple([a['id'],b['id']]) for a,b in zip(specs,specs[1:]) if a['families']!=b['families']}
    order=[p for p in core_order(cards) if tuple(p['cards']) in allowed];association=''')
body='\n'.join(line[1:] if line.startswith(' ') else line for line in body.splitlines())+'\n'
run='''def infer(models,ch,raw,word):
 net,base,rn,traits,heads,caps=models
 tick=time.perf_counter();stop=len(ch['frame_id']);rate=1;cid='live';kind='live'
 if word not in rc.FAMILIES:raise ValueError('Unsupported feedback word')
 if stop<4:raise ValueError('At least four observed frames required')
 if stop>480:raise ValueError('Recording exceeds the 15-second capture envelope')
'''+body+''' result=qualify(variants)
 result['runtime']=dict(version='G6-frozen-G4',frames=stop,candidate_windows=len(sp),seconds=time.perf_counter()-tick,
                        trained=False,reference_video_reads=0,rgb_model_calls=0)
 return result
'''
(U/'engine.py').write_text(imports+init+run,encoding='utf-8')
(U/'CONTRACT.json').write_text(json.dumps(dict(
 goal='Connect frozen G4 qualified evidence + G5 visual contract to new capture; public verification before user-started webcam.',
 implementation='Codex',verification='Separate Codex checks, not independent human expert',
 no_training=True,no_user_tuning=True,baseline_preserved='C·T3-F and common100',
 gates=['G4 replay equivalence','quarantine input contract','public camera-loop countdown/auto-end/occlusion','HTTP and browser feedback/time','user-started physical webcam acceptance'],
 completion='Only actual physical capture acceptance closes G6; no camera opened automatically',
 g4_source_sha=hashlib.sha256(src.encode()).hexdigest(),
 performance='Separate camera/queue drain, TCN+DTW, latest evidence, browser delivery; no accuracy-preserving claim without equivalence'),ensure_ascii=False,indent=2),encoding='utf-8')
print('runtime adapter materialized')
