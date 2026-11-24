"""Fixed evidence-only fallback; no fitting or outcome-based choice."""
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np
import sys,json,hashlib,datetime,traceback
sys.stdout.reconfigure(encoding='utf8')
U=Path(__file__).resolve().parent;O=U.parent;F=O/'word100_fractional_phase_alignment_20261026';P=O/'word100_point_sequence_split_20261022'
NORMAL={'fast075','slow15','pace','pause'}
def read(p):return json.loads(Path(p).read_text('utf-8-sig'))
def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def now():return datetime.datetime.now().astimezone().isoformat()
def write(p,x):
    with Path(p).open('x',encoding='utf8') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False)
def check(pins):
    for p,h in pins.items():assert sha(p)==h,p
def pick(old,new):
    # Select each complete 5-component temporal descriptor by availability only.
    # Never take the smaller discrepancy or mix before/middle/after sources.
    previous=old['sequence_comparable'];fresh=new['sequence_comparable'];assert not (fresh&~previous).any()
    source=np.where(fresh,1,np.where(previous,2,0)).astype(np.uint8)
    bad=np.where((source==1)[...,None],new['component_bad'],np.where((source==2)[...,None],old['component_bad'],False))
    return source,bad
def freeze():
    v=read(F/'VERIFIED.json');assert v['status']=='PASS_NUMERIC_AND_PRESERVATION_NOT_SEMANTIC_TRUTH' and not v['gate_passed'];f=read(F/'FROZEN.json');res=read(F/'RESULT.json');check(res['pins']);check(f['protected'])
    a={'sequence_comparable':np.array([True,True,False]),'component_bad':np.array([[1,0,0,0,0],[0,1,0,0,0],[0,0,0,0,0]],bool)}
    b={'sequence_comparable':np.array([True,False,False]),'component_bad':np.array([[0,0,1,0,0],[0,0,0,0,0],[0,0,0,0,0]],bool)}
    src,bad=pick(a,b);assert src.tolist()==[1,2,0] and bad[0,2] and bad[1,1] and not bad[2].any()
    paths=[Path(__file__),U/'verify.py',F/'VERIFIED.json',F/'RESULT.json',F/'FROZEN.json',P/'ANALYSIS.json']
    pins={**res['pins'],**f['pins'],**{str(p):sha(p) for p in paths}}
    write(U/'FROZEN.json',dict(at=now(),pins=pins,protected=f['protected'],cases=res['rows'],
      authorization='Existing autonomous feedback-improvement goal; follow failed fractional phase coverage gate without replacing models or operating.',
      prior_failure='F timingMAE improves but normal sequence evidence lost4644cells. Retain this failure and prior report unchanged. No new parameter or threshold tuning.',
      rule='For each named reference,time,hand,channel: use whole F before/middle/after+2delta descriptor iff all original F validity conditions hold; otherwise retain whole original descriptor if valid, else unknown. No minimum-distance or class-dependent choice. Source states0unknown/1refined/2original-only. Report original-only as refined timing insufficient; never newly observed or confirmed correct. Original pointwise and native missing masks immutable.',
      evaluation='All2808 fixed saved comparisons. Exact original coverage required; normal new warning components and ANY_COMPONENT collapsed-channel warning counted separately. Omission primary reproduces original PSEQ split_any_bad definition: new bad channel absent in own identity, then>=5of8references. 3frame81variants;20pct81. Report strictcommon-F and allavailable separately; cannot equate prior component78 with original collapsed70. Known effective timing weighted by comparable channels with source-specific phase; not all-reference-frame raw F timing. Cross exact timing remains unknown.',
      gate='All invariants/unknown/crossing/rest preservation; no original sequencecoverage loss; known effective comparable-channel timingerror lower; cross normal collapsed warning and component warning decrease; no previously >=5of8 detected shortvariant lost. Even pass researchonly, no semantic validatedfeedback or unseen100claim.',
      limitations='Three repeated-development TRAIN words27clips9IDs. Fallback may retain original timing mistakes. Per-channel sources differ, so no coherent wholepose/wholeaction claim. No known time/deletion info in pick. No classifiertraining/inference/userdata.',
      training=0,production_changed=False))
    print('FROZEN_EVIDENCE_ONLY_FALLBACK',len(res['rows']),flush=True)
def run():
    f=read(U/'FROZEN.json');check(f['pins']);check(f['protected']);(U/'cases').mkdir();pins={};counts=Counter()
    for meta in f['cases']:
        old=load(meta['old']);new=load(meta['output']);src,bad=pick(old,new);name=meta['folder']+'__'+meta['name'];p=U/'cases'/(name+'.npz')
        keep={k:old[k] for k in ('expected','point_comparable','point_bad','native_expected','native_unknown')}
        np.savez_compressed(p,**keep,source=src,sequence_comparable=src>0,component_bad=bad,refined_timing_unavailable=src==2)
        pins[str(p)]=sha(p)
        for k in (0,1,2):counts[str(k)]+=int((old['expected']&(src==k)).sum())
    check(f['protected']);write(U/'RESULT.json',dict(at=now(),status='COMPUTED_PENDING_VERIFY',counts=dict(counts),pins=pins,frozen_sha256=sha(U/'FROZEN.json')));print('COMPLETE',dict(counts))
if __name__=='__main__':
    try:globals()[sys.argv[1]]()
    except Exception:
        write(U/('ERROR_'+now().replace(':','_')+'.json'),dict(trace=traceback.format_exc()));raise
