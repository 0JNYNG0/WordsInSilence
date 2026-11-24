"""Separate saved candidate selection and paired arithmetic verification."""
from pathlib import Path
import json,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parent;U=ROOT/'offset_candidate';O=ROOT.parent
P=O/'word100_point_sequence_split_20261022'
def read(p):return json.loads(Path(p).read_text('utf-8-sig'))
def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
f=read(U/'FROZEN.json');r=read(U/'RESULT.json');totals={};inc={};rows=[]
lookup={(m['folder'],m['name']):m for m in r['rows']};local={}
normal={'identity','fast075','slow15','pace','pause'}
windows=0;chosen_count=0;unknown_veto=0
for m in r['rows']:
    z=load(U/'cases'/(m['folder']+'__'+m['name']+'.npz'));old=load(P/m['folder']/(m['name']+'.npz'))
    for k in ('expected','point_comparable','point_bad','sequence_comparable'):assert np.array_equal(z[k],old[k])
    assert np.array_equal(z['raw_sequence_bad'],old['component_bad'])
    assert np.array_equal(z['timing_sensitive'],z['raw_sequence_bad']&~z['remaining'])
    assert not (z['remaining']&~z['raw_sequence_bad']).any()
    for j in range(len(z['expected'])):
        for h in range(2):
            before=z['raw_sequence_bad'][j,h];valid=z['sequence_comparable'][j,h]
            best=0;selected=0;expected=before
            for k,delta in enumerate((-1,1)):
                new=z['option_bad'][k,j,h];v=z['option_valid'][k,j,h]
                supported=all(v[c] for c in np.flatnonzero(valid))
                added=any(new[c,a] and not before[c,a] for c in np.flatnonzero(valid) for a in range(5))
                removed=sum(before[c,a] and not new[c,a] for c in np.flatnonzero(valid) for a in range(5))
                if not supported and removed:unknown_veto+=1
                if supported and not added and removed>best:best=removed;selected=delta;expected=new&valid[:,None]
            assert selected==z['chosen_half_frames'][j,h]
            assert np.array_equal(expected,z['remaining'][j,h]);windows+=1;chosen_count+=selected!=0
    cohort=m['folder']+'/'+('normal' if m['kind'] in normal else ('short_delete' if m['kind'].startswith('skip3') else 'altered'))
    a=totals.setdefault(cohort,dict(cases=0,raw_flags=0,remaining_flags=0,sensitive_flags=0))
    a['cases']+=1
    for key,arr in [('raw_flags','raw_sequence_bad'),('remaining_flags','remaining'),('sensitive_flags','timing_sensitive')]:a[key]+=int(z[arr].sum())
    if m['folder'] in ('known','cross') and m['kind']!='identity':
        base='identity__'+m['query']+('__'+m['reference'] if m['folder']=='cross' else '')
        bz=load(U/'cases'/(m['folder']+'__'+base+'.npz'))
        common=bz['sequence_comparable']&z['sequence_comparable']
        before_new=z['raw_sequence_bad'].any(-1)&~bz['raw_sequence_bad'].any(-1)&common
        after_new=z['remaining'].any(-1)&~bz['remaining'].any(-1)&common
        a=inc.setdefault(cohort,dict(cases=0,common=0,raw_new=0,remaining_new=0,resolved=0,added=0))
        a['cases']+=1;a['common']+=int(common.sum());a['raw_new']+=int(before_new.sum());a['remaining_new']+=int(after_new.sum())
        a['resolved']+=int((before_new&~after_new).sum());a['added']+=int((after_new&~before_new).sum())
        if m['kind'].startswith('skip3'):
            original=load(P/m['folder']/(base+'.npz'))
            removed=np.setdiff1d(np.arange(len(original['source_index'])),old['source_index'])
            # Same deleted-source-region mapping as old saved identity path.
            target=np.zeros(len(common),bool)
            for qi,ri in original['path']:
                if qi in removed:target[ri]=True
            before_detect=bool((before_new&target[:,None,None]).any());after_detect=bool((after_new&target[:,None,None]).any())
            local.setdefault((m['folder'],m['query'],m['kind']),[]).append((before_detect,after_detect))
for p,h in f['pins'].items():assert sha(p)==h,p
for p,h in f['protected'].items():assert sha(p)==h,p
deletions={}
for folder in ('known','cross'):
    cases=[v for (fo,cid,k),v in local.items() if fo==folder]
    d={'variants':len(cases),'anchors_per_variant':sorted(set(len(x) for x in cases))}
    for refs in ([1] if folder=='known' else [1,5]):
        pairs=[(sum(a for a,b in x)>=refs,sum(b for a,b in x)>=refs) for x in cases]
        d[str(refs)+'refs']=dict(before=sum(a for a,b in pairs),after=sum(b for a,b in pairs),lost=sum(a and not b for a,b in pairs),gained=sum(b and not a for a,b in pairs))
    deletions[folder]=d
out=dict(status='PASS',cases=len(r['rows']),hand_windows_verified=windows,selected_hand_windows=chosen_count,
    evidence_loss_proposals_vetoed=unknown_veto,totals=totals,incremental=inc,short_deletions=deletions,
    verification='Same Codex separate selection and saved paired arithmetic, not new detector or human ground truth.',
    cautions=['Removing flags may discard real short-motion information.','No new raw flags is by construction; paired incremental flags can still increase if the original baseline flag disappears.','Local offsets do not constitute a globally consistent alignment.'])
with (U/'VERIFIED.json').open('x',encoding='utf-8') as h:json.dump(out,h,ensure_ascii=False,indent=2)
print(json.dumps(out,ensure_ascii=False))
