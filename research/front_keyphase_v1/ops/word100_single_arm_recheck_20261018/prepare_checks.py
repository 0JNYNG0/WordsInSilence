"""Copy separate verifiers from the frozen parent, adapting one explicit predicate."""
from pathlib import Path
U=Path(__file__).resolve().parent;T=U.parent/'word100_track_revalidation_20261018'
s=(T/'verify.py').read_text(encoding='utf-8')
assert s.count("and all(old['arms_visible'][f])")==1
s=s.replace("and all(old['arms_visible'][f])","and bool(old['arms_visible'][f,want])")
(U/'verify.py').write_text(s,encoding='utf-8')
s=(T/'integration_checks.py').read_text(encoding='utf-8')
s=s.replace("ids={x['clip_id'] for x in read(P/'ARM_CONFLICTS.json')}","ids={x['clip_id'] for x in read(U/'DELTA_PREVIOUS.json')} | {x['clip_id'] for x in read(P/'ARM_CONFLICTS.json')}")
(U/'integration_checks.py').write_text(s,encoding='utf-8')
s=(T/'render_evidence.py').read_text(encoding='utf-8').replace("changes=read(U/'CHANGES.json')","changes=read(U/'DELTA_PREVIOUS.json')").replace("for q in (A,P)","for q in (A,P,T)").replace("old=load(P/'results'","old=load(T/'results'")
(U/'render_evidence.py').write_text(s,encoding='utf-8')
print('checks prepared before freeze')
