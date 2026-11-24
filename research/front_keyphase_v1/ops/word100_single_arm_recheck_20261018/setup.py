"""Prepare one isolated candidate; never edits a prior unit or production."""
import json,hashlib,shutil
from pathlib import Path
from datetime import datetime,timezone,timedelta
U=Path(__file__).resolve().parent;O=U.parent;T=O/'word100_track_revalidation_20261018'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def main():
    assert not (U/'CONTEXT.json').exists() and not (T/'RUNNING.json').exists()
    assert read(T/'FINAL_AUDIT.json')['status']=='PASS'
    s=(T/'candidate.py').read_text(encoding='utf-8')
    assert s.count("if s<0 or not all(g['arms_visible']):return -1")==1
    s=s.replace("if s<0 or not all(g['arms_visible']):return -1","if s<0 or not g['arms_visible'][s]:return -1")
    s=s.replace('Research-only causal recheck of a provisional anatomical identity.',
        'Research-only single-visible-arm extension of the frozen causal recheck.\nOnly the currently supporting arm must be visible; missing opposite arm is\nunknown, not negative evidence. No other gate or temporal rule is changed.')
    (U/'candidate.py').write_text(s,encoding='utf-8')
    s=(T/'contracts.py').read_text(encoding='utf-8')
    s=s.replace("m=Revalidator();out,_=m.step(h,[.9],0,[1],['provisional'],geo(arms=False));ok('elbow only cannot change',out==[1])", """m=Revalidator();out,_=m.step(h,[.9],0,[1],['provisional'],geo(arms=False));ok('elbow only cannot change',out==[1])
    one=geo();one['arms_visible']=[True,False];one['arm_cost'][0,1]=np.inf
    m=Revalidator();out,_=m.step(h,[.9],0,[1],['provisional'],one);ok('single observed arm first conflict withheld',out==[None])
    out,_=m.step(h+1,[.9],33333,[1],['provisional'],one);ok('single observed arm second independent agreement',out==[0])
    m=Revalidator();out,_=m.step(h,[.9],0,[0],['provisional'],one);ok('single arm matching normal label preserved',out==[0])
    unknown=geo();unknown['arms_visible']=[False,True]
    m=Revalidator();out,_=m.step(h,[.9],0,[1],['provisional'],unknown);ok('unobserved supporting arm rejected',out==[1])""")
    (U/'contracts.py').write_text(s,encoding='utf-8')
    write(U/'PROTOCOL.json',{
        'at':datetime.now(timezone(timedelta(hours=9))).isoformat(),
        'authorization':'User asked to continue remaining errors. Public-only isolated assignment repair and validation. No training until joint review.',
        'single_change':'A current observed shoulder-elbow-wrist chain supporting the hand can trigger provisional identity recheck even if the opposite chain is unusable. Keep the previous geometric gates, competing-candidate check, two-frame confirmation, physical association,100ms evidence lifetime, confirmed/unassigned protections.',
        'rationale':'Missing opposite wrist is unknown. It need not erase current positive evidence from the visible arm. One observation only withholds; two agreeing physical observations needed to change side.',
        'known_development_data':'Remaining original10 errors plus additional airport1 informed hypothesis. These are TRAIN audit cases, not an independent test set.',
        'scope':'PublicTRAIN899 videos times2 streams. Reuse frozen raw observations. Compare all226362frames/305602candidates to preceding candidate; preserve old4changes/441normal/315crossing; inspect all new changes in RGB and context.',
        'acceptance':'No established normal or crossing regression. All new changed scenes visually accounted for. Independent scalar reconstruction plus actual wrapper/order/prefix checks. If new ambiguous or wrong changes occur report failed/not adopted without tuning thresholds to results.',
        'remaining_audit':'Inspect all11 remaining frames: first-entry weak wrist confidence, partial hands, duplicate ambiguity. No lowering visibility threshold, forcing screen-half labels, reconstructing missing fingers, deleting all provisional hands, or word-specific exceptions.',
        'prohibited':['training','classifier inference','user/mentor/DEV/heldout reads','C/web/DTW changes','new detector extraction','camera','download','external transfer','Git','post-result coefficient search'],
        'roles':'Codex implementation, source-video review and separate verifier. No independent human ground truth or independent agent claimed.'})
    print('isolated candidate +23 contracts prepared; not run')
if __name__=='__main__':main()
