"""Source-reviewed labels only, never imported by inference.

F/L/R crops and front overlays were inspected before candidate inference.
0=clearly flexed,1=clearly extended,-1=intermediate/occluded/uncertain.
The change target is visible posture variation, NOT arm rotation or movement.
"""
from common import *
assert not(U/'LABELS.json').exists()
src={r['clip_id']:r for r in read(U/'READY_SOURCES.json')['rows']};rr=read(U/'DENSE_REVIEW.json')['rows'];out=[]
def ys(s):return [{'u':-1,'0':0,'1':1}[c]for c in s]
shape={
 ('WORD2551',2):dict(zip([42,43,44,45,46,54,55,56,57,58,68,69,70,71,72],['0000','0000','0000','u000','u000','0000','0000','uuuu','uuuu','uuuu','0000','0000','uuuu','1111','1111'])),
 ('WORD2551',5):dict(zip([51,52,53,54,55,68,69,70,71,72,80,81,82,83,84],['0000','0000','0000','u000','u000','1111','1111','uuuu','000u','0000','1111','1111','1111','1111','1111'])),
 ('WORD2551',7):dict(zip([25,26,27,28,29,41,42,43,44,45,52,53,54,55,56],['0000','0000','0000','u000','u000','0000','uuuu','uuuu','uuuu','uuuu','000u','uuuu','u111','u111','u111'])),
 ('WORD2551',10):dict(zip([46,47,48,49,50,61,62,63,64,65,80,81,82,83,84],['u000','u000','u000','0000','0000','0000','0000','000u','uuuu','uuuu','00uu','00uu','00uu','00uu','00uu'])),
 ('WORD2551',16):dict(zip([36,37,38,39,40,49,50,51,52,53,65,66,67,68,69],['0000','u000','u000','0000','0000','0000','uuuu','uuuu','uuuu','uuuu','000u','uuuu','1111','1111','1111']))}
index={('WORD2748',2):dict(zip(range(47,62),'111u00000uu1111')),('WORD2748',5):dict(zip(range(55,70),'111u00000000011')),('WORD2748',7):dict(zip(range(50,65),'11u00000000u111')),('WORD2748',11):dict(zip(range(63,78),'11u00000000000u')),('WORD2748',12):dict(zip(range(50,65),'u000000000000u1'))}
phase={('WORD2551',2):{44:'0000',56:'1111',70:'1111'},('WORD2551',5):{53:'0000',70:'1111',82:'0000'},('WORD2551',7):{27:'0000',43:'1111',54:'1111'},('WORD2551',10):{48:'0000',63:'1111',82:'0000'},('WORD2551',16):{38:'0000',51:'1111',67:'1111'},('WORD2748',2):{49:'1000',51:'1000',53:'0000',55:'1000',57:'1000',59:'1000'},('WORD2748',5):{57:'1000',59:'1000',61:'0000',63:'0000',65:'0000',67:'1000'},('WORD2748',7):{52:'1000',54:'1000',56:'0000',58:'0000',60:'1000',62:'1000'},('WORD2748',11):{65:'1000',67:'1000',69:'0000',71:'0000',73:'0000',75:'1000'},('WORD2748',12):{52:'1000',54:'0000',56:'0000',58:'0000',60:'0000',62:'1000'}}
for a in rr:
 r=src[a['clip_id']];key=r['word'],r['person'];f=a['frame'];hand=a['hand']
 y='1111'if r['word']=='WORD1290'else(index[key][f]+'000'if r['word']=='WORD2748'else shape[key][f])
 ph='0000'if r['word']=='WORD1290'else phase[key].get(f,'uuuu')
 # Quality: own chain identity, not exact 3D correctness. Hidden folded endpoints are not certified.
 q=[1,1,1,1]
 if r['word']=='WORD2748':q=[1,-1,-1,-1]
 if r['word']=='WORD2551'and f<min(phase[key])+5:q=[1,-1,-1,-1]
 if r['word']=='WORD2551'and r['person']==7:q[3]=-1
 if r['word']=='WORD1290'and hand=='L':q=[-1]*4
 if r['word']=='WORD1290'and hand=='L'and r['person']==15 and f==46:q=[1,1,1,0] # little chain follows thumb, source palm-down fully open
 brief=[False]*4
 if key==('WORD2551',5)and f in [68,69]:brief=[True]*4
 out.append(r|a|dict(shape=ys(y),phase=ys(ph),quality=q,brief=brief,source_sha256=r['views']['F']['sha256'],origin='Codex source RGB multi-view/adjacent-frame audit before model prediction'))
write(U/'LABELS.json',dict(rows=out,legend=dict(shape='1clearlyextended,0clearlyflexed,-1intermediate/uncertain',phase='1actualposturechange in reviewed neighbors,0posturemaintained,-1notannotated',quality='1ownchainconsistent,0confirmedwrongfingertrack,-1unclear',brief='Source-reviewed short open plateau; timing separately audited, not semantic importance'),notes=['No model predictions/gloss-imputed GT. No interpolated state for intermediate frames. Labels are Codex research annotation, not expert-certified.','New dense examples focus right-hand transitions; existing FIT includes both hands. New left evaluation is open-hand control, not evidence of all left transitions.','A short shape observation alone does not imply pedagogical error or a new feedback sentence.'],review_status='PRE_PREDICTION_AUDITED'))
print({role:dict(anchors=sum(r['role']==role for r in out),shape_known=sum(sum(v>=0 for v in r['shape'])for r in out if r['role']==role),phase_known=sum(sum(v>=0 for v in r['phase'])for r in out if r['role']==role),bad_chains=sum(r['quality'].count(0)for r in out if r['role']==role))for role in ['fit','diagnostic','confirmation']})
