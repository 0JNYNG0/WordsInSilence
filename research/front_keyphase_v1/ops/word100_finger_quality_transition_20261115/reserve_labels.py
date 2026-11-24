from quality_common import *
assert(U/'CANDIDATE_FROZEN.json').exists()and not(U/'confirmation_reserved_RESULTS_V2.json').exists()
G=[1]*4;S=[0]*4;F=[0]*4;I=[1,0,0,0];T=[1,1,1,0];E=[1]*4;A=[1,1,1,0];QI=[-1,1,1,1];QONLYI=[1,-1,-1,-1]
spec={
 'WORD2700':{1:(G,S,I),2:(G,S,I),3:(G,S,I),4:(G,I,F),5:(G,S,F),6:(G,[-1,0,0,0],F),7:(G,A,T),8:(G,S,T),9:(G,S,T),10:(G,[-1,-1,-1,0],T)},
 'WORD2718':{0:(G,S,T),1:(G,S,T),2:(G,A,T),3:(G,A,F),4:(G,A,F),5:(G,[-1,0,0,0],F),6:(G,I,I),7:(G,S,I),8:(G,S,I),9:(G,S,I),10:(G,S,I)},
 'WORD0071':{**{i:(G,S,E)for i in range(1,5)},5:(G,[0,1,1,1],I),**{i:(QONLYI,S,I)for i in range(6,12)}},
 'WORD0949':{**{i:(G,S,I)for i in range(2,7)},**{i:(QI,S,I)for i in range(7,10)}}}
rows=[]
for r in read(U/'confirmation_reserved_REVIEW_INPUTS.json')['rows']:
 v=spec[r['word']].get(r['index'])
 if v:rows.append(r|dict(quality=v[0],phase=v[1],posture=v[2]))
write(U/'CONFIRMATION_LABELS.json',dict(rows=rows,policy='Same sparse fixed source triplets +/-3; reviewed AFTER candidate freeze BEFORE any reserve predictions. Pose truth not sign correctness. Unclear chain = -1. No obvious wrong-finger chain observed at these selected reserve anchors; do not fabricate positives.',reviewer='Codex, not independent KSL expert',candidate_sha256=sha(U/'CANDIDATE_FROZEN.json')))
print('reserve anchors',len(rows))
