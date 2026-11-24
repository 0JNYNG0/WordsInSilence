from quality_common import *
G=[1]*4;S=[0]*4;B=[1,1,1,0];I=[1,0,0,0];A=[1,1,1,0];M=[0,1,1,0];QL=[1,1,1,-1];QI=[-1,1,1,1]
spec={
 ('WORD2700',1):{3:(G,S),4:(G,I),5:(G,I),6:(G,M),7:(B,S),8:(B,S),9:(B,S)},
 ('WORD2700',4):{4:(G,S),5:(G,I),6:(G,S),7:(G,A),8:(G,M),9:(B,S),10:(B,S)},
 ('WORD2700',8):{2:(G,S),3:(G,S),4:(G,A),5:(B,M),6:(B,S),7:(B,S)},
 ('WORD2718',1):{3:(QL,S),4:(G,A),5:(QL,S),6:(G,I),7:(G,S),8:(G,S),9:(G,S),10:(G,S)},
 ('WORD2718',4):{6:(G,A),7:(G,S),8:(G,I),9:(G,S),10:(G,S)},
 ('WORD2718',8):{4:(QL,[-1,-1,-1,0]),5:(B,A),6:(G,I),7:(G,S),8:(G,S),9:(G,S)},
 ('WORD0949',3):{**{i:(G,S)for i in range(2,7)},**{i:(QI,S)for i in range(7,10)}},
 ('WORD0949',6):{**{i:(G,S)for i in range(3,8)},**{i:(QI,S)for i in range(8,11)}},
 ('WORD0949',9):{5:(G,[-1,0,0,0]),6:(G,S),7:(G,S),8:(QI,S),9:(QI,S)}
}
rows=read(U/'FIT_LABELS.json')['rows'];added=[]
for role in ['fit_more','fit_front']:
 for r in read(U/f'{role}_REVIEW_INPUTS.json')['rows']:
  v=spec[r['word'],r['person']].get(r['index'])
  if v:added.append(r|dict(quality=v[0],phase=v[1],review_note='Source RGB +/-3 frames and overlay viewed; no current model probabilities; unclear chain/phase masked.'))
write(U/'FIT_LABELS_V2.json',dict(rows=rows+added,added=len(added),reviewer='Codex visual review, not independent expert',label_policy=read(U/'FIT_LABELS.json')['policy']))
print('anchors',len(rows)+len(added),'added',len(added),'clips',len(set(r['clip_id']for r in rows+added)))
