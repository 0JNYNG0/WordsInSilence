from quality_common import *
# Explicit original-image review. Order index,middle,ring,little.
# quality1 = visually consistent chain,0 = clearly follows wrong finger/region,
# -1 = not decidable. phase1 = visible flexion/extension within +/-3frames,
# phase0 = stable finger posture even if hand translates/rotates. Not sign correctness.
G=[1,1,1,1];B=[1,1,1,0];S=[0,0,0,0];A=[1,1,1,0];I=[1,0,0,0];X=[-1]*4
spec={
 ('WORD2700',3):{1:(G,S),2:(G,[1,0,0,0]),3:(G,I),4:(G,[-1,1,1,0]),5:(G,A),6:(B,[-1,-1,-1,0]),7:(B,S),8:(B,S)},
 ('WORD2700',6):{3:(G,S),4:(G,[-1,0,0,0]),5:(G,I),6:(G,S),7:(G,A),8:(G,[-1,-1,-1,0]),9:(G,S)},
 ('WORD2700',9):{2:(G,S),3:(G,I),4:(G,[-1,0,0,0]),5:(G,A),6:(G,S),7:(G,S),8:(G,S)},
 ('WORD2718',3):{3:([1,-1,-1,-1],[-1,-1,-1,0]),4:(B,A),5:(B,A),6:(G,I),7:(G,I),8:(G,S),9:(G,S)},
 ('WORD2718',6):{3:(G,S),4:(G,S),5:(G,A),6:(G,[-1,-1,-1,0]),7:(G,I),8:(G,S),9:(G,S)},
 ('WORD2718',9):{3:(B,S),4:(B,A),5:(G,I),6:(G,S),7:(G,S),8:(G,S)}
}
rows=[]
for r in read(U/'fit_REVIEW_INPUTS.json')['rows']:
 v=spec[(r['word'],r['person'])].get(r['index'])
 if v:rows.append(dict(**r,quality=v[0],phase=v[1]))
write(U/'FIT_LABELS_NUMERALS.json',dict(rows=rows,reviewer='Codex direct RGB triplet and RTMW overlay inspection, no model predictions, not KSL expert',policy='Two nonexclusive targets; unknown masked. Gross chain consistency not calibrated anatomical precision. Sparse anchors only.'))
print(len(rows))
