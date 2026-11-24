from quality_common import *
G=[1]*4;S=[0]*4;X=[-1]*4;I=[1,0,0,0];T=[1,1,1,0];F=[0]*4;E=[1]*4;B=[1,1,1,0];A=[1,1,1,0]
left={3:{2:(G,S),3:(G,S),4:(G,S),5:(G,S),6:(G,[0,1,1,1]),7:(G,S),8:(G,S),9:(G,S)},6:{0:(G,S),1:(X,[0,1,1,1]),2:(G,[0,1,1,1]),3:(G,S),4:(G,S),5:(G,S),6:(G,S),7:(G,S),8:(G,S),9:(G,S),10:(G,S)},9:{7:(G,S),8:(G,S),9:(G,[0,1,1,1]),10:(G,S),11:(G,S)}}
rows=[]
for r in read(U/'fit_left_REVIEW_INPUTS.json')['rows']:
 v=left[r['person']].get(r['index'])
 if v:rows.append(dict(**r,quality=v[0],phase=v[1]))
write(U/'FIT_LABELS.json',dict(rows=read(U/'FIT_LABELS_NUMERALS.json')['rows']+rows,policy=read(U/'FIT_LABELS_NUMERALS.json')['policy'],reviewer='Codex RGB+overlay, not independent expert'))
spec={
 'WORD2700':{1:(G,S,I),2:(G,S,I),3:(G,S,I),4:(G,I,F),5:(G,A,F),6:(B,[0,1,1,0],T),7:(B,S,T),8:(B,S,T),9:(B,S,T)},
 'WORD2718':{1:(B,S,T),2:(B,I,[-1,1,1,0]),3:(B,A,F),4:([1,1,1,-1],I,F),5:(G,I,I),6:(G,S,I),7:(G,S,I),8:(G,S,I),9:(G,S,I)},
 'WORD0071':{0:(G,S,E),1:(G,S,E),2:(G,S,E),3:(G,S,E),4:(G,[0,1,1,1],I),5:(G,S,I),6:(G,S,I),7:(G,S,I),8:([1,-1,-1,-1],S,I),9:([1,-1,-1,-1],S,I)},
 'WORD0949':{3:(G,S,I),4:(G,S,I),5:(G,S,I),6:(G,S,I),7:([-1,1,1,1],S,I),8:([-1,1,1,1],S,I),9:([-1,1,1,1],S,I),10:([-1,1,1,1],S,I)}
}
rr=[]
for r in read(U/'diagnostic_REVIEW_INPUTS.json')['rows']:
 v=spec[r['word']].get(r['index'])
 if v:rr.append(dict(**r,quality=v[0],phase=v[1],posture=v[2]))
write(U/'DIAGNOSTIC_LABELS.json',dict(rows=rr,policy='Source RGB triplets and center overlay only, before all new/current observer predictions; phase is change within +/-3sourceframes, not semantic transition/nonessential judgment; ambiguous masked',reviewer='Codex, not KSL expert'))
print('fit',len(read(U/'FIT_LABELS.json')['rows']),'diagnostic',len(rr))
