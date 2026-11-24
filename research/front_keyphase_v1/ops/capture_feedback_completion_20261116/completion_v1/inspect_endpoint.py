from visible_lesson import *
sys.path.insert(0,str(V.parent));from diagnose_terminal import sources
net=VisibleLesson();torch.set_num_threads(2)
for r in sources():
 z=dict(np.load(r['raw_path']));ch=adapter.load(r['raw_path']);f,q,_=net.feature(ch);s,_=net.relation(f,q,ch['timestamp_us']);m=net.motion.sequence(ch);events=bridge(s,m)
 from target_region import region_distance
 d=region_distance(ch);p=net.plane.sequence(ch,'R');sh=net.shape.sequence(ch,'R',p)
 print('PERSON',r['person'])
 for e in events:
  ids=range(max(0,e['end']-3),min(len(s),e['end']+7))
  print(e,[(i,int(s[i]),int(m[i]),round(float(d[i]),2),int(sh['states'][i]))for i in ids])
