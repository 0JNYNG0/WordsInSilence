from prepare import U,O,read,sha,write
rows=[];actions=[]
def point(w,p,f,h,shape=None,region=None):
 rows.append(dict(clip_id=f'NIA_SL_WORD{w}_REAL{p}_F',frame=f,hand=h,shape=shape,region=region))
for p,fs in [(17,[43,50,64,71,78]),(18,[78,84,97,104,110])]:
 for i,f in enumerate(fs):point('0739',p,f,'R','four_open'if i<2 else'four_closed','face_front'if i<2 else'upper_body')
 a,b,g,_,end=fs
 actions.extend([dict(clip_id=f'NIA_SL_WORD0739_REAL{p}_F',card='close_chest',frames=[b,(b+g)//2,g],span=[a,g],partial_end=b,goal_only=g),dict(clip_id=f'NIA_SL_WORD0739_REAL{p}_F',card='raise_forearm',frames=[g,(g+end)//2,end],span=[g,end])])
for p,fs in [(17,[41,50,54,58,62]),(18,[58,63,68,78])]:
 for f in fs:
  point('1133',p,f,'R','four_open','upper_body');point('1133',p,f,'L','four_closed','upper_body')
 actions.append(dict(clip_id=f'NIA_SL_WORD1133_REAL{p}_F',card='brush_arm',frames=fs[:3],span=[fs[0],fs[2]]))
for p,fs in [(17,[43,48,57,62]),(18,[47,59,65,77])]:
 for i,f in enumerate(fs):
  high='R'if i in [0,2]else'L'
  if p==17 and f==62:high='R'
  for h in ['L','R']:point('1567',p,f,h,'four_closed','face_front'if h==high else'upper_body')
 actions.append(dict(clip_id=f'NIA_SL_WORD1567_REAL{p}_F',card='alternating',frames=fs,span=[fs[0],fs[-1]]))
for p,fs in [(17,[33,38,46,54]),(18,[51,56,65,70])]:
 for i,f in enumerate(fs):
  for h in ['L','R']:point('2523',p,f,h,'four_open'if i<2 else None,'upper_body'if i>=2 or(p==17 and f==33)else None)
 actions.append(dict(clip_id=f'NIA_SL_WORD2523_REAL{p}_F',card='spread_hands',frames=[fs[1],(fs[1]+fs[2])//2,fs[2]],span=[fs[1],fs[2]]))
# Unknown labels are deliberately not expanded into guessed finger/contact labels.
write(U/'LABELS_FROZEN.json',dict(rows=rows,actions=actions,review='Codex visual inspection of original RGB uniform24-frame sheets and enlarged6-frame sheets, before feedback scores. Not independent human or KSL expert annotation.',
 fine_shape='Four non-thumb digits extended/folded; curled intermediate Egyptian fingertip pose left unlabeled. Thumb/contact excluded.',
 action_scope='Sparse visible source witnesses, not dense exact boundaries or proof of linguistic acceptability. Witness-frame middle is not a new posture label.',
 pins={str(p):sha(p)for p in(U/'review').glob('*.jpg')},no_training=True))
print(len(rows),'point records;',len(actions),'action witnesses frozen')
