from prepare_review import *
selected={
'WORD2523_REAL01':[31,49,55],'WORD2523_REAL04':[73,93,102],'WORD2523_REAL07':[39,51,57],
'WORD1140_REAL01':[50,69,78],'WORD1140_REAL02':[45,59,73],'WORD1140_REAL03':[55,64,81],
'WORD0196_REAL01':[50,59,77],'WORD0196_REAL02':[44,54,84],'WORD0196_REAL03':[75,87,111],
'WORD2551_REAL01':[37,44,52],'WORD2551_REAL02':[54,62,79],'WORD2551_REAL03':[61,69,77]}
tasks=[dict(clip_id='NIA_SL_'+k+'_F',frames=v) for k,v in selected.items()]
wr(U/'ANCHOR_SELECTION.json',dict(tasks=tasks,selection='Direct RGB-only FIT screening; three native anchors per clip. Contrasts: bilateral gathered/extended fingers, bilateral moving curved hands, bilateral fists and right F/B/E opening. Not educational error labels. No new DEV or model prediction used.',locked_before_new_model_outputs=True))
allrows=rd(U/'SCREEN_INPUTS.json')['rows']+rd(U/'FOLLOWUP_INPUTS.json')['rows'];bykey={(r['clip_id'],r['frame']):r for r in allrows}
rows=[bykey[(t['clip_id'],f)] for t in tasks for f in t['frames']]
wr(U/'ANCHOR_INPUTS.json',dict(rows=rows,complete=True))
edges=[]
for s in [1,5,9,13,17]:edges.extend([(0,s),(s,s+1),(s+1,s+2),(s+2,s+3)])
for t in tasks:
 rr=[r for r in rows if r['clip_id']==t['clip_id']];board=Image.new('RGB',(1120,len(rr)*380),'white')
 for j,r in enumerate(rr):
  im=Image.open(r['image']);z=np.load(r['raw_path']);i=int(np.flatnonzero(z['frame_id']==r['frame'])[0])
  cell=Image.new('RGB',(1120,380),'white');d=ImageDraw.Draw(cell);d.text((8,3),f"{r['word']} ID{r['person']:02} f{r['frame']}   RGB / RTMW     R -> L",font=F,fill='#17283f')
  for h,(side,sl) in enumerate([('R',slice(112,133)),('L',slice(91,112))]):
   box=r['crop_boxes'][side];cut=im.crop(box);xy=z['xy_px'][i,sl]-np.array(box[:2]);sc=z['score'][i,sl];ov=cut.copy();dr=ImageDraw.Draw(ov)
   for a,b in edges:
    if sc[a]>.2 and sc[b]>.2 and np.isfinite(xy[[a,b]]).all():dr.line([tuple(xy[a]),tuple(xy[b])],fill='#ffb74d',width=2)
   for k,(x,y) in enumerate(xy):
    if sc[k]>.2 and np.isfinite([x,y]).all():dr.ellipse([x-2,y-2,x+2,y+2],fill='#1b65e0')
   for v,pic in enumerate([cut,ov]):
    pic.thumbnail((270,330));cell.paste(pic,(h*560+v*280+(270-pic.width)//2,45+(330-pic.height)//2))
  board.paste(cell,(0,j*380))
 board.save(U/'panels'/f"{t['clip_id']}_geometry.jpg",quality=98)
print('selected',len(rows),'native frames',len(tasks),'clips')

