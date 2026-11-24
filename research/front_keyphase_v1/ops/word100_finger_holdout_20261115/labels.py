from holdout_common import *
assert not(U/'LABELS_FROZEN.json').exists()
I=[1,0,0,0];E=[1,1,1,1];F=[0,0,0,0];THREE=[1,1,1,0];X=[-1]*4
labels={}
def put(person,word,indices,state):
 for i in indices:labels[person,word,i]=state
put(17,'WORD0003',range(2,9),I)
put(17,'WORD0071',range(3,7),E);put(17,'WORD0071',range(8,12),I)
put(17,'WORD0949',range(1,9),I)
put(17,'WORD2551',range(4,7),F);put(17,'WORD2551',range(7,10),[-1,1,1,1]);put(17,'WORD2551',[10],E)
put(17,'WORD1290',[5,6,11],E)
put(17,'WORD2700',range(4,7),I);put(17,'WORD2700',range(7,11),THREE)
put(17,'WORD2748',range(3,11),I)
put(17,'WORD2718',[1,2],THREE);put(17,'WORD2718',[3,4],F);put(17,'WORD2718',range(5,10),I)
put(18,'WORD0003',range(5,12),I)
put(18,'WORD0071',[2,3,4],E);put(18,'WORD0071',[5],[1,-1,-1,-1]);put(18,'WORD0071',range(6,12),I)
put(18,'WORD0949',range(4,12),I)
put(18,'WORD2551',[5,6,8],F);put(18,'WORD2551',[7,9],E)
put(18,'WORD1290',[3,4,10,11],E)
put(18,'WORD2700',[1,2,3,4],I);put(18,'WORD2700',range(5,10),THREE)
put(18,'WORD2748',range(2,11),I)
put(18,'WORD2718',[0,1],THREE);put(18,'WORD2718',[2,3,4],F);put(18,'WORD2718',range(5,11),I)
front={(17,'WORD0949',i)for i in[6,7,8]}|{(18,'WORD0949',i)for i in[9,10,11]}|{(17,'WORD2748',i)for i in[4,6]}|{(18,'WORD2748',6),(18,'WORD0071',7)}
rows=[]
for r in read(U/'CROP_INPUTS.json')['rows']:
 key=r['person'],r['word'],r['index'];y=labels.get(key,X)
 rows.append(dict(**r,y=y,view='front_pointing_index'if key in front else'other',review_image=str(U/'crops'/f"{r['clip_id']}.jpg"),review_image_sha256=sha(U/'crops'/f"{r['clip_id']}.jpg"),known_from='Direct public RGB with preceding/following source-time context; not word-name inferred'if any(v>=0 for v in y)else'Not assigned: rest/transition, hidden individual fingers or ambiguous image; not a negative label'))
write(U/'LABELS_FROZEN.json',dict(rows=rows,definition='index,middle,ring,little;1=visibly extended,0=visibly flexed,-1=not labeled/uncertain. Thumb not graded. Not contact truth, not handshape name, not a 3D angle target.',predictions_seen=False,reviewer='Codex inspected16 full hand crop sheets; not independent KSL expert',extraction_sha256=sha(U/'EXTRACTED.json'),crop_inputs_sha256=sha(U/'CROP_INPUTS.json'),scope='Full sequence inference; manually reviewed sparse source-time anchors. Unlabeled frames are not counted correct.',amendment='Numeral videos inspected as right-hand actions; corrected crop side before any finger prediction. Left-hand tests retained in 수어/화상/감사. No mirrored label is treated as new real left-hand evidence.'))
print({role:dict(anchors=sum(r['role']==role and any(v>=0 for v in r['y'])for r in rows),cells=sum(sum(v>=0 for v in r['y'])for r in rows if r['role']==role))for role in['diagnostic','confirmation_reserved']})
