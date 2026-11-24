from form_candidate import *
import zipfile,io,av

def main():
 torch.set_num_threads(2);net=FormLesson();cid='NIA_SL_WORD0678_REAL11_F'
 r=next(r for r in read(O/'word100_rtmw_joint_learning_20261108/SOURCES.json')if r['clip_id']==cid)
 with zipfile.ZipFile(r['archive'])as z:video=z.read(r['member'])
 assert hashlib.sha256(video).hexdigest()==r['source_sha256']
 tiles=[]
 with av.open(io.BytesIO(video))as c:
  for i,f in enumerate(c.decode(video=0)):
   if i in[43,44,45,46,47]:
    im=f.to_ndarray(format='bgr24');im=im[:850,600:1350];im=cv2.resize(im,(375,425));cv2.putText(im,str(i),(10,25),0,.8,(255,255,255),2);tiles.append(im)
 cv2.imwrite(str(D/'audit/failure0678.jpg'),np.hstack(tiles))
 paths=[(cid,read(O/'word100_rtmw_joint_learning_20261108/records'/f'{cid}.json')['raw_path'],list(range(41,49)))]
 paths.extend((r['id'],r['raw'],[48,49,51,57,60,61,63,65,69])for r in read(H/'PUBLIC_INPUTS.json')['rows']if r['id']=='NIA_SL_WORD1290_REAL07_F')
 for cid,path,ids in paths:
  raw=dict(np.load(path));ch=adapter.convert(raw['xy_px'],raw['score'],raw['timestamp_us'],raw['frame_id'],int(raw['source_width']),int(raw['source_height']));q=net.shape.sequence(ch,'R');s,recovered,form=net.form_states(ch,'R');plane=net.plane.sequence(ch,'R')
  print(cid,flush=True)
  for i in ids:print(i,'old',q['states'][i],'form',s[i],'fingers',q['fingers'][i].tolist(),'plane',plane[i],flush=True)
if __name__=='__main__':main()
