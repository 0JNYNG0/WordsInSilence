from apple_io import *
def main():
 rows={r['person']:r for r in read(U/'INPUTS.json')['rows']};truth={r['person']:r for r in read(U/'SOURCE_TRUTH.json')['rows']};alltiles=[]
 for person in[1,2,6,14,15]:
  r=rows[person];z=np.load(r['raw']);cap=cv2.VideoCapture(r['video']);ts=[]
  for fid in [truth[person]['pair_anchors'][0]-3,truth[person]['pair_anchors'][0],truth[person]['pair_anchors'][2]]:
   cap.set(cv2.CAP_PROP_POS_FRAMES,fid);ok,im=cap.read();assert ok;p=z['xy_px'][fid,91:133]
   lo=np.maximum(np.floor(p.min(0)-65),0).astype(int);hi=np.minimum(np.ceil(p.max(0)+65),[im.shape[1],im.shape[0]]).astype(int)
   crop=im[lo[1]:hi[1],lo[0]:hi[0]];scale=min(360/crop.shape[1],260/crop.shape[0]);im2=cv2.resize(crop,None,fx=scale,fy=scale);over=im2.copy();xy=np.rint((p-lo)*scale).astype(int)
   for offset,col in[(0,(0,140,255)),(21,(230,170,0))]:
    for k in[1,5,9,13,17]:
     for a,b in zip([0]+list(range(k,k+4)),list(range(k,k+4))):cv2.line(over,tuple(xy[offset+a]),tuple(xy[offset+b]),col,2)
   tile=np.full((300,740,3),255,np.uint8);tile[35:35+im2.shape[0],:im2.shape[1]]=im2;tile[35:35+over.shape[0],380:380+over.shape[1]]=over
   cv2.putText(tile,f'P{person:02} F{fid} L=orange R=cyan',(4,24),cv2.FONT_HERSHEY_SIMPLEX,.62,(20,30,40),1);ts.append(tile)
  cap.release();cv2.imwrite(str(U/'review'/f'overlap{person:02}.jpg'),np.vstack(ts))
 print('OVERLAP_REVIEW_READY')
if __name__=='__main__':main()
