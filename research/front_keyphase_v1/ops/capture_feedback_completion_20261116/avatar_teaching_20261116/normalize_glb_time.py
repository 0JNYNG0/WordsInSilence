from pathlib import Path
import struct,json,shutil
import numpy as np
def normalize(path):
 path=Path(path);b=path.read_bytes();jl=struct.unpack_from('<I',b,12)[0];j=json.loads(b[20:20+jl]);bl,kind=struct.unpack_from('<II',b,20+jl);binary=bytearray(b[28+jl:28+jl+bl]);inputs={s['input']for a in j.get('animations',[])for s in a['samplers']};arrays=[]
 for i in inputs:
  a=j['accessors'][i];v=j['bufferViews'][a['bufferView']];assert a['componentType']==5126 and a['type']=='SCALAR';x=np.ndarray((a['count'],),dtype='<f4',buffer=binary,offset=v.get('byteOffset',0)+a.get('byteOffset',0),strides=(v.get('byteStride',4),));arrays.append((a,x))
 first=min(float(x[0])for a,x in arrays)
 if first<1e-6:return 0
 backup=path.with_name('motion_blender_frame1.glb')
 if not backup.exists():shutil.copy2(path,backup)
 for a,x in arrays:x-=first;a['min']=[float(x.min())];a['max']=[float(x.max())]
 jb=json.dumps(j,separators=(',',':')).encode('utf-8');jb+=b' '*((-len(jb))%4);out=struct.pack('<III',0x46546c67,2,28+len(jb)+len(binary))+struct.pack('<II',len(jb),0x4e4f534a)+jb+struct.pack('<II',len(binary),kind)+binary;path.write_bytes(out);return first
if __name__=='__main__':
 U=Path(__file__).resolve().parent;report={}
 for clip,rev in [('NIA_SL_WORD0099_REAL01','guided_v2'),('NIA_SL_WORD1290_REAL01','baseline')]:report[clip]=normalize(U/clip/rev/'motion.glb')
 (U/'TIME_ORIGIN_CORRECTION.json').write_text(json.dumps(dict(reason='Blender frame1 export began at 1/30 second; shifted animation timestamps to public native frame0, without changing any pose.',offset_seconds=report),indent=2),'utf-8');print(report)
