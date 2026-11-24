from pathlib import Path
import sys,json,time,hashlib,datetime
J=Path(__file__).resolve().parent;G=J.parent;E=G/'public_motion_decoupling_20261116';O=G.parent
sys.path.insert(0,str(E))
from palm_geometry import PalmMotionLesson
from common import np,torch,adapter,read,write,channels,sources as old_sources,V,H,B,D

def sources():return old_sources()+read(E/'confirmation/INPUTS.json')['rows']+read(E/'unselected_12/INPUTS.json')['rows']
def baseline(cid):
 for folder in ['palm_checks','palm_views','unselected_12']:
  p=E/folder/(cid+'.json')
  if p.exists():return read(p)['after']
 raise KeyError(cid)

def contract():
 write(J/'CONTRACT.json',dict(at=datetime.datetime.now().astimezone().isoformat(),goal='Correct remaining view-related false warnings and unsupported observations without losing frontal normals, genuine errors, or speed.',graph=['A native RGB + landmark + head diagnosis','B source-reviewed public FIT augmentation only where justified','C fixed learning and known normal/error/partial controls','D unselected public views and isolated web/latency verification','E admit, preserve old pins, record limits'],loop='Continue cause-based repairs, no arbitrary threshold/word/person exceptions or unlimited repeated tests.',private_data=False,frozen_C_T3F=True,unchanged100=True,gate='No new normal errors, no lost correct controls, preserve known real distinction checks; unresolved occluded geometry remains unknown, never filled as truth.',baseline=str(E/'RESULT.json')))

if __name__=='__main__':contract()
