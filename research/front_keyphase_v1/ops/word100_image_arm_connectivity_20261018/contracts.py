from pathlib import Path
import json
import numpy as np,cv2
from candidate import *
U=Path(__file__).resolve().parent

def main():
    tests=[]
    def canvas(color=(100,150,190)):
        im=np.full((180,240,3),(160,45,15),np.uint8)
        cv2.line(im,(55,40),(55,145),color,15);cv2.line(im,(185,40),(185,145),color,15)
        return im
    e=[(55,40),(185,40)];wr=[(55,145),(185,145)]
    def test(name,im,el,ws,expected):
        o=pixel_evidence(im,el,ws,130.);assert o['rgb_side']==expected,(name,o['rgb_side'],o['pixel_reason']);tests.append(name)
    test('separate arms connected',canvas(),e,wr,[0,1])
    test('different local arm color',canvas((25,50,80)),e,wr,[0,1])
    test('raw order reversed',canvas(),e,wr[::-1],[1,0])
    test('unseen wrist not fabricated',canvas(),e,[None,wr[1]],[-1,1])
    test('same hand duplicated retains uncertainty',canvas(),e,[wr[0],(56,142)],[-1,-1])
    im=canvas();cv2.line(im,(55,100),(185,100),(100,150,190),15)
    test('touching arms ambiguous',im,e,wr,[-1,-1])
    test('uniform image no anatomical evidence',np.full((180,240,3),110,np.uint8),e,wr,[-1,-1])
    test('grayscale no color separation',cv2.cvtColor(cv2.cvtColor(canvas(),cv2.COLOR_BGR2GRAY),cv2.COLOR_GRAY2BGR),e,wr,[-1,-1])
    im=canvas();im[85:100,40:70]=(160,45,15)
    test('actual gap not filled',im,e,wr,[-1,1])
    m=np.zeros((30,30),bool);m[2:26,2]=1;m[25,2:26]=1;m[2:26,25]=1
    assert not shortest(m,(2,2),(25,2));tests.append('long detour not arm connection')
    frame=canvas();pose=np.zeros((33,2));pose[11]=[55,10];pose[12]=[185,10];pose[13]=e[0];pose[14]=e[1];usable=np.zeros(33,bool);usable[[11,12,13,14]]=True;hand=np.zeros((2,21,2));hand[0]=wr[0];hand[1]=wr[1];mask=np.ones((2,21),bool)
    a=propose(frame,240,180,pose,usable,hand,mask,[1,0],[-1,-1]);assert a['sides']==[1,0];tests.append('occupied destinations not overwritten')
    a=propose(frame,240,180,pose,usable,hand[:1],mask[:1],[1],[1]);assert a['sides']==[1];tests.append('confirmed side retained even if contrary RGB')
    a=propose(frame,240,180,pose,usable,hand[:1],mask[:1],[-1],[-1]);assert a['sides']==[-1];tests.append('withheld side not released')
    full=np.zeros((240,240,3),np.uint8);full[:180]=frame;alter=full.copy();alter[180:]=255
    a=propose(full,240,180,pose,usable,hand[:1],mask[:1],[1],[-1]);b=propose(alter,240,180,pose,usable,hand[:1],mask[:1],[1],[-1]);assert a['sides']==b['sides'] and np.array_equal(a['support'],b['support']);tests.append('outside sensor pixels cannot affect evidence')
    for x in [full,pose,usable,hand,mask]:x.flags.writeable=False
    propose(full,240,180,pose,usable,hand[:1],mask[:1],[1],[-1]);tests.append('all original arrays read only')
    (U/'CONTRACTS.json').write_text(json.dumps({'status':'PASS','count':len(tests),'tests':tests},ensure_ascii=False,indent=2),encoding='utf8');print('PASS',len(tests))
if __name__=='__main__':main()
