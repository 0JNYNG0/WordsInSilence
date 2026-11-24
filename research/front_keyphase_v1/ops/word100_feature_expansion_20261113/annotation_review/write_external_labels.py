from prepare_ordinary import U,read,sha,write
from pathlib import Path
from PIL import Image
import collections
LABELS={
'WORD2632':{'fit':['UEFFF/UUUUU','UEFFF/UUUUU','UEFFF/UUUUU','UEFFF/UUUUU','UEFFF/UUUUU','UEFFF/UUUUU'],'development':['UEFFF/UUUUU','UEFFF/UUUUU','UEFFF/UUUUU','UEFFF/UUUUU','UEFFF/UUUUU','UEFFF/UUUUU']},
'WORD2634':{'fit':['UEEEF/UUUUU','UEEEF/UUUUU','UEEEF/UUUUU','UEEEF/UUUUU','UEEEF/UUUUU','UEEEF/UUUUU'],'development':['UEEEF/UUUUU','UEEEF/UUUUU','UEEEF/UUUUU','UEEEF/UUUUU','UEEEF/UUUUU','UEEEF/UUUUU']},
'WORD2636':{'fit':['EFFFF/UUUUU','EFFFF/UUUUU','EFFFF/UUUUU','EFFFF/UUUUU','EFFFF/UUUUU','EFFFF/UUUUU'],'development':['EFFFF/UUUUU','EFFFF/UUUUU','EFFFF/UUUUU','EFFFF/UUUUU','EFFFF/UUUUU','EFFFF/UUUUU']},
'WORD0099':{'fit':['UEFFF/UUUUU','UBBBU/UUUUU','UEFFF/UUUUU','UBBBB/UUUUU','UEFFF/UUUUU','UBBBB/UUUUU']}
}
NOTES={
'WORD2632':'원본 RGB에서 오른 검지가 길게 펴지고 중지·약지·소지는 손바닥 쪽으로 접힌 모습. 엄지는 손가락 위에 겹쳐 있어 관절 굽힘을 확정하지 않았다. 숫자 이름에서 정답을 만들지 않았다.',
'WORD2634':'원본 RGB에서 오른 검지·중지·약지가 펴지고 소지는 손바닥 쪽에 접힌 모습. 엄지는 소지와 겹쳐 굽힘/모음의 차이가 불분명하여 모름으로 두었다.',
'WORD2636':'원본 RGB에서 오른 엄지가 손바닥 밖으로 곧게 뻗고 네 긴 손가락의 접힌 마디가 보인다. 단어 숫자를 펼친 손가락 개수와 연결하지 않았다.',
'WORD0099':'첫 시점의 오른 검지 펼침과 뒤 시점의 오른 손가락이 반대 손등을 감싸며 굽는 모습을 직접 구별했다. 왼손은 가려져 개별 손가락을 확정하지 않았다. 의미 변이가 보류된 ID05는 사용하지 않았다.'}
def main():
    inp=read(U/'EXTERNAL_REVIEW_INPUTS.json')['rows'];mp={'E':0,'B':1,'F':2,'U':-1};out=[]
    for word,roles in LABELS.items():
        for role,vals in roles.items():
            rr=[r for r in inp if r['word']==word and r['role']==role];assert len(rr)==len(vals)==6
            for r,codes in zip(rr,vals):
                assert sha(r['image'])==r['image_sha256']
                for side,code in zip(['R','L'],codes.split('/')):
                    out.append(dict(**r,side=side,y=[mp[x] for x in code],states=list(code),known=sum(x!='U' for x in code),annotation_basis=NOTES[word],reviewer='Codex expansion_labels_review_1113 direct RGB review',review_status='agent_rgb_review_pending_root',derived_from_word_identity=False,derived_from_model=False))
    counts={role:dict(collections.Counter(s for r in out if r['role']==role for s in r['states'])) for role in ['fit','development']}
    write(U/'EXTERNAL_LABELS.json',dict(schema=dict(fingers=['thumb','index','middle','ring','little'],codes=mp),rows=out,counts=counts,frame_count=len(inp),scope='new outside100 physical-finger sparse labels; no whole-word feedback claim; DEV evaluation-only',selected_before_predictions=True))
    print(len(out),counts)
    allin=read(U/'ORDINARY_REVIEW_INPUTS.json')['rows']+inp
    specs=[
      [('WORD0003',1,43),('WORD0739',1,57),('WORD0739',4,120),('WORD1133',4,83),('WORD1290',1,60),('WORD1567',7,58)],
      [('WORD0003',12,42),('WORD0739',11,95),('WORD1133',12,69),('WORD1290',11,43),('WORD1567',12,57),('WORD1139',12,52)],
      [('WORD2523',1,59),('WORD2523',13,85),('WORD2551',2,56),('WORD2551',13,64),('WORD0099',1,79),('WORD0099',3,89)],
      [('WORD2632',1,54),('WORD2632',12,59),('WORD2634',4,51),('WORD2634',11,55),('WORD2636',7,35),('WORD2636',13,55)],
    ];review=[]
    for b,spec in enumerate(specs,1):
        board=Image.new('RGB',(1800,700),'white');selected=[]
        for j,(word,person,frame) in enumerate(spec):
            r=next(x for x in allin if x['word']==word and x['person']==person and x['frame']==frame)
            board.paste(Image.open(r['review']),((j%3)*600,(j//3)*350));selected.append(r)
        p=U/'panels'/f'ROOT_REVIEW_{b:02}.jpg';board.save(p,quality=97);review.append(dict(panel=str(p),frames=selected))
    write(U/'ROOT_REVIEW_INPUTS.json',dict(panels=review,frames=24,purpose='independent same-Codex spot-check before training; selected for state/hand/occlusion coverage, never candidate tuning'))
if __name__=='__main__':main()
