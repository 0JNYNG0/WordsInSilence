from pathlib import Path
import json,hashlib,collections
U=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text('utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n','utf-8')
# Explicit RGB review results in frozen panel order: 3 recorded IDs, 2 frames each.
# No word-to-shape rule or model prediction is used to generate these entries.
LABELS={
'WORD0003':{
'fit':['UEFFF/UEFFF','UEFFF/UEFFF','UEFFF/UEFFF','UEFFF/UEFFF','UEFFF/UEFFF','UEFFF/UEFFF'],
'development':['UEFFF/UEFFF','UEFFF/UEFFF','UBFFF/UEFFF','UBFFF/UEFFF','UEFFF/UEFFF','UEFFF/UEFFF']},
'WORD0739':{
'fit':['EEEEE/UUUUU','UFFFF/UUUUU','EEEEE/UUUUU','UFFFF/UUUUU','EEEEE/UUUUU','UFFFF/UUUUU'],
'development':['EEEEE/UUUUU','UFFFF/UUUUU','EEEEE/UUUUU','UFFFF/UUUUU','EEEEE/UUUUU','UFFFF/UUUUU']},
'WORD1133':{
'fit':['EEEEE/UFFFF','EEEEE/UFFFF','UEEUU/UFFFF','EEEEE/UFFFF','UEEEE/UFFFF','EEEEE/UFFFF'],
'development':['EEEEE/UFFFF','EEEEE/UFFFF','UEEEE/UFFFF','UEEEE/UFFFF','EEEEE/UFFFF','EEEEE/UFFFF']},
'WORD1290':{
'fit':['EEEEE/UUUUU','EEEEE/UUUUU','EEEEE/UUUUU','EEEEE/UUUUU','EEEEE/UUUUU','EEEEE/UUUUU'],
'development':['UEEEE/UEEEE','EEEEE/UUUUU','EEEEE/UUUUU','EEEEE/UUUUU','EEEEE/UUUUU','EEEEE/UUUUU']},
'WORD1567':{
'fit':['EFFFF/EFFFF','EFFFF/EFFFF','EFFFF/EFFFF','EFFUU/EFFFF','EFFFF/EFFFF','EUUUU/EFFFF'],
'development':['UUUUU/UFFFF','EFFFF/EFFFF','UUUUU/EFFFF','UUUUU/EFFFF','EFFFF/UFFFF','EFFFF/EFFFF']},
'WORD2523':{
'fit':['EEUUU/EEUUU','UBUUU/UBUUU','EEUUU/EEUUU','UBUUU/UBUUU','EEUUU/EEUUU','UBUUU/UBUUU'],
'development':['EEUUU/EEUUU','UBUUU/UBUUU','EEUUU/EEUUU','UBUUU/UBUUU','EEUUU/EEUUU','UBBBB/UBBBB']},
'WORD0071':{
'fit':['EEEEE/EEEEE','UEFFF/UEFFF','EEEEE/EEEEE','UEFFF/UEFFF','EEEEE/EEEEE','UEFFF/UEFFF'],
'development':['EEEEE/EEEEE','UEFFF/UEFFF','UEFFF/UEFFF','UEFFF/UEFFF','EEEEE/UUUUU','UEFFF/UEFFF']},
'WORD0500':{
'fit':['UEFFF/UUFFF','UEFFF/UUFFF','UEFFF/UUFFF','UEFFF/UUFFF','UEFFF/UUFFF','UEFFF/UUFFF'],
'development':['UUUUU/UUUUU','UEFFF/UUFFF','UEFFF/UUFFF','UEFFF/UUFFF','UEFFF/UUUUU','UEFFF/UUFFF']},
'WORD0678':{
'fit':['EEFFF/UUUUU','EEFFF/UUUUU','EEFFF/UUUUU','EEFFF/UUUUU','EEFFF/UUUUU','EEFFF/UUUUU'],
'development':['UBBBB/UUUUU','EEFFF/UUUUU','UBBBU/UUUUU','EEFFF/UUUUU','UUUUU/UUUUU','EEFFF/UUUUU']},
'WORD0949':{
'fit':['UEFFF/UUUUU','UUFFF/UUUUU','UEFFF/UUUUU','UUFFF/UUUUU','UEFFF/UUUUU','UUFFF/UUUUU'],
'development':['UUFFF/UUUUU','UUFFF/UUUUU','UEFFF/UUUUU','UUFFF/UUUUU','UUUUU/UUUUU','UUFFF/UUUUU']},
'WORD1139':{
'fit':['UEFFF/EUUUU','UEFFF/EUUUU','UUFFF/EUUUU','UEFFF/EUUUU','UEFFF/EUUUU','UEFFF/EUUUU'],
'development':['UEFFF/EEEEE','UUFFF/EUUUU','UBFFF/EUUUU','UUFFF/UUUUU','UUFFF/EUUUU','UUFFF/EUUUU']},
'WORD2551':{
'fit':['UEEEE/UUUUU','UEEEE/UUUUU','UBBBB/UUUUU','UBEEE/UUUUU','UBBBB/UUUUU','UEEEE/UUUUU'],
'development':['UUUUU/UUUUU','UBEEE/UUUUU','UUUUU/UUUUU','UEEEE/UUUUU','UBBBB/UUUUU','UEEEE/UUUUU']}
}
NOTES={
'WORD0003':'검지의 길게 편 윤곽과 나머지 긴 손가락의 접힌 마디를 구별했다. DEV12 오른 검지는 일부 굽은 시점이다. 엄지는 겹쳐 확정하지 않았다.',
'WORD0739':'활동 오른손의 펼친 손가락 또는 주먹에 접힌 긴 손가락을 직접 확인했다. 아래의 비활동 왼손은 정답 확대를 하지 않았다.',
'WORD1133':'팔을 쓸어가는 오른손의 곧은 손가락과 지지 왼손의 접힌 마디를 확인했다. 가려진 엄지와 겹친 끝은 모름으로 두었다.',
'WORD1290':'오른손의 펴진 각 손가락을 확인했다. 왼손은 손날 시점에서 개별 손가락을 분리할 수 없으면 모름이다. DEV11 f43은 손등이 보여 별도 확인했다.',
'WORD1567':'보이는 엄지와 주먹의 접힌 긴 손가락을 확인했다. 반대 손에 가려진 오른손 일부는 모름으로 남겼다.',
'WORD2523':'처음 삼각형 시점은 엄지와 검지만 확정했다. 마지막 손끝 모으기에서는 검지 굽힘만 확정한 경우가 많으며 겹친 다른 손가락은 확장하지 않았다.',
'WORD0071':'펼친 손과 검지만 편 손의 원본 RGB를 각각 확인했다. 교차로 가려진 DEV13 왼손은 모름이다.',
'WORD0500':'오른 검지와 접힌 긴 손가락을 확인했다. 왼 원형 틈을 이루는 엄지/검지는 겹침과 투영 때문에 정확한 굽힘을 확정하지 않았다.',
'WORD0678':'엄지와 검지가 벌어진 손에서 보이는 상태를 확인했다. 준비 상태나 손날 방향은 별도로 모름/굽힘으로 표시했다.',
'WORD0949':'옆으로 보이는 검지는 펼침, 화면 앞을 향해 길이가 눌린 검지는 모름으로 구분했다. 앞뒤 시점에서 값을 복사하지 않았다.',
'WORD1139':'오른손의 접힌 중지·약지·소지와 보이는 검지를 확인했다. 왼손의 앞을 향하는 겹친 긴 손가락은 모름이다.',
'WORD2551':'펴짐과 아직 완전히 접히지 않은 곡선 마디를 구분했다. 가려진 엄지는 모름이며 준비 시점의 흔들린 윤곽은 강제 분류하지 않았다.'}
def main():
    inputs=read(U/'ORDINARY_REVIEW_INPUTS.json')['rows'];out=[];mp={'E':0,'B':1,'F':2,'U':-1}
    for word,roles in LABELS.items():
        for role,vals in roles.items():
            rr=[r for r in inputs if r['word']==word and r['role']==role];assert len(rr)==len(vals)==6
            for r,codes in zip(rr,vals):
                assert sha(Path(r['image']))==r['image_sha256']
                for side,code in zip(['R','L'],codes.split('/')):
                    assert len(code)==5
                    out.append(dict(**r,side=side,y=[mp[x] for x in code],states=list(code),known=sum(x!='U' for x in code),annotation_basis=NOTES[word],reviewer='Codex expansion_labels_review_1113 direct RGB review',review_status='agent_rgb_review_pending_root',derived_from_word_identity=False,derived_from_model=False))
    counts={role:dict(collections.Counter(s for r in out if r['role']==role for s in r['states'])) for role in ['fit','development']}
    write(U/'ORDINARY_LABELS.json',dict(schema=dict(fingers=['thumb','index','middle','ring','little'],codes=mp,definitions={'E':'visibly extended at this frame','B':'visible bending, not fully folded into palm','F':'visibly folded toward palm','U':'cannot distinguish reliably; not wrong'}),rows=out,counts=counts,frame_count=len(inputs),scope='sparse direct native RGB labels; no interpolation, no expert annotation claim; DEV labels evaluation-only',selected_before_predictions=True))
    print(len(out),counts)
if __name__=='__main__':main()
