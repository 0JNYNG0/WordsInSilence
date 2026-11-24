"""Seven pinned original RGB frames and provenance for root review, no relabel."""
from pathlib import Path
import json,hashlib,math
from collections import Counter
import cv2
from PIL import Image,ImageDraw,ImageFont
U=Path(__file__).resolve().parent;G=U.parent;O=G.parent
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def main():
    assert not (U/'STRONG_CONFLICT_PROVENANCE.json').exists()
    audit=read(U/'DIRECT_ROLE_AUDIT.json')
    selected=[r for r in audit['rows'] if r['subtype']=='existing_direct_w1_conflict'];assert len(selected)==7
    bp=G/'reviews/obedience_occurrence_labels.json';ep=G/'reviews/egypt_occurrence_labels.json'
    sourceby={r['clip_id']:r for r in read(bp)['clips']+read(ep)['rows']}
    R=O/'word100_candidate_role_learning_20261111';P=O/'word100_action_episode_supervision_20261110';L=O/'word100_event_guide_linkage_20261110';E=O/'word100_event_completion_20261109'
    anchors={r['clip_id']:r for r in read(E/'REVIEW_LABELS.json')['rows']};episodes=read(P/'ACTION_EPISODES.json')['rows'];current=read(R/'ROLE_LABELS.json')
    mapfamily={0:0,1:1,2:2,3:3,4:4,5:4,6:5,7:5,8:6,9:7}
    font=ImageFont.truetype('C:/Windows/Fonts/malgun.ttf',23);small=ImageFont.truetype('C:/Windows/Fonts/malgun.ttf',18)
    sheet=Image.new('RGB',(2160,1320),'#edf1f5');draw=ImageDraw.Draw(sheet)
    labelsko={0:'배경',1:'얼굴 앞 제시',6:'중앙 손 모음',7:'양손 분리',8:'준비',9:'복귀'}
    imgs=U/'strong_conflict_frames';imgs.mkdir(exist_ok=True);rows=[];sourcecache={}
    for n,r in enumerate(selected):
        cid=r['clip_id'];f=r['source_frame_id'];source=sourceby[cid]
        sp=Path(source['source'])
        if str(sp) not in sourcecache:sourcecache[str(sp)]=sha(sp)
        assert sourcecache[str(sp)]==r['source_sha256']
        cap=cv2.VideoCapture(str(sp));cap.set(cv2.CAP_PROP_POS_FRAMES,f);ok,bgr=cap.read();cap.release();assert ok
        im=Image.fromarray(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB));assert im.size==(1920,1080)
        full=imgs/f'{cid}_f{f:04d}.png';im.save(full)
        # Same broad torso/arm crop for every frame; no pose-derived ROI.
        box=[500,0,1420,1080];crop=im.crop(box);cp=imgs/f'{cid}_f{f:04d}_body.png';crop.save(cp)
        x=n%4*540;y=n//4*660;draw.rectangle((x+6,y+6,x+534,y+654),fill='white')
        draw.text((x+18,y+15),f"{cid.split('_')[2]}  FIT{r['record_id']:02d}  f{f}",font=font,fill='#172c43')
        draw.text((x+18,y+49),f"{r['old_family']} {labelsko[r['old_family']]} → {r['proposed_family']} {labelsko[r['proposed_family']]}",font=font,fill='#9a4b00')
        thumb=crop.resize((480,564));sheet.paste(thumb,(x+30,y+87))
        old=anchors[cid];a=next((a for a in old['anchors'] if a['frame']==f),None);assert a is not None
        assert old['source_sha256']==r['source_sha256'] and sha(old['sheet'])==old['sheet_sha256']
        rawlabel=a['label'];linklabel=(rawlabel+1)//2 if rawlabel else 0;pre=mapfamily[linklabel]
        covering=[e for e in episodes if e['clip_id']==cid and e['family'] is not None and e['onset'][0]<=f<=e['offset'][1]]
        projected=covering[-1]['family'] if covering else pre
        assert projected==r['old_family']
        assert not any(c['clip']==cid and c['frame']==f for c in current['changes'])
        if r['old_family']==0:
            cause='taxonomy_granularity_background_vs_entry_return'
            reason='11/09의 0은 주요 행동 외 배경을 뜻하며 정지 자세만 뜻하지 않는다. 11/11 역할 확장 때 이 FIT01/07 프레임은 재검토 대상이 아니어서 준비/복귀 분류가 추가되지 않았다. 새 주석은 같은 움직임을 더 세분화한 것이며 기존 RGB 오판이라고 단정할 수 없다.'
            recommendation='새 occurrence 학습 자료에서만 준비8/복귀9로 구분할 수 있다. 원본 0을 오류로 삭제하지 말고 기존 목적과 새 분류체계를 함께 기록한다.'
        elif rawlabel%2==1 and r['old_family']==1:
            cause='approach_and_presented_core_collapsed_to_one_family'
            reason='11/09 원래 라벨1은 얼굴 앞 행동으로 접근/진행 중인 상태, 라벨2가 보이는 핵심 자세였다. 이후 (label+1)//2로 둘을 family1로 합쳤다. 새 주석은 접근 부분을 준비8로 분리하므로 원래 손 모양을 잘못 본 문제보다 역할 범위 정의가 달라진 사례다.'
            recommendation='새 자료에서는 준비를 주요 제시 완료와 구분한다. 원래 접근 라벨의 의미를 보존하고 모양 라벨은 바꾸지 않는다.'
        else:
            cause='uncertain_onset_band_promoted_to_strong_whole_interval'
            reason='원래 직접 RGB anchor16(중앙 모음 핵심)은 family6으로 올바르게 변환됐다. 11/10 ACTION_EPISODES의 분리 onset[37,39]..offset[43,45]를 prepare.py가 [37:46] 전체 w=1/family7로 덮어써 f37도 벌림으로 바뀌었다. 불확실한 시작 경계의 하한을 확정된 행동으로 만든 구성 문제이며, 새 직접 관찰은 f37 중앙 모음을 지지한다.'
            recommendation='새 학습 자료에서 불확실 경계 밴드를 마스크하고 직접 확인한 f37의 family6을 별도로 채택할지 root가 RGB를 보고 결정한다. 기존 동결 파일은 그대로 둔다.'
        rows.append(dict(clip_id=cid,source_frame_id=f,source=str(sp),source_sha256=r['source_sha256'],
            original_rgb=str(full),original_rgb_sha256=sha(full),body_crop=str(cp),body_crop_sha256=sha(cp),crop_xyxy=box,
            old_family=r['old_family'],new_proposed_family=r['proposed_family'],old_weight=r['old_weight'],
            original_anchor=dict(label=rawlabel,meaning=('background' if rawlabel==0 else 'approach_or_in_progress' if rawlabel%2 else 'visible_core'),
              word_index=old['word'],event_group_index=(rawlabel-1)//2 if rawlabel>0 else None,anchor_file=str(E/'REVIEW_LABELS.json'),sheet=old['sheet'],sheet_sha256=old['sheet_sha256']),
            label_chain=dict(raw_anchor=rawlabel,link_group_after_pair_collapse=linklabel,episode_family_before_boundary_overwrite=pre,
               covering_boundary_events=covering,after_boundary_overwrite=projected,later_role_extension_changed_this_frame=False),
            classification=cause,interpretation_ko=reason,recommendation_ko=recommendation,
            exact_endpoint_claim=False,semantic_correctness_claim=False,root_review_required=True,old_labels_modified=False))
    draw.text((1640,740),'검토용 7개 원본 프레임',font=font,fill='#172c43')
    draw.text((1640,790),'기존 라벨 → 새 역할 후보\n새 데이터 채택 여부는 별도 검토\n기존 정답/모델은 수정하지 않음',font=small,fill='#394c5f',spacing=12)
    out=U/'STRONG_CONFLICTS.png';sheet.save(out)
    pinfiles=[U/'DIRECT_ROLE_AUDIT.json',bp,ep,E/'REVIEW_LABELS.json',E/'labels.py',L/'link_core.py',P/'ACTION_EPISODES.json',P/'prepare.py',P/'episode_core.py',R/'labels.py',R/'ROLE_LABELS.json']
    result=dict(status='ROOT_RGB_REVIEW_PACK_READY_NO_RELABEL',sheet=str(out),sheet_sha256=sha(out),rows=rows,
       counts=dict(Counter(r['classification'] for r in rows)),
       overall='4개는 배경→준비/복귀의 세분화,2개는 접근/핵심을 합친 후 생긴 역할범위 차이,1개는 불확실한 시작 경계를 강한 벌림 라벨로 넓힌 구성 충돌이다. 일괄적인 원래 RGB 오표기7개로 부르지 않는다.',
       provenance_code_lines=[dict(path=str(E/'labels.py'),line=8,meaning='Odd labels approach/in-progress; even labels visible core.'),
         dict(path=str(L/'link_core.py'),line=38,meaning='Pair collapse (label+1)//2 combines approach and core.'),
         dict(path=str(P/'prepare.py'),line=24,meaning='Whole onset-low to offset-high interval assigned family and w1.'),
         dict(path=str(R/'labels.py'),line=15,meaning='Prior zeros remain generic background unless explicitly re-reviewed.')],
       pins=[dict(path=str(p),sha256=sha(p)) for p in pinfiles],source_files_unchanged=True,
       no_model_inference=True,no_training=True,no_dev_or_private_access=True,no_old_labels_modified=True,
       reviewer='Codex provenance arithmetic and original frame extraction; root visual review pending, not independent human truth')
    write(U/'STRONG_CONFLICT_PROVENANCE.json',result);print(json.dumps(result['counts'],ensure_ascii=False))

if __name__=='__main__':main()
