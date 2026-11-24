from audit import *
R=U/'rgb_review'
assert not(R/'VISUAL_REVIEW.json').exists()
m=read(R/'MANIFEST.json');matched=read(R/'MATCHED_COORDINATES.json');rad=read(U/'RADIAL_HOLD_AUDIT.json')
notes={
'NIA_SL_WORD1133_REAL02_F_sign_conflict':dict(progress='54~64에서 열린 오른손이 반대 전완을 따라 손목 쪽으로 진행한다. 관측된 흐름에서 명백한 역방향 재동작은 보이지 않는다.',anchor='반대팔 팔꿈치 부근은 움직이는 손에 가렸다 드러난다. 팔도 조금 움직이므로 실제 회전과 관절점 추정 오차의 비율은 단안 RGB로 확정할 수 없다.',contract='동일 원본 시각의 full/crop 좌표가 방향 부호까지 다르므로, projected delta를 곧바로 영상 속 행동 방향 정답으로 부를 수 없다.'),
'NIA_SL_WORD1133_REAL05_F_sign_conflict':dict(progress='47~54의 전체 흐름은 열린 손이 반대팔을 따라 손목 쪽으로 진행한다.49~52만 떼면 이동이 짧고 작다.',anchor='반대팔과 몸의 작은 움직임이 함께 보인다. 큰 급회전을 단정할 근거는 없으며, 명확한 실제 팔꿈치 관절 중심은 표면 RGB에서 직접 계측하지 않았다.',contract='full -0.0030/crop +0.0234의 작은 몸손목 투영 차이를 역방향 오류로 확정하지 않는다.'),
'NIA_SL_WORD1133_REAL02_F_amplitude_overlap':dict(progress='52~55는 팔을 따라 이어지는 진행,67~70은 손목 부근에서 손을 얹은 관계를 유지하는 모습이다.',anchor='유지에서도 미세한 팔/손 이동이 있어 완전한 픽셀 정지는 아니다.',contract='몸손목 단일 변위 폭만으로 진행과 유지의 의미를 정의하지 않는다.'),
'NIA_SL_WORD1133_REAL04_F_amplitude_overlap':dict(progress='79~85는 작은 범위의 진행,92~95는 손을 얹은 위치 관계를 대체로 유지한다.',anchor='팔의 표면과 손이 겹쳐 실제 관절 중심/물리적 접촉은 확정하지 않는다.',contract='짧고 작은 이동도 주요 움직임의 일부일 수 있다. 수행 완료/언어적 정답 여부는 판정하지 않는다.'),
'NIA_SL_WORD1133_REAL05_F_amplitude_overlap':dict(progress='48~54에서 열린 손이 반대팔을 따라 옮겨가고65~68은 손목 쪽에 둔 관계를 유지한다.',anchor='두 팔 전체의 작은 이동과 손의 상대 이동을 구분해야 한다.',contract='전완 축이 조금 변한 것을 곧바로 별도 쓸기 동작으로 해석하지 않는다.'),
'NIA_SL_WORD1133_REAL07_F_amplitude_overlap':dict(progress='56~61의 작은 이동은 이어지는 쓸기 진행으로 보이고69~72는 이동 끝에서 위치를 유지/정착하는 모습이다.',anchor='후반에도 자연스러운 조금의 이동은 보인다.',contract='유지를 수치 변화가 정확히0인 상태로 정의하지 않는다.')}
out=[];unique={}
for c in m['cases']:
    assert c['id']in notes and sha(c['path'])==c['sha256'] and sha(c['source'])==c['source_sha256']
    for aa,bb in c['ranges']:unique.setdefault(c['clip_id'],set()).update(range(aa,bb+1))
    overlay=next(q for q in matched['sheets']if q['id']==c['id'])
    out.append(dict(id=c['id'],clip_id=c['clip_id'],source_sha256=c['source_sha256'],source_ranges=c['ranges'],raw_sheet=c['path'],raw_sheet_sha256=c['sha256'],raw_sheet_viewed=True,
        coordinate_overlay=overlay['path'],coordinate_overlay_sha256=overlay['sha256'],coordinate_overlay_viewed=c['id'].endswith('sign_conflict'),**notes[c['id']]))
h=rad['largest_hold'];assert sha(h['sheet'])==h['sheet_sha256'];unique.setdefault(h['clip_id'],set()).update(h['frames'])
out.append(dict(id=h['clip_id']+'_largest_hold',clip_id=h['clip_id'],source_sha256=h['source_sha256'],source_frames=h['frames'],coordinate_overlay=h['sheet'],coordinate_overlay_sha256=h['sheet_sha256'],coordinate_overlay_viewed=True,
    progress='56~60의 reviewed hold는 손을 얹은 관계를 대체로 유지하면서 두 팔 전체가 조금 내려간다. 별도 반대팔 쓸기 동작으로 바뀌는 근거는 확인되지 않는다.',
    anchor='전완이 카메라 앞쪽으로 향해 짧게 투영되고 손이 겹친다. 축 회전 수치가 크다고 실제 팔이 그 각도만큼 회전했다고 할 수 없다.',
    contract='0.1081의 투영변위 중0.0863은 축 변화 항이다. 자연스러운 정착·상대관계 유지와 별도 진행을 구분해야 한다.'))
write(R/'VISUAL_REVIEW.json',dict(at=stamp(),status='CODEX_SOURCE_RGB_REVIEW_NOT_HUMAN_EXPERT_GOLD',reviewer='motion_contract_review_1113 Codex, actual displayed continuous RGB sheets; not independent human or KSL expert',rows=out,unique_source_frames=sum(map(len,unique.values())),source_clips=len(unique),new_educational_correctness_labels=False,new_direction_target_labels=False,existing_labels_changed=False,new_model_inference=False,
    limitation='2D RGB and estimated landmarks do not establish exact3D rotation, true joint centres or contact. No coordinate sign was promoted to semantic ground truth.'))
print('Visual review recorded',len(out),'sheets/cases',sum(map(len,unique.values())),'unique frames',len(unique),'FIT clips')
