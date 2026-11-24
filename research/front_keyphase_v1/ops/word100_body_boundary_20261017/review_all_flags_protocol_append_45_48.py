"""Record already visually inspected condition pages45-48, no detector/inference."""
from review_all_flags_protocol_log import append, read, sha, OUT, U
import json

notes={
'W0530_C01':('팔십: 원본과 추가 p02의 59·70·80·87에서 오른손 숫자 동작은 가슴·얼굴 앞에 있고 왼팔은 아래로 고정되어 있다. 세 관측에서 오른손 21점이 유지되고, 제외된 왼손은 하단 밖 휴식손이다. 새 필수 동작 잘림은 이 표본에서 보이지 않는다.',False),
'W0533_C01':('팔십: peak56에서 오른손은 가슴 앞 숫자 동작, 왼손은 몸 옆 아래에 있다. 끝에서 오른손을 내리는 문맥이며 왼손 하단 제거에 따른 새 동작 잘림은 보이지 않는다.',False),
'W0536_C01':('팔십: peak35의 오른손 가슴 앞 동작과 반대편 아래로 내린 왼손이 분리된다. 마지막은 손을 내려 놓은 상태이고, 제외 영역은 왼손 휴식 위치이다.',False),
'W0539_C01':('팔십일: 추가 p02의 50·62·74·87에서 오른손이 가슴부터 얼굴 앞에서 숫자 모양을 바꾸고 왼손은 계속 옆으로 내려져 있다. 세 관측의 오른손 21점은 유지되어, 왼손 제거는 관찰 표본상 휴식손 제외이다.',False),
'W0542_C01':('팔십일: peak72에서 오른손 숫자 동작이 얼굴 앞에 있고 왼손은 하단 밖에 내려져 있다. 끝은 오른손 하강이며 이 표본에서 의미 있는 움직이는 왼손의 추가 잘림은 보이지 않는다.',False),
'W0545_C01':('팔십일: peak30의 오른손 숫자 동작은 가슴 앞에서 보존되고 왼손은 몸 옆 아래에 정지해 있다. 하단 경계가 새로 자른 핵심 움직임은 표본에서 보이지 않는다.',False),
'W0548_C01':('팔십육: 추가 p02의 45·57·69·81에서 오른손 숫자 동작이 가슴·얼굴 앞에서 이어지고 왼손은 내려진 위치를 유지한다. 세 관측 오른손 21점 보존, 왼손 제거는 휴식손 제외로 보인다.',False),
'W0551_C01':('팔십육: peak79 오른손 숫자 동작은 가슴 앞, 왼손은 옆으로 내려져 있다. masked/sensor에서 사라진 왼손은 이 표본의 하단 휴식손이며 새 동작 잘림은 보이지 않는다.',False),
'W0554_C01':('팔십육: peak27에서 오른손은 가슴 앞 숫자 모양이고 왼손은 아래에 내려져 있다. 시작·끝과 함께 보아 왼손 하단 제외의 새 핵심 잘림 근거는 없다.',False),
'W0557_C01':('팔십구: 추가 p02의 48·58·70·83을 확인했다. 왼손은 계속 아래 정지, 오른손은 가슴·얼굴 앞 숫자 동작이다. 48에서 native/masked 오른손 0점인 반면 sensor는 21점이며 이후 세 관측 모두 21점이다. 추출 차이는 있으나 경계로 새로 잘린 동작은 이 표본에서 보이지 않는다.',False),
'W0560_C01':('팔십구: peak74에서 오른손 숫자 동작이 가슴 앞에 있고 왼손은 아래 휴식 위치이다. 하단 제외는 반대 손에 해당하며 새로운 핵심동작 잘림은 관찰되지 않는다.',False),
'W0563_C01':('팔십구: peak50에서 오른손 숫자 동작은 얼굴 앞, 왼손은 아래에 내려져 있다. 세 보기의 동작하는 손은 경계 위이며 제외는 휴식 왼손에 해당한다.',False),
'W0566_C01':('백: peak32 오른 검지가 전방에 제시되고 왼손은 아래로 내려진 상태이다. 경계 밖 왼손 제거에 의한 새 필수 동작 잘림은 이 표본에서 보이지 않는다.',False),
'W0569_C01':('백: peak29에서 오른 검지가 가슴 앞에 있고 왼손은 몸 옆 하단에 있다. 동작 손은 경계 위에 보존되고 반대 휴식 손이 제외된다.',False),
'W0572_C01':('백오: peak43에서 오른 검지 전방 동작, 왼손은 몸 옆 아래에 있다. 앞뒤 원본 문맥에서도 왼손의 별도 핵심 동작 잘림 근거는 보이지 않는다.',False),
'W0575_C01':('백오: peak30 오른 검지 동작은 가슴 앞, 왼손은 하단 밖으로 내려져 있다. 관찰된 제거는 휴식 손이고 새 핵심동작 잘림은 보이지 않는다.',False),
'W0578_C01':('백오: peak45 오른 검지 전방 동작과 아래 정지한 왼손이 분리된다. 제외된 왼손의 필수 움직임은 표본에서 보이지 않는다.',False),
'W0581_C01':('백십육: peak42 오른 검지 전방 동작, 끝에서는 오른손 다른 숫자 모양이다. 왼손은 아래로 내려진 채이며 새로 잘린 왼손 동작은 보이지 않는다.',False),
'W0584_C01':('백십육: peak48 오른 검지 동작은 가슴 앞에서 유지되고 왼손은 아래 휴식 위치이다. 하단 제거는 반대편 휴식 손으로 보인다.',False),
'W0587_C01':('백십육: peak43에서 오른 검지 동작은 가슴 앞이고 왼손은 아래에 있다. 종료는 손을 내려놓은 상태이며 이 표본에 새 핵심동작 잘림 근거는 없다.',False),
}
append([45,46,47,48],notes)

partial={
 'W0530':[59,70,80,87], 'W0539':[50,62,74,87],
 'W0548':[45,57,69,81], 'W0557':[48,58,70,83],
}
rows=[]
for wid,frames in partial.items():
 m=read(OUT/(wid+'.json'));p=OUT/(wid+'_p02.jpg')
 assert [r['frame_id'] for r in m['selected_frames']][4:8]==frames
 im=next(r for r in m['images'] if str(r['path'])==str(p))
 assert sha(p)==im['sha256']
 rows.append(dict(window_id=wid,condition_review_id=wid+'_C01',material_manifest=str(OUT/(wid+'.json')),material_manifest_sha256=sha(OUT/(wid+'.json')),record_sha256=m['record_sha256'],source_sha256=m['source_sha256'],viewed_images=[im],actual_viewed_frames=frames,actual_overlay_compared_frame_ids=frames,observation_ko=notes[wid+'_C01'][0],every_source_frame_viewed=False,only_listed_detail_page_viewed=True,unresolved_essential_geometric_cut=False))
out=U/'review_all_flags_protocol_PARTIAL_DETAIL_45_48.json'
assert not out.exists()
out.write_text(json.dumps(dict(status='ACTUAL_PARTIAL_DETAIL_VIEW_RECORDED',reviewer='roi_training_protocol',rows=rows,scope='새 추론/검출/학습 없음. p02 실제 열람만 기록; p01/p03/p04 미열람은 주장하지 않음.'),ensure_ascii=False,indent=2),encoding='utf-8')
print(out)
