from motion_core import *
specs={
'0003':(4,[([44,46],[50,52],0),([51,53],[57,59],2),([59,61],[65,67],0)],[[72,78]],'양손 높이 관계의 세 방향 구간. 원 회전 횟수 판정은 아님.'),
'0500':(0,[([56,58],[64,66],0)],[[68,74]],'오른손이 얼굴 부근에서 왼손 쪽으로 내려감. 접촉 정답은 붙이지 않음.'),
'0678':(0,[([76,78],[85,87],0)],[[90,102]],'오른손 높이가 내려간 뒤 유지. 손모양 전환 정답과 분리.'),
'1139':(3,[([52,54],[59,61],0)],[[64,70]],'양손 간격 감소 후 유지. 실제 접촉 유무는 주장하지 않음.'),
'1290':(0,[([55,57],[59,61],0),([60,61],[63,64],2),([63,64],[67,69],0)],[[76,84]],'짧은 하강-상승-하강. 이후 작은 흔들림은 추가 의미 동작으로 강제 분할하지 않음.'),
'2551':(0,[],[[66,75]],'손가락 모양은 바뀔 수 있지만 이 구간의 손목 높이 유지 정답만 사용.')}
assert not (U/'EVALUATION_LABELS.json').exists()
rows=[]
for r in read(U/'RESERVED_PUBLIC.json')['rows']:
 w=r['word_id'][4:];axis,events,holds,note=specs[w];sheet=U/'public_rgb'/f'dense_{w}.jpg';sheet=sheet if sheet.exists() else Path(r['sheet'])
 rows.append(dict(clip_id=r['clip_id'],word_id=r['word_id'],person=13,axis=axis,moving='R',anchor='L',events=[dict(onset=a,offset=b,direction=d) for a,b,d in events],holds=holds,source_sha256=r['source_sha256'],sheet=str(sheet),sheet_sha256=sha(sheet),notes=note))
write(U/'EVALUATION_LABELS.json',dict(rows=rows,reviewer='Same Codex RGB review before model scores; not independent human/expert truth.',scope='Observed 2D motion only; not educational correctness.',originals=6,motion_legs=9,holds=6))
write(U/'EVAL_PROTOCOL.json',dict(primary_seed=42,seeds=[42,43,44],fresh_records=13,prior_test_exposure='Common developmental data previously exposed; only fine labels new.',manual_and_automatic_separated=True,invalid_in_denominator=True,reverse_is_synthetic=True,fullpath_modes=['motion_only','combined'],fresh_kinds=['normal','whole_reverse','speed075','speed125'],adopt='No added normal/speed/crossing warnings; no extra false movement on holds; old shape preserved; no web deployment. Report even if failed.',completion_claim='local stop/turn/continuation is separate from semantic completion; never claim omitted action from unknown.',no_postscore_tuning=True))
print('reserved labels frozen: 6 originals, 9 motion legs, 6 height/relationship holds')
