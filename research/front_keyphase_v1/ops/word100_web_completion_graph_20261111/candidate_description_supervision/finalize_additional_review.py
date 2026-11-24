"""Record inspected RGB evidence, not model-generated labels or expert correctness."""
from prepare_additional_source_review import D
from core import *
SPEC={
 '0739_3':dict(events=[(2,[66,68],[78,80],[70,72,74,76]),(3,[82,84],[90,92],[84,86,88])],holds=[[94,102]],entry=[0,6,12,18,24,30,36,42,48],other=[54,60,64,66],ret=[108,114,120,126,132,138,144]),
 '0739_7':dict(events=[(2,[50,52],[62,64],[54,56,58,60]),(3,[64,66],[70,72],[66,68,70])],holds=[[72,78]],entry=[0,6,12,18,24,30],other=[36,42,48,50],ret=[84,90,96,102,108,114,120]),
 '1133_3':dict(events=[(4,[52,54],[64,66],[56,58,60,62,64])],holds=[[68,78]],entry=[0,6,12,18,24,30,36,42,48],other=[],ret=[84,90,96,102,108,114]),
 '1133_7':dict(events=[(4,[50,52],[64,66],[54,56,58,60,62,64])],holds=[[68,72]],entry=[0,6,12,18,24,30,36,42],other=[],ret=[78,84,90,96,102,108,114]),
 '1567_3':dict(events=[(5,[40,42],[48,50],[44,46]),(5,[50,52],[58,60],[52,54,56]),(5,[60,62],[68,70],[62,64,66])],holds=[[72,78]],entry=[0,6,12,18,24,30,36],other=[],ret=[84,90,96,102,108,114]),
 '1567_7':dict(events=[(5,[34,36],[40,42],[38,40]),(5,[44,46],[52,54],[46,48,50]),(5,[54,56],[64,66],[56,58,60,62])],holds=[[68,72]],entry=[0,6,12,18,24,30],other=[],ret=[78,84,90,96,102,108]),
 '2523_3':dict(events=[(7,[66,68],[76,78],[68,70,72,74])],holds=[[78,84]],entry=[0,6,12,18,24,30,36,42,48,54],other=[60,62,64,66],ret=[90,96,102,108,114,120]),
 '2523_7':dict(events=[(7,[44,46],[52,54],[46,48,50,52])],holds=[[56,60]],entry=[0,6,12,18,24,30,36],other=[40,42,44],ret=[66,72,78,84])}
NOTES={'WORD0739':'편 손을 얼굴 앞에 제시한 다음 손을 닫으며 아래로 내리고, 닫힌 손을 유지하면서 팔뚝을 다시 세운다. 준비 손올림과 마지막 손내림은 별도다.','WORD1133':'왼팔을 내밀고 오른손을 반대팔 위에 준비한 다음 손목 방향으로 쓸어 이동하고 잠시 유지한다.','WORD1567':'엄지를 든 두 손의 높이가 번갈아 바뀌는 흐름과 마지막 유지 구간을 확인했다. 관측된 교대 회차 수를 모든 표현의 필수 반복 횟수로 정하지 않는다.','WORD2523':'모은 두 손을 얼굴 앞에서 벌려 내리는 흐름과 벌린 뒤 유지 구간을 확인했다. 이번 주석은 움직임 범위이며 손가락 굽힘 정답을 추가 추정하지 않는다.'}
queue=read(D/'QUEUE.json');dense={r['clip_id']:r for r in read(D/'DENSE_QUEUE.json')['rows']};rows=[];contrasts=[]
for r in queue['rows']:
 cid=r['clip_id'];spec=SPEC[r['word_id'][4:]+'_'+str(r['person'])];ds=dense[cid];assert sha(r['source'])==r['source_sha256'] and sha(r['overview'])==r['overview_sha256'] and sha(ds['path'])==ds['sha256'];seen=set(r['viewed_frames']+ds['frames']);events=[];groups=[]
 for i,(f,on,off,points) in enumerate(spec['events']):
  assert set(on+off+points)<=seen and on[0]<=min(points)<=max(points)<=off[1] and on[1]<off[0]
  events.append(dict(event_id=f'{cid}:motion{i}',axis=AX[f],family=f,onset=on,offset=off,progression_witness_frames=points,linguistic_completion=None))
 for f in sorted({e['family'] for e in events}):
  ee=[e for e in events if e['family']==f];groups.append(dict(group_id=f'{cid}:axis{AX[f]}',axis=AX[f],family=f,event_ids=[e['event_id'] for e in ee],onset=ee[0]['onset'],offset=ee[-1]['offset'],observed_submovements=len(ee),required_repetition_count=None,definition='Observed source occurrence extent, not required semantic duration/repetition count.'))
 hold=[p for p in sorted(seen) if any(a<=p<=b for a,b in spec['holds'])];nonmotion=spec['entry']+spec['other']+spec['ret'];assert set(nonmotion)<=seen;assert not set(nonmotion)&{p for e in events for p in e['progression_witness_frames']};assert not set(hold)&set(nonmotion)
 rows.append(dict(clip_id=cid,person=r['person'],word_id=r['word_id'],role='fit',source=r['source'],source_sha256=r['source_sha256'],fps=r['fps'],frames=r['frames'],sheets=[dict(path=r['overview'],sha256=r['overview_sha256'],frames=r['viewed_frames'],viewed=True),dict(path=ds['path'],sha256=ds['sha256'],frames=ds['frames'],viewed=True)],events=events,groups=groups,holds=spec['holds'],hold_sample_frames=hold,observation_ko=NOTES[r['word_id']],reviewer='root Codex actual source RGB inspection, not independent human/KSL expert',dense_frame_truth=False))
 contrasts.append(dict(clip_id=cid,entry=spec['entry'],other_held_presentation=spec['other'],returning=spec['ret']))
out=D/'SOURCE_EPISODES_ADDED.json';assert not out.exists();write(out,dict(at=stamp(),rows=rows,contrasts=contrasts,status='8_PUBLIC_FIT_RGB_REVIEWED',source_count=8,source_people=[3,7],all_new_head_fit_only=True,training_allowed=False,expert_gold=False,source_language_correctness_not_newly_adjudicated=True,scope='Boundary bands and sparse movement witnesses only; no interpolated or model-predicted dense gold.'))
print('Recorded8 public FIT RGB reviews;',sum(len(r['groups']) for r in rows),'groups')
