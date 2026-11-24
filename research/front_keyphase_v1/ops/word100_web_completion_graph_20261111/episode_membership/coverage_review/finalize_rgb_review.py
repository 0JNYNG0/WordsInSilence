"""Root's actual source-RGB observations; not expert or dense-frame truth."""
from pathlib import Path
import json,hashlib,collections
D=Path(__file__).resolve().parent;U=D.parents[1]
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
# Each event: axis, onset band, offset band, directly viewed progression witnesses.
# Each group contains all visible turns of ONE recorded alternating occurrence.
specs={
 '0739_02':([ (0,[62,64],[70,72],[64,66,68]),(1,[72,74],[80,82],[74,76,78]) ],[[84,96]],'손을 닫으며 얼굴에서 가슴 쪽으로 내린 뒤, 닫힌 손을 유지하며 팔뚝을 세운다.'),
 '1133_02':([ (2,[46,48],[60,64],[48,50,52,54,56,58,60]) ],[[64,72]],'반대 팔 윗부분 부근에서 손목 부근으로 열린 손이 이동한다. 정확한 접촉은 미확정.'),
 '1567_02':([ (4,[46,48],[54,56],[48,50,52]),(4,[56,58],[64,66],[58,60,62]),(4,[66,68],[74,76],[68,70,72]) ],[[80,100]],'오른손-왼손의 상대 높이가 세 차례 바뀐 뒤 마지막 자세를 유지한다. 언어적 필수 반복 횟수는 아님.'),
 '2523_02':([ (3,[54,56],[60,62],[56,58,60]) ],[[64,80]],'중앙에 모인 양손이 양옆·아래로 벌어지고 이후 유지한다. 세부 엄지나 접촉 정답은 없음.'),
 '0739_04':([ (0,[86,88],[96,100],[88,90,92,94]),(1,[102,104],[110,116],[104,106,108,110]) ],[[116,128]],'기존04 검토와 합쳐 하강 끝96~100, 전완 세움 끝110~116으로 보수적 경계 유지. 끝 자세의 작은 정착을 운동/유지로 단정하지 않는다.'),
 '1133_04':([ (2,[76,78],[88,92],[78,80,82,84,86,88]) ],[[92,100]],'팔 위쪽 부근에 접근한 뒤 손목 쪽으로 이동하고 유지. 접근을 쓸어내리기와 분리.'),
 '1567_04':([ (4,[66,68],[74,76],[68,70,72]),(4,[76,78],[86,88],[78,80,82,84]),(4,[88,90],[98,100],[90,92,94,96]) ],[[104,120]],'세 번의 양손 상대 높이 교체를 한 반복 동작 묶음 안에 보존.'),
 '2523_04':([ (3,[86,88],[94,96],[88,90,92,94]) ],[[96,104]],'중앙 양손에서 옆·아래로 펼친 뒤 유지. 손가락 상세나 실제 접촉은 미판정.'),
 '0739_05':([ (0,[52,56],[64,68],[56,60,64]),(1,[68,72],[80,84],[72,76,80]) ],[[84,88]],'가슴 쪽 하강 뒤 닫힌 손의 전완을 세우는 별도 행동이 보인다.'),
 '1133_05':([ (2,[44,48],[60,64],[48,52,56,60]) ],[[64,68]],'반대팔 위쪽에서 손목 쪽으로 열린 손이 이동.'),
 '1567_05':([ (4,[44,48],[52,56],[48,52]),(4,[52,56],[60,64],[56,60]),(4,[64,68],[72,76],[68,72]),(4,[72,76],[84,88],[76,80,84]) ],[[88,92]],'왼손 높은 시작에서 오른손52,왼손64,오른손72,왼손84로 교대. 첫 교대는 준비와 이어져 경계 불확실. 세부 필수 반복 수로 사용 금지.'),
 '2523_05':([ (3,[60,64],[68,72],[64,68]) ],[[72,80]],'중앙 모양에서 양옆으로 벌어져 유지.'),
 '0739_06':([ (0,[68,72],[80,84],[72,76,80]),(1,[84,88],[92,96],[88,92]) ],[[96,104]],'손 닫기/가슴 하강과 이후 전완 세우기가 순차로 보임.'),
 '1133_06':([ (2,[40,44],[52,56],[44,48,52]) ],[[56,60]],'열린 손이 반대팔 위쪽에서 손목 부근으로 이동.'),
 '1567_06':([ (4,[52,56],[60,64],[56,60]),(4,[60,64],[68,72],[64,68]),(4,[68,72],[76,80],[72,76]) ],[[80,92]],'오른손 높은 자세에서 세 차례 교대 후 왼손 높은 자세 유지.'),
 '2523_06':([ (3,[68,72],[76,80],[72,76]) ],[[80,84]],'중앙 양손이 옆으로 내려가며 벌어진 뒤 유지.'),
 '0739_08':([ (0,[60,64],[68,72],[64,68]),(1,[72,76],[80,84],[76,80]) ],[[84,96]],'얼굴 앞 열린 손이 닫혀 내려간 뒤 전완이 세워짐.'),
 '1133_08':([ (2,[68,72],[88,92],[72,76,80,84,88]) ],[[92,100]],'열린 손이 반대팔 위쪽에서 손목 쪽으로 길게 이동한 뒤 유지.'),
 '1567_08':([ (4,[60,64],[68,72],[64,68]),(4,[68,72],[76,80],[72,76]),(4,[80,84],[92,96],[84,88,92]),(4,[92,96],[100,104],[96,100]) ],[[104,112]],'왼손 높은 시작 이후 오른손68,왼손76,오른손92,왼손104로 교대. 첫 교대와 준비의 경계는 넓게 둔다.'),
 '2523_08':([ (3,[52,56],[60,64],[56,60]) ],[[64,72]],'중앙에 모은 양손이 옆·아래로 벌어져 유지.'),
 '0739_09':([ (0,[52,56],[60,64],[56,60]),(1,[64,68],[72,76],[68,72]) ],[[76,80]],'손 닫기/가슴 하강 뒤 전완 세우기. 마지막 유지와 귀환 분리.'),
 '1133_09':([ (2,[44,48],[56,60],[48,52,56]) ],[[60,68]],'반대팔 위쪽에서 손목 쪽으로 열린 손 이동.'),
 '1567_09':([ (4,[56,60],[64,68],[60,64]),(4,[68,72],[76,80],[72,76]),(4,[76,80],[84,88],[80,84]) ],[[88,100]],'오른손 높음→왼손 높음→오른손 높음→왼손 높음 뒤 유지.'),
 '2523_09':([ (3,[64,68],[72,76],[68,72]) ],[[76,80]],'중앙 양손이 양옆·아래로 벌어진 뒤 유지.')
}
def main():
 out=D/'SOURCE_EPISODES.json';assert not out.exists()
 queue={r['clip_id']:r for p in ['REVIEW_QUEUE.json','REMAINING_REVIEW_QUEUE.json'] for r in read(D/p)['rows']}
 dense={r['clip_id']:r for r in read(D/'COARSE_RGB_REVIEW.json')['rows']};rows=[]
 for key,(events,holds,note) in specs.items():
  word,person=key.split('_');cid=f'NIA_SL_WORD{word}_REAL{person}_F';r=queue[cid];assert sha(r['overview'])==r['overview_sha256'];assert sha(r['source'])==r['source_sha256']
  sheets=[dict(path=r['overview'],sha256=r['overview_sha256'],frames=r['sampled_source_frames'],viewed=True)]
  if cid in dense:
   dr=dense[cid];assert sha(dr['dense_path'])==dr['dense_sha256'];sheets.append(dict(path=dr['dense_path'],sha256=dr['dense_sha256'],frames=dr['dense_frames'],viewed=True))
  seen=set(f for s in sheets for f in s['frames']);es=[]
  for j,(axis,on,off,witness) in enumerate(events):
   assert on[0]<=on[1]<off[1] and on[0]<off[0]<=off[1];assert set(witness)<=seen
   es.append(dict(event_id=f'{cid}:motion{j}',axis=axis,family={0:2,1:3,2:4,3:7,4:5}[axis],onset=on,offset=off,progression_witness_frames=witness,linguistic_completion=None))
  groups=[]
  for axis in sorted(set(e['axis'] for e in es)):
   ee=[e for e in es if e['axis']==axis];groups.append(dict(group_id=f'{cid}:axis{axis}',axis=axis,family=ee[0]['family'],event_ids=[e['event_id'] for e in ee],onset=ee[0]['onset'],offset=ee[-1]['offset'],observed_submovements=len(ee),required_repetition_count=None,definition='Extent of the observed source occurrence, not required semantic duration/repetition count. A part may be linguistically adequate; no whole-sign correctness label.'))
  rows.append(dict(clip_id=cid,person=int(person),word_id='WORD'+word,role='fit',source=r['source'],source_sha256=r['source_sha256'],fps=r['fps'],frames=r['frames'],sheets=sheets,events=es,groups=groups,holds=holds,hold_sample_frames=[f for f in sorted(seen) if any(a<=f<=b for a,b in holds)],observation_ko=note,reviewer='root Codex actual source RGB inspection, not independent human/KSL expert',dense_frame_truth=False))
 reconciliation=[dict(clip='NIA_SL_WORD0739_REAL04_F',axis=1,old_end_band=[110,112],new_dense_impression=[114,116],retained_uncertain_band=[110,116],resolution='Do not choose the narrower impression. Preserve wider movement/settling uncertainty; no strong endpoint supervision inside this band.'),dict(clip='NIA_SL_WORD0739_REAL04_F',axis=0,old_occurrence_end=[102,104],old_movement_end=[96,100],new_movement_end=[96,100],resolution='Occurrence includes terminal hold; different semantic fields, not contradictory boundaries.')]
 report=dict(status='SOURCE_RGB_EPISODE_REVIEW_COMPLETE_PENDING_NUMERIC_ADMISSION',rows=rows,counts=dict(clips=len(rows),recorded_people=len(set(r['person'] for r in rows)),events=sum(len(r['events']) for r in rows),groups=sum(len(r['groups']) for r in rows),dense_sheets=8,overview_sheets=24),reconciliation=reconciliation,training_allowed=False,policy=['Only sampled source frames directly inspected. Unsampled instants are not dense manual truth.','Boundary bands remain uncertain; no exact single-frame target.','Viewed movement witnesses can be used for occurrence/extent supervision; no invisible finger/contact labels.','Hold-only is not proof the preceding movement occurred.','Separate partial source extent from educational error. Do not enforce recorded repetition number or identical movement amplitude.','Preparatory/return and unreviewed axes are unknown, not generic negative labels.','Existing public FIT recordings only; no private or fresh-final access.','Model probabilities did not define these annotations.'])
 write(out,report);print(report['counts'])
if __name__=='__main__':main()
