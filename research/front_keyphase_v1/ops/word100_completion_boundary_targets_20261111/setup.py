from boundary_core import *
from datetime import datetime,timezone,timedelta

def main():
 assert not (U/'SETUP_COMPLETE.json').exists()
 write(U/'FIRST_SETUP_ERROR.json',dict(error='NameError ROOT',cause='Generic core import name was shadowed by historical experiment module',fix='Unique boundary_core module name',training_started=False))
 for path in [ROOT/'WordsInSilence_PROJECT_CONTEXT.md',ROOT/'NEW_CHAT_HANDOFF.md',O/'STATE.json',ROOT/'Docs/PROJECT_DIRECTION_HISTORY_KO.txt']:
  dest=U/'before'/path.name;dest.parent.mkdir(exist_ok=True);dest.write_bytes(path.read_bytes())
 protected=read(P/'PROTECTIONS.json')
 protected.update({r['path']:r['after'] for r in read(P/'DEPLOYMENT.json')['changes']})
 for pattern in ['*.py','*.json','runs/*/*.pt']:
  protected.update({str(p):sha(p) for p in P.glob(pattern)})
 write(U/'PROTECTIONS.json',protected)
 write(U/'AUTHORIZATION.json',dict(user='다음 보완 작업을 세세하게 계획하고 순서대로 실행',scope='Public FIT supervision and endpoint model comparison; only passing research functions eligible for adoption',private_data=False,camera=False,downloads=False,external_transfer=False,C_T3F_preserved=True,implementation='Codex',independent_human=False))
 rows=read(P/'LABELS.json')['rows'];fitlookup={r['clip_id']:r for r in rows if r['role']=='fit'}
 extra={('0003',1):(0,[(34,36,38,40,2),(40,42,44,46,0),(46,48,50,52,2)],[(53,57)]),('0003',3):(0,[(56,58,62,64,2),(63,65,68,70,0),(69,71,74,76,2),(75,77,80,82,0)],[(84,90)]),('0003',5):(0,[(30,32,37,39,2),(39,41,44,46,0),(46,48,52,54,2)],[]),('0500',1):(1,[(51,54,61,63,2)],[(65,81)]),('0500',3):(1,[(30,33,40,43,2)],[(46,60)]),('0500',5):(1,[(65,68,73,76,2)],[(81,96)])}
 additions=[]
 for (word,person),(axis,es,holds) in extra.items():
  clip=f'NIA_SL_WORD{word}_REAL{person:02d}_F';src=fitlookup[clip];assert sha(src['sheet'])==src['sheet_sha256']
  additions.append(dict(clip_id=clip,person=person,role='fit',axis=axis,moving='L',anchor='R',events=[dict(onset=[a,b],offset=[c,d],direction=y) for a,b,c,d,y in es],holds=holds,background=[],source_sha256=src['source_sha256'],sheet=src['sheet'],sheet_sha256=src['sheet_sha256'],notes='Original RGB reviewed for real left wrist/forearm motion. Includes support-hand preparation; physical motion labels are NOT sign completion. Same Codex, no expert annotation.'))
 labels=rows+additions;write(U/'LABELS.json',dict(rows=labels,added_annotations=additions,existing_rows_unchanged=True,source='Original public RGB contact sheets; no model overlays',new_expert_labels=0))
 s=read(P/'TRAIN_SAMPLES.json');audit=dict(previous_multiple_boundary00=sum(r.get('kind')=='multiple' and r.get('boundaries')==[0,0] for r in s),meaning='A multi-event interval has outer start/end evidence; uniqueness belongs to a separate target. Former all-zero boundary target mixed these meanings.',prior_failed_full=[r for r in read(P/'DIRECT_SCORES.json') if r['seed']==42 and r['target']==2 and not r['after']['physical_event_complete']],causality='Supervision inconsistency identified, impact requires ablation; not claimed sole cause')
 write(U/'CAUSE_AUDIT.json',audit)
 plan='''이번 목표: 정상적으로 끝낸 이동을 보류하지 않으면서, 일부만 한 이동을 완료로 올리지 않기.
1. 이전 학습/저장결과의 원인을 감사한다. 복수행동의 시작·끝00과전체완료를혼동한감독을분리한다. 원본FIT6영상의실제왼손이동13개를보강하고시작/끝범위불확실성을유지한다. 평가의실패영상은학습에추가하지않는다.
2. 학습정답은후보가시작과끝을각각담았는지,행동하나인지/부분인지/여러개인지를나눠만든다. 검토경계안쪽에서잘린구간은확정정답으로만들지않는다. 시작·끝부근을별도로읽는Conv+GRU를주후보로미세조정하고,정답만수정한같은구조와비교한다. 기존공통100단어/모양/방향은고정한다.
3. 직접구간100개와기존자동78조건에서이전/정답만/정답+끝점표현/연결보호를비교한다. 판정에단어ID·원하는방향·편집시점을입력하지않는다. 전체완료추정이실패해도이미관측한손모양/방향정보를삭제하지않고완료항목만보류한다. 잘못된실제방향시도는기록에서빼지않는다.
4. 별도6표현의자동후보와,기존공개캐시에남아있는미사용정답구간을조건고정후검사한다. 기록ID/원본해시·과거노출을명시한다. 모든3seed에서부분오완료증가0·정상완료퇴행0·정상속도/교차추가경고0·기존모양방향오류손실0·관측부족정상승격0을요구한다. 통과하지않으면기존웹판정유지;완료학습/평가반복금지.
기대효과는가설이다. 현재자료의자연학습자오류정답이없어합성시험을실제교정정확도로보고하지않는다. 같은Codex 검산과독립전문가검증을구분한다. 추가다운로드/사용자자료/카메라/외부전송/Git없음.
'''
 (U/'PLAN_KO.txt').write_text(plan,'utf-8')
 state=read(O/'STATE.json');state.update(current_stage=U.name,last_updated=datetime.now(timezone(timedelta(hours=9))).isoformat(),next_single_action='정답감사→경계범위감독·실제왼손보강→고정학습비교→저장검산·선별적용');state[U.name]=dict(status='PREPARING',unit=str(U),private_data=0);write(O/'STATE.json',state)
 (ROOT/'NEW_CHAT_HANDOFF.md').write_text('# 최신 인수인계\n\n사용자 승인 후 정상완료 보완 실행중. 현재단위 '+str(U)+'\nPLAN_KO.txt의4단계: 감독모순분리/경계범위정답/끝점판정부비교/회귀·별도검증. 이전단위완료결과재실행금지. C·T3-F/공통100단어/웹판정보존. 개인자료금지.\n','utf-8')
 write(U/'CURRENT_EXECUTION.json',dict(status='PREPARING',pid=None));write(U/'SETUP_COMPLETE.json',dict(fit_annotations=len([r for r in labels if r['role']=='fit']),added_left_events=sum(len(r['events']) for r in additions),new_videos=0))
 event('setup',added_left_events=sum(len(r['events']) for r in additions))
if __name__=='__main__':main()
