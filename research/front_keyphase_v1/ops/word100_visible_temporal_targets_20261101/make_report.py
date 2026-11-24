from pathlib import Path
import json,html
from experiment import U,read,write,sha
KO={f'{h}_{k}':f'{side} {text}' for h,side in [('R','오른손'),('L','왼손')] for k,text in [('index_only','검지만 펴짐'),('open_four','네 긴 손가락 펴짐'),('closed_four','네 긴 손가락 굽힘'),('higher','다른 손보다 높음'),('motion_up','위로 이동'),('motion_down','아래로 이동'),('motion_horizontal','가로 이동'),('motion_still','위치 유지'),('motion_mixed','방향 전환·복합 이동')]}
ROLE={'fit':'학습','development':'다른 촬영자 개발 비교','new_feature_expression_evaluation':'추가 표현 비교'}
def main():
 info=read(U/'DATASET.json');audit=read(U/'LABEL_AUDIT.json');inputs=read(U/'INPUTS.json')['rows'];by={r['alias']:r for r in inputs};vi={r['sample_id']:r for r in read(U/'VISUAL_INDEX.json')['samples']}
 scores=read(U/'SCORES.json') if (U/'SCORES.json').exists() else None;diag=read(U/'DIAGNOSTICS.json') if (U/'DIAGNOSTICS.json').exists() else None
 probs={} if diag is None else {(r['sample_id'],r['condition']):r['targets'] for r in diag['details']}
 items=[]
 def asset(p):return Path(p).resolve().relative_to(U).as_posix()
 for r in info['rows']:
  v=vi[r['sample_id']];a=by[r['alias']]
  items.append(dict(id=r['sample_id'],label=a['label'],role=ROLE[r['role']],person=r['person'],center=r['center_frame'],frames=[dict(id=f['frame_id'],time=f['seconds'],url=asset(f['file'])) for f in v['frames']],sheet=asset(v['sheet']),box=a['box'],size=v['source_image_size'],targets=[dict(key=t,name=KO[t],truth=int(r['y'][j]) if r['target_mask'][j] else None,reason=r['reasons'][t],trainable=t in info['selected_targets'],pose=probs.get((r['sample_id'],'pose_only'),{}).get(t),rgb=probs.get((r['sample_id'],'pose_plus_signrep'),{}).get(t)) for j,t in enumerate(info['targets'])],notes=r['primary_note'],cross_review=r['secondary_review']))
 if scores:
  text=['공개 영상의 시간별 특징 학습·비교 완료','', '공개56영상 / 168개 구간 / 각구간13연속원본프레임. 학습36영상108구간, 다른 촬영자 개발12영상36구간, 추가표현8영상24구간.','정답은 영상에 직접 보이는 특징을 Codex가 기록한 연구용 주석이다. 한국수어 전문가 정답/교정 기준은 아니다.','학습 지원 특징: '+', '.join(KO[t] for t in info['selected_targets']), '','균형정확도: 예/아니오를 각각 맞힌 비율의 평균. 단어정답률이 아니다.']
  for s in scores['summary']:
   if s['role']=='fit':continue
   value='산출불가' if s['mean'] is None else f"{100*s['mean']:.2f}% (3회범위 {100*s['range'][0]:.2f}~{100*s['range'][1]:.2f}%)"
   text.append(f"{ROLE[s['role']]} / {'관절만' if s['condition']=='pose_only' else '관절+영상'}: {value}, 두정답종류가있는특징{s['balanced_supported_targets']}개, 평가가능셀{s['known_cells']}개")
  text.append('')
  for role in ['development','new_feature_expression_evaluation']:
   a=next(s for s in scores['summary'] if s['role']==role and s['condition']=='pose_only');b=next(s for s in scores['summary'] if s['role']==role and s['condition']=='pose_plus_signrep')
   if a['mean'] is not None and b['mean'] is not None:text.append(f"{ROLE[role]}에서 영상정보 추가의 차이: {(b['mean']-a['mean'])*100:+.2f}%p. 두 비교 묶음의 지원특징 수가 다르면 묶음 사이 수치를 직접 비교하지 않는다.")
  text += ['', '같은 시간창의 작은 판정부를 각3회, 총6회 학습. 기존SignRep본체는동결, 사용자자료없음.','원본 관측좌표/마스크/정상교차를 수정하지 않았다. 새판정부의 정상손·교차 성능이 모든 영상에서 좋아졌다는 보장은 별도다.','정해준 짧은 구간에서 특징을 맞히는 시험이다. 전체영상을 자동으로 나누거나 올바른 수어인지 판정하는 완성품은 아니다.']
 else:
  text=['시간별 연구용 정답 준비·검증 결과: 아직 저장된 학습 비교 결과 없음','준비 상태: '+audit['status'],'공개56영상168구간의 13연속 원본프레임을 검토했다.','지원이 부족한 특징과 교차검토 보류를 보고한다.']
 cats=read(U/'review/CATEGORY_AGREEMENT.json')['groups']
 text += ['',f"학습지원 {len(info['selected_targets'])}/18특징. 교차검토 확정끼리 불일치 {audit['known_cross_conflicts']}/{audit['known_cross_comparisons']}개 ({audit['known_conflict_rate']*100:.2f}%). 불일치는 정답에서 보류.",f"위 비율에는 같은 움직임을 여러 예/아니오 질문으로 센 셀이 포함된다. 원래 분류 단위에서는 움직임 {cats['motion_category']['disagree']}/{cats['motion_category']['both_known']}, 양손 높낮이 {cats['bilateral_category']['disagree']}/{cats['bilateral_category']['both_known']}가 불일치했다. 손모양 확정끼리 {cats['shape_bit']['disagree']}/{cats['shape_bit']['both_known']}가 불일치했다.",'관절 시간창은13프레임이지만, 영상 특징은 기존SignRep의16프레임 창·보간을 재사용하므로 더 바깥 문맥도 포함된다.','기존 C·T3-F / 운영 DTW·웹 / 개인촬영 / 기존동결결과는 보존. 이 화면은 읽기전용 연구보고서.']
 text += ['', '남은 일: 보류된 특징의 더 다양한 공개표본과 교차검토를 확보하고, 자동 단계 구분과 설명에 연결한다. 관측 불확실성을 정답/오답으로 바꾸지 않는다.']
 (U/'RESULT_KO.txt').write_text('\n'.join(text),encoding='utf8')
 prior=read(U/'PREVIOUS_STORED_COMPARISON.json') if (U/'PREVIOUS_STORED_COMPARISON.json').exists() else None
 data=dict(title='손 모양·양손 관계·짧은 움직임: 시간별 정답과 비교',text=text,audit=audit,scores=scores,previous=prior,latencies=[] if diag is None else diag['latencies'],targets_ko=KO,items=items)
 write(U/'REPORT_DATA.json',data)
 template=(U/'report_template.html').read_text('utf8');embedded=json.dumps(data,ensure_ascii=False).replace('</','<\\/')
 page=template.replace('__DATA__',embedded)
 for name in ['RESULT_KO.html','index.html']:(U/name).write_text(page,encoding='utf8')
 print(json.dumps(dict(samples=len(items),trained=bool(scores),selected=info['selected_targets']),ensure_ascii=False))
if __name__=='__main__':main()
