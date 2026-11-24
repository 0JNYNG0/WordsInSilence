"""Add post-result interpretation, without touching frozen analysis or scores."""
from pathlib import Path
import json,html,hashlib
U=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text('utf8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 verify=read(U/'review/TRAIN_RESULTS_VERIFICATION.json');assert verify['status'].startswith('PASS')
 assert not (U/'REPORT_FINALIZED.json').exists()
 s=read(U/'SCORES.json');prior=read(U/'PREVIOUS_STORED_COMPARISON.json');d=read(U/'DIAGNOSTICS.json')
 lines=['준비·검증·6회 학습·기존 출력 비교를 완료했다. 새 판정부를 운영 모델로 채택하지 않는다.',
 '좋아진 점: 영상 정보를 더하면 개발 자료의 손내림·위치 유지·방향전환 판정이 좋아졌다. 10특징 평균은81.75%→86.33%(+4.58%p)였다.',
 '유지되지 않은 점: 추가4표현8영상에서는70.24%→68.97%(−1.26%p)였다. 새 표현 전체에서 이득이 있다는 근거는 없다. 이 묶음은 양성·음성 모두 있는7특징 평균이므로 개발10특징과 절대수치를 직접 비교하지 않는다.',
 '기존 방식과 공통 손모양3종만 비교하면 기존 관절+영상93.18%→이번85.15%(−8.03%p)였다. 관절만 조건은기존94.35%→이번94.27%로 비슷했다. 이 비교는 기존 저장 확률을 새 RGB 주석과 대조한 것이며 예전 성적표를 다시 쓴 것이 아니다.',
 '주의: 개발 자료에서 왼손의 네 손가락이 펴진 확정 양성은1구간뿐이다. 영상 추가 모델이3회 중2회 그1구간을 놓쳐 평균이 크게 흔들렸다. 큰 비율 차이만으로 모든 왼손 동작에 문제가 있다고 일반화하지 않는다.',
 '이번에 배운 것은 정해준 짧은 구간에서 손가락 펴짐·양손 높낮이·위치 유지/이동을 구분하는 것이다. 전체 수어의 순서와 필수 반복을 자동 이해하거나 사용자에게 잘못된 수어라고 교정하는 완성 단계는 아니다.',
 '정상 손·교차의 원본 좌표와 미검출 마스크, 기존 C·T3-F와 운영 DTW는 바꾸지 않았다. 별도의 정상/교차 정답 집합과 충분한 평가 분모는 없어 새 판정부의 교차 성능 보존을 입증했다고 말할 수 없다.',
 '정답의 한계: 같은 Codex가 만든 연구용 RGB 주석이다. 별도 에이전트 검토는 독립된 수어 전문가 검토가 아니다. 보류774셀을 틀린 동작으로 취급하지 않았다.',
 '다음 우선순위: 잘되던 손모양 판정부를 고정해 보호하고, 움직임·양손관계 판정부를 별도로 학습하는 비교를 설계한다. 왼손 펴기 등 드문 양성과 학습 지원 부족8특징을 다양한 공개 표현에서 보충한다. 이번 추가8영상은 이미 평가에 사용했으므로 다음 새 평가로 다시 부르지 않는다. 이는 다음 가설이며 아직 성능개선이 증명된 방법이 아니다.',
 '새 대형 모델·검출기 추가, 사용자 촬영 학습·튜닝, 운영 모델 교체는 없었다.']
 timing=d['encoder_preparation']['per_clip_seconds'];lines.append(f'작은 판정부만 계산하면 준비된3구간 기준 약0.28~0.43ms였다. 그러나 새8영상의 영상특징 준비에는 영상당{min(timing):.2f}~{max(timing):.2f}초가 기록됐다. 둘 다 녹화 종료부터 피드백까지의 전체 지연 측정은 아니다.')
 lines.append('검산 중 새8영상의 미정 중앙 프레임을 그대로 비교한 검산기 가정 오류를 수정하고 최초 기록을 보존했다. 진단 스크립트는 결과를 모두 저장한 뒤 마지막 화면 출력에서 json import 누락으로 종료1이 됐다. 완전 저장된 결과를 별도 검산했으며 학습·점수·타이밍은 반복 실행하지 않았다.')
 (U/'INTERPRETATION_KO.txt').write_text('\n\n'.join(lines),encoding='utf8')
 original=(U/'RESULT_KO.txt').read_text('utf8');(U/'RESULT_KO.txt').write_text('\n\n'.join(lines)+'\n\n'+original,encoding='utf8')
 intro='<section class="card" id="decision"><h2>결론: 일부 움직임은 개선, 전체 채택은 보류</h2>'+''.join('<p>'+html.escape(t)+'</p>' for t in lines)+'</section>'
 for name in ['index.html','RESULT_KO.html']:
  p=U/name;text=p.read_text('utf8');needle='<section class="card"><h2>원본 13프레임과 정답을 직접 확인</h2>';assert text.count(needle)==1;p.write_text(text.replace(needle,intro+needle),encoding='utf8')
 report=dict(status='COMPLETE_LIMITED_FEATURE_STUDY_NOT_ADOPTED',training_runs=6,main_files={n:sha(U/n) for n in ['SCORES.json','PREVIOUS_STORED_COMPARISON.json','DIAGNOSTICS.json','RESULT_KO.html','RESULT_KO.txt','INTERPRETATION_KO.txt','index.html','REPORT_DATA.json','finalize_report.py','review/TRAIN_RESULTS_VERIFICATION.json']},production_changed=False,private_data_used=False,normal_crossing_performance_not_measured=True)
 with (U/'REPORT_FINALIZED.json').open('x',encoding='utf8') as f:json.dump(report,f,indent=2,ensure_ascii=False)
 print(report['status'])
if __name__=='__main__':main()
