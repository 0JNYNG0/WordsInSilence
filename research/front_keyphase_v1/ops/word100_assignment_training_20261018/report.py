"""Create a local Korean report from independently verified saved comparisons."""
from common import *
import html

def main():
 r=read(U/'COMPARISON_INDEPENDENT.json');assert r['status']=='PASS_SAVED_SCORES_AND_TRACES'
 s=r['summary'];passed=[a for a in ['P','Q'] if r['gates'][a+'_vs_C']['pass']]
 conclusion=('사전에 정한 C 대비 비교 기준을 통과한 후보: '+', '.join(passed)+'. 운영 적용은 별도 검토한다.') if passed else 'P와 Q 모두 사전에 정한 C 대비 개선 기준을 통과하지 못했다. 기존 C를 유지한다.'
 lines=['손 배정 보완 P/Q 학습·C 비교 결과',now(),'',conclusion,'',
 '적용 범위','특정 단어 이름·번호·촬영자별 예외를 넣지 않은 공통 관측 처리다. 화면 안 팔–손 연결, 추적 근거, 중복 손 후보를 처리하므로 다른 단어에도 적용할 수 있는 구조다. 다만 개발·검증한 범위는 공개100단어이며 모든 수어·사람·카메라에서 효과가 검증됐다는 뜻은 아니다. 단어 분류 모델의 어휘도 아직100종이다.','',
 '무엇을 비교했나','C: 기존 모델과 저장된3회 결과. 다시 학습하거나 재채점하지 않았다.','P: 좌표 기반 팔–손 연결·추적 재확인·관측과 좌우 확실성 분리·보호된 중복 배정 보류를 연결했다.','Q: P에 현재 화면 안 팔 윤곽을 이용한 좌우 확인을 추가했다. RGB를 단어 분류기의 새 특징으로 넣지 않았다.','모두 같은280채널 TCN,37636개 매개변수,같은3개 초기상태·학습순서·최대80회·검증기준. 공개 TRAIN899(9명)/DEV300(다른3명). 사용자·멘토·보류 사람·외부자료 사용0. 새 학습은P3회+Q3회.','',
 '입력 준비 결과','전체1199개 입력 준비·독립 수치 검산 완료. 필요한 입력을 만들지 못한 영상0.','TRAIN899의 두 관측 조건226362프레임에서 C 대비 P11/Q29프레임의 관측이 바뀌었다. 29개 전체 변경은 기존24개 검토 근거 재사용+신규5개 원본확인으로 검토했다. 기존 정상441건 모두 유지. 기하학적 교차 후보315개 중 이미 오배정으로 알려진 공항1개를 보류하고 나머지314개 유지. 315개 전부가 정상 정답 사례인 것은 아니다.','새 Q 정정5프레임: 무산 REAL01 1개,무산 REAL05 2개,후원 REAL02 1개,추출 REAL09 1개. 손을 복원한 것이 아니라 실제 관측된 손의 잘못된 좌우 칸을 정정했다.','P/Q 모두 기존 C에서 잘못 배정되거나 중복된177개 점/프레임 관측을 named 손 슬롯에서 제외했다. 이 수는 실제 손177개나 영상177개가 아니다. 원본 좌표는 보존했다. 좌우를 정하지 못해 슬롯에 넣지 않은 ROI 관측점은 각46282개이며 이를 삭제나 복원 성공으로 세지 않는다.','DEV는1프레임만 바뀌었고 P/Q 입력은 동일하다. 무산 REAL11의 실제 왼손 관측을 오른손으로 넣던 오류를 보류했으며 좌우를 새로 확정하지 않았다. 규칙을 DEV에 맞춰 수정하지 않았다.','동작 구간 시작·종료 변경0, 내부 누락 프레임·원래 시간간격 유지.','',
 '성능: 각각 DEV300,3회 평균과 범위','sensor = 실제로 잘린 화면만으로 검출하여 화면 밖 손을 볼 수 없는 조건.','masked = 전체 화면에서 검출하되 기존 상반신 범위 관절만 모델에 넣는 조건.','이 수치는 모델 선택에도 사용한 공개 개발 검증 정확도이며 실제 웹캠 사용자 정확도나 독립 최종시험이 아니다.','']
 table=[]
 for a in ['C','P','Q']:
  for v in ['sensor','masked']:
   x=s[a][v];delta=0 if a=='C' else x['delta_C_pp'];text=f"{a} / {v}: {x['top1_counts']} / 각300, 평균{x['accuracy_mean']:.2f}%, 범위{x['accuracy_range'][0]:.2f}~{x['accuracy_range'][1]:.2f}%, C 대비{delta:+.2f}%p"
   lines.append(text);table.append((a,v,x,delta))
 lines+=['','P는 C의 좌우 오배정2건 정정+9건 보류, Q는20건 정정+9건 보류다. 단위는 손 후보/프레임이며 단어 인식 정답 수와 다르다. 원래 보이지 않던 손을 복원한 사례는0이다.','변경 중 주석의 핵심 동작 구간에 들어가는 것은 P5/Q9프레임이다. 나머지는 핵심 구간 밖이며, 변경 수가 곧 동작 전체의 큰 변화라는 뜻은 아니다.','각3회는 학습을 시작할 때의 초기값을 달리한3번이다. 같은300개 영상을3번 평가했으므로900개 독립 영상이나900명 시험으로 해석하면 안 된다.','', '개선·회귀 판정']
 for a in ['P','Q']:
  failed=[k for k,v in r['gates'][a+'_vs_C']['checks'].items() if not v]
  lines.append(f"{a}: {'비교기준 통과' if not failed else '비교기준 미통과'}. 미통과 항목: {', '.join(failed) or '없음'}")
  for seed,x in r['paired'][a].items():
   for v,p in x.items():lines.append(f"  초기값 {seed}/{v}: C오답→정답 {len(p['improved'])}개, C정답→오답 {len(p['regressed'])}개, {p['delta_pp']:+.2f}%p")
 lines+=['','어떻게 해석해야 하나','입력 오류를 바로잡는 것과 단어 정답률 상승은 별도다. 수정이 필요한 프레임은 전체에 비해 적고, 손이 겹쳐 실제 손가락이 검출되지 않는 문제는 여전히 남아 있다. 또한 학습 결과는 작은 입력 변화에도 달라질 수 있어 좋은 한 번만 고르지 않고3회 모두 비교했다.','단어 종류는 고정되어 있으며 이번 작업으로 새로운 단어를 학습한 것은 아니다. 새 사람/새 촬영환경에서의 일반화를 이번 개발 점수만으로 주장하지 않는다.','',
 '남은 문제','좌우 보류 손의 원본 관절은 저장되지만 기존 좌/우 고정280채널에는 모두 활용하지 못한다. 양쪽에 복사하거나 Pose손목으로 손가락을 만들어 넣지 않았다.','손 겹침·장시간 가림·손목 위치 왜곡 자체는 해결되지 않았다. 팔 윤곽은 옷·배경·조명·해상도 영향을 받을 수 있고 실시간 웹캠 처리속도도 별도 미검증이다.','DTW·웹·기존 C는 변경하지 않았다. 이번 점수는 DTW 손모양 판정 정확도 향상을 의미하지 않는다.','',
 '검증과 최초 오류','별도 계산으로 TRAIN132505520/DEV23315600 채널값 일치, 팔윤곽 TRAIN1224/DEV294프레임 검산. 저장logit 정답수·Top5·교차엔트로피와 실제 배치별 원본ID/관측뷰/선택epoch/체크포인트 연결을 검산했다. 완성 모델의 결과 재추론0. 구현·시각검토·별도 수치검증은 같은 Codex가 수행했으며 독립 사람 평가가 아니다.','초기 JSON 기록에서 NumPy 정수 직렬화 오류가 있었다. 원본·최초 실패를 보존하고 JSON쓰기만 복구했다. 복구 기록의 초기0완료표기는1완료로 정정했고,8개 완료파일을 해시 그대로 복원했다. 재생성 배열도 동일함을 검산했다. 모델·계수·데이터·결과에 맞춘 조건 변경은 없었다.','',
 '다음 단계','이 결과를 사용자와 검토한다. C는 유지하며 새 후보를 자동 적용하지 않는다. 추가 학습은 이번 점수에 맞춰 반복하지 않는다. 새 실험이 필요하면 남은 실제 관측 손실과 새 촬영환경 검증을 분리해 계획부터 정한다.']
 lines+=['','정규화 영향의 관찰','원래 C의 영상 단위 손 크기 정규화는 그대로 유지했다. 손 배정이 바뀌면서 P19/Q36개 손·구간 조합의 크기 기준이 달라졌다. 따라서 관측이 직접 바뀐11/29프레임보다 최종 숫자가 달라진 텐서 프레임은 P611/Q1108개로 많다(whole/context 및 관측조건별 중복 포함). 최대 손 크기 기준 변화는 Q에서약7.80%였다. 이는 입력 변화의 전달 경로를 확인한 결과이며 정확도 하락의 단독 원인으로 입증한 것은 아니다. 결과를 보고 정규화 기준을 다시 튜닝하지 않았다.']
 if (U/'TRAIN_STATUS_RECOVERY.json').exists():
  lines+=['','학습 실행 복구','첫 Q 실행63회 완료 후 STATUS.json 교체에서 Windows 접근 오류가 났다. 완료 체크포인트·옵티마이저·난수상태·63회 배치기록을 확인한 뒤64회부터 재개했다. 첫P완료결과는 재실행하지 않았다. STATUS쓰기 재시도만 별도wrapper에 추가했고 동결 학습·계수·자료·점수선택 조건은 유지했다. 최초오류와당시상태파일을보존했다.']
 (U/'RESULT_KO.txt').write_text('\n'.join(lines)+'\n','utf8')
 def esc(x):return html.escape(str(x))
 tablehtml=''.join(f"<tr><td>{a}</td><td>{v}</td><td>{'/'.join(map(str,x['top1_counts']))}</td><td><b>{x['accuracy_mean']:.2f}%</b></td><td>{x['accuracy_range'][0]:.2f}–{x['accuracy_range'][1]:.2f}%</td><td>{delta:+.2f}%p</td></tr>" for a,v,x,delta in table)
 details=''.join('<p>'+esc(line)+'</p>' for line in lines[3:] if line)
 page='<!doctype html><html lang="ko"><meta charset="utf-8"><title>P/Q 입력 보완·학습 비교</title><style>body{margin:0;background:#f3f5f8;color:#182638;font:17px/1.7 "Malgun Gothic",sans-serif}main{max-width:1080px;margin:auto;padding:44px 30px}h1{font-size:32px}article,section{background:white;border-radius:16px;padding:25px 32px;margin:20px 0}table{border-collapse:collapse;width:100%;font-size:16px}td,th{padding:13px;border-bottom:1px solid #e5e9ef;text-align:left}.tag{color:#166263}.lead{font-size:21px;font-weight:700}p{margin:10px 0}details{margin-top:24px}summary{cursor:pointer;font-weight:700}a{color:#1263b6}</style><main><div class="tag">WordsInSilence · 공개 자료 검증 · 기존 C 유지</div><h1>손 배정 보완을 학습에 연결한 결과</h1><article><p class="lead">'+esc(conclusion)+'</p><p>공통 규칙으로 구현했지만 검증 범위는100단어다. 사용자 촬영 자료는 사용하지 않았다.</p></article><section><h2>C · P · Q 비교</h2><p>동일한 공개 검증영상300개,각3회. 모델 선택용 개발 결과이며 실사용 정확도가 아니다.</p><table><thead><tr><th>후보</th><th>관측 조건</th><th>3회 정답 수</th><th>평균 정확도</th><th>범위</th><th>C 대비</th></tr></thead><tbody>'+tablehtml+'</tbody></table><p>sensor: 실제 잘린 화면의 미검출 포함 · masked: 전체검출 후 영역 제한</p></section><details open><summary>설명과 검증 근거</summary><article>'+details+'</article></details><p><a href="COMPARISON_INDEPENDENT.json">독립 검산 결과</a> · <a href="TRAIN_VISUAL_REVIEW.json">변경 장면 검토</a> · <a href="RESULT_KO.txt">설명 원문</a></p></main></html>'
 (U/'RESULT_KO.html').write_text(page,'utf8');print(conclusion)

if __name__=='__main__':main()
