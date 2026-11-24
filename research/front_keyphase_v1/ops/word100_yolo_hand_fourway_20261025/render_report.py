from pathlib import Path
import json,html,hashlib
from collections import Counter
import cv2,numpy as np
U=Path(__file__).resolve().parent;O=U.parent
def read(p):return json.loads(Path(p).read_text('utf-8-sig'))
def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
LINKS=[(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(5,9),(9,10),(10,11),(11,12),(9,13),(13,14),(14,15),(15,16),(13,17),(0,17),(17,18),(18,19),(19,20)]
NAMES={'M0':'기존 관측','M1':'MediaPipe 재검출','Y':'YOLO 손 단독','MY':'MediaPipe+YOLO 보조'}
def esc(x):return html.escape(str(x))
def main():
    summ=read(U/'SUMMARY.json');rows=read(U/'SCORED_ROWS.json');fr=read(U/'FROZEN.json');ind=read(U/'INDEX.json');notes=read(U/'FINAL_NOTES.json')
    by={(r['id'],r['condition']):r for r in rows};mr={r['clip_id']:r for r in fr['rows']};ix={r['clip_id']:r for r in ind}
    # Fixed known diagnostic examples, plus every newly supported/incorrect MY annotation.
    pick=['R01','R02','R13','R17','V05','V11','V12','V19']
    pick+=summ['paired']['MY']['new_supported_vs_M0']+summ['paired']['MY']['new_outside_or_ambiguous_vs_missing_M0']
    pick=list(dict.fromkeys(pick));(U/'visual').mkdir(exist_ok=True);cards=[]
    ra={r['sample_id']:r for r in read(O/'word100_recovery_image_benchmark_20261025/BLIND_REVIEW_A.json')['rows']}
    vb={r['sample_id']:r for r in read(O/'word100_temporal_refiner_pilot_20261025/VISIBLE_LABELS_FROZEN.json')['labels']}
    for sid in pick:
        if (sid,'M0') not in by:continue
        rr=by[(sid,'M0')];cid=rr['clip_id'];i=rr['frame_id'];s=rr['side'];r=mr[cid]
        z=load(r['arrays']);q=load(ix[cid]['arrays']);cap=cv2.VideoCapture(r['rgb_path']);cap.set(cv2.CAP_PROP_POS_FRAMES,i);ok,im=cap.read();cap.release();assert ok
        box=ra[sid]['wrist_box'] if sid.startswith('R') else vb[sid]['rectangle']
        if box is None:box=[700,400,1000,700]
        cx=(box[0]+box[2])/2;cy=(box[1]+box[3])/2
        x0=max(0,min(1920-600,int(cx-300)));y0=max(0,min(1080-420,int(cy-210)))
        panels=[]
        for c in NAMES:
            canvas=im.copy()
            if c in ('M0','M1'):
                pref='baseline_roi' if c=='M0' else 'redetected_roi';p=z[pref+'_xy_px'][i,33:].reshape(2,21,2);v=z[pref+'_usable'][i,33:].reshape(2,21)
            else:pref='y' if c=='Y' else 'my';p=q[pref+'_xy'][i];v=q[pref+'_valid'][i]
            for side,color in [(0,(70,220,40)),(1,(40,160,255))]:
                for a,b in LINKS:
                    if v[side,a] and v[side,b]:cv2.line(canvas,tuple(p[side,a].astype(int)),tuple(p[side,b].astype(int)),color,2)
                for j in np.flatnonzero(v[side]):cv2.circle(canvas,tuple(p[side,j].astype(int)),3,color,-1)
            cv2.rectangle(canvas,(box[0],box[1]),(box[2],box[3]),(255,220,50),2)
            cv2.line(canvas,(0,int(z['roi_box'][3])),(1919,int(z['roi_box'][3])),(0,220,220),2)
            tile=canvas[y0:y0+420,x0:x0+600].copy();tile=cv2.copyMakeBorder(tile,36,0,0,0,cv2.BORDER_CONSTANT,value=(25,30,38))
            cv2.putText(tile,c+' : '+by[(sid,c)]['status'],(12,25),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),1,cv2.LINE_AA);panels.append(tile)
        image=np.vstack([np.hstack(panels[:2]),np.hstack(panels[2:])]);path=U/'visual'/f'{sid}.jpg';cv2.imwrite(str(path),image)
        cards.append(f'<figure><img src="visual/{sid}.jpg"><figcaption>{sid} · {esc(r["display_label"])} · {i}프레임 · 대상 {"왼손" if s==0 else "오른손"}. 초록=왼손, 주황=오른손, 하늘색 상자=기존 영상 판독 범위.</figcaption></figure>')
    def table(group):
        ret='<table><tr><th>조건</th><th>범위 안</th><th>범위 밖</th><th>경계·판독 차이</th><th>미검출</th><th>판독 보류</th></tr>'
        for c,a in summ['wrist_image_consistency'][group].items():
            co=a['counts'];ret+=f'<tr><td>{NAMES[c]}</td>'+''.join(f'<td>{co.get(k,0)}</td>' for k in ('inside_both','outside_both','inside_one_ambiguous','unavailable','unscorable'))+'</tr>'
        return ret+'</table>'
    latency='<table><tr><th>조건</th><th>평균</th><th>중앙값</th><th>느린 5% 경계</th></tr>'
    for c,a in summ['latency_cpu_detector_components'].items():latency+=f'<tr><td>{NAMES[c]}</td><td>{a["mean_ms"]:.1f}ms</td><td>{a["median_ms"]:.1f}ms</td><td>{a["p95_ms"]:.1f}ms</td></tr>'
    latency+='</table>'
    bars=[]
    for c,a in summ['wrist_image_consistency']['R24'].items():
        val=a['supported_of_scorable_pct'];bars.append(f'<div class="barrow"><span>{NAMES[c]}</span><div class="rail"><div style="width:{val:.2f}%"></div></div><b>{a["counts"].get("inside_both",0)}/{a["scorable"]}</b></div>')
    text=f'''YOLO 손 보조 네 조건 비교 — 2026-10-25
결론: {notes['conclusion']}
핵심수치: {notes['metrics_line']}
남은문제: {' / '.join(notes['remaining'])}
다음작업: {notes['next']}
공개39영상·18표현·6명·4781프레임. 기존TRAIN01–06 진단이며 새사용자/전체수어 성능이 아님.
모델: NeuralVulture/yolo-hand-pose, YOLOv8n 손21점, commit {read(U/'MODEL_SOURCE.json')['commit']}.
설정: CPU4threads,640입력,boxconf0.25,keypointconf0.5,IoU0.7. 결과를 보고 변경하지 않음.
M0/M1은 기존 동결 관측 재사용, Y/MY는 전체 원영상에서 새 추론 후 같은 고정 상반신 범위 적용.
Y는 손 검출만 YOLO이며 해부학 좌우 연결에는 공통 MediaPipe 몸 좌표 사용. 손목 신뢰가 부족하면 좌우 보류.
MY는 기존 좌표 보존, 빠진 손만 원영상/팔 주변 확대 두 번 일치+팔 연결+중복배정 검사 후 추가.

손목 판독 R24: {json.dumps(summ['wrist_image_consistency']['R24'],ensure_ascii=False)}
이전 정상/교차/재등장 V18: {json.dumps(summ['wrist_image_consistency']['V18'],ensure_ascii=False)}
기존 대비 변화: {json.dumps(summ['paired'],ensure_ascii=False)}
실제 전체 영상의 새 추가 손: 원영상{summ['accepted_native_hand_frames']}손-프레임,ROI에남음{summ['accepted_roi_hand_frames']}.
보존: {json.dumps(summ['preservation'])},기존 교차 메타데이터{summ['crossing_metadata_frames']}프레임.
CPU 검출 부품 합산 시간: {json.dumps(summ['latency_cpu_detector_components'])}
시간은 같은 장비의 새 순차 호출을 측정하여 조건별 부품을 합산. 초기화/비디오디코딩/좌우규칙/정규화/TCN/DTW/웹제외. 실제 웹캠 FPS아님.
기존MP새실행과저장결과의손검출개수차이 {summ['fresh_saved_mp_hand_count_mismatches']}/4781; 정확도는저장관측을사용하며시간재측정을재현성/정확도검증으로혼동하지않음.
판독상자는 손목의 대략적인 영상 근거이며 전문가 손가락21점 정답이 아님. 미검출/판독불가는 성공이나0오차로계산하지않음.
새 학습0,TCN추론0,사용자·멘토자료접근0,기존C·T3-F/운영변경0. 이 시험만으로 손가락교정정확도나새사용자수어인식률향상주장불가.
담당: Codex구현/별도산술검산. 다른에이전트나사람독립검토로표현하지않음.
'''
    (U/'RESULT_KO.txt').write_text(text,encoding='utf8')
    page=f'''<!doctype html><html lang="ko"><meta charset="utf-8"><title>YOLO 손 보조 네 조건 비교</title><style>
body{{font-family:"Malgun Gothic",sans-serif;background:#edf2f7;color:#172437;margin:0;line-height:1.7}}main{{max-width:1100px;margin:auto;padding:36px}}section{{background:white;padding:28px;margin:22px 0;border-radius:16px}}h1{{font-size:32px}}h2{{font-size:23px}}.tag{{color:#425a7b}}table{{width:100%;border-collapse:collapse}}td,th{{padding:12px;border-bottom:1px solid #dde5ed;text-align:left}}th{{background:#eef4fa}}figure{{margin:24px 0}}img{{max-width:100%;border-radius:10px}}figcaption{{font-size:14px;color:#52637a}}.barrow{{display:grid;grid-template-columns:210px 1fr 60px;gap:12px;align-items:center;margin:12px 0}}.rail{{background:#e6edf6;height:24px;border-radius:8px}}.rail>div{{height:100%;background:#287eb2;border-radius:8px}}.call{{padding:18px;background:#fff4de;border-left:5px solid #d8992c}}pre{{white-space:pre-wrap;font-size:13px}}</style><main>
<p class="tag">공개 자료 · 관측 부품 비교 · 기존 C/T3-F 유지</p><h1>YOLO가 놓친 손을 더 정확하게 찾는가?</h1>
<p>39개 영상 · 18개 표현 · 6명 · 4,781프레임. 전체 원영상에서 검출하고, 마지막에 동일한 상반신 범위를 적용했습니다.</p>
<section><h2>결론: 이번 후보는 운영에 추가하지 않습니다</h2><p>{esc(notes['conclusion'])}</p><p class="call">기존 관측 대비 보조가 새로 확인한 손목 3장면 · 기존에 맞던 장면 손실 0 · MediaPipe 재검출 대비 평균 검출시간 약16% 증가</p><p>선정된 문제 표본의 진단입니다. 기존 MP의 누락·복원 상태를 기준으로 모은 장면이 포함되어 있어, 아래 비율을 평소 전체 손 검출률이나 두 모델의 일반적 우열로 해석하면 안 됩니다.</p></section>
<section><h2>네 조건</h2><p>① 기존 관측　② 기존 관측+MediaPipe 재검출　③ YOLO 손 검출　④ 기존 관측+YOLO 보조</p><p>YOLO 손 검출에는 좌우 라벨이 없어 공통 몸 관절로 손의 주인을 확인합니다. 보조 조건은 기존 손을 그대로 보존하고, 빠진 손에 대해서만 현재 팔 연결과 두 크기의 영상 검출이 일치할 때 추가합니다.</p></section>
<section><h2>실제 복원 문제 장면 24개</h2><p>23개는 손목 위치를 비교할 수 있었고 1개는 흐림 때문에 판독을 보류했습니다. 막대는 비교 가능한 23개 중 기존 영상 판독 범위에 들어온 수입니다.</p>{''.join(bars)}{table('R24')}<p class="call">이 수치는 손목의 대략적인 위치 근거입니다. 손가락 21개 관절의 정확도나 TCN 단어 인식률이 아닙니다.</p></section>
<section><h2>기존 정상·교차·재등장 표본 18개</h2>{table('V18')}<p>기존 판독 범위를 그대로 재사용했습니다. R24와 범위 작성 방식이 달라 각각 표시합니다.</p></section>
<section><h2>처리 시간</h2>{latency}<p>CPU, PyTorch4스레드. 같은 영상에 새로 호출한 검출 부품 시간을 조건별로 합산했습니다. 준비·디코딩·좌우 배정·TCN·DTW·웹 화면은 제외되어 실제 실시간 FPS로 해석하면 안 됩니다.</p></section>
<section><h2>보존과 남은 한계</h2><p>YOLO 보조에서 기존 관측 {summ['preservation']['MY_original_joint_slots']:,}개 관절-프레임을 그대로 보존했습니다. 기존 교차 메타데이터 {summ['crossing_metadata_frames']}프레임도 보존 검사했습니다. 이것은 좌표가 바뀌지 않았다는 뜻이며, 기존 좌표가 모두 정답이라는 뜻은 아닙니다.</p><p>정밀 손가락 정답, 새 사람 일반화, TCN/DTW의 성능 변화는 이번 시험에서 측정하지 않았습니다. 사용자 촬영은 접근하거나 학습·튜닝에 사용하지 않았습니다.</p></section>
<section><h2>실제 장면 비교</h2><p>아래는 결과를 보기 쉽게 확대한 화면입니다. <b>검출 입력은 잘라낸 이 화면이 아니라 전체 원영상</b>이며, 확대 재검출 좌표도 원영상 좌표로 되돌린 뒤 합쳤습니다.</p>{''.join(cards)}</section>
<section><h2>왜 보조의 추가 이득이 작았는가?</h2><p>MP 재검출이 놓치거나 범위를 벗어난 6장면에서 YOLO 손목은 판독 범위에 들어왔습니다. 그러나 보조 통합은 원영상·확대영상의 손바닥 관절 일치까지 요구하므로 5장면은 보류하고 1장면만 추가했습니다. 손 위치는 맞아도 손가락 모양까지 믿을 수 있는지는 별개의 문제라, 결과를 보고 기준을 낮추지 않았습니다.</p><p>새로 추가한 25개 손-프레임은 모두 원영상과 함께 확인했습니다. 큰 좌우 뒤집힘이나 빈 배경에 손이 생기는 현상은 보지 못했지만, 이 육안 검토를 정밀 손가락 정답으로 사용하지 않았습니다.</p><p>{esc(notes['next'])}</p></section>
<section><h2>상세 기록</h2><pre>{esc(text)}</pre><p><a href="VERIFY_SAVED.json">저장 수치 별도 검산</a> · <a href="FROZEN.json">실행 전 고정 조건</a> · <a href="https://github.com/NeuralVulture/yolo-hand-pose">모델 출처</a></p></section></main></html>'''
    (U/'RESULT_KO.html').write_text(page,encoding='utf8');print('REPORT',len(cards),'visuals')
if __name__=='__main__':main()
