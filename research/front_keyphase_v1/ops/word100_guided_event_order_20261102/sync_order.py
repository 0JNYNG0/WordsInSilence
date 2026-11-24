from pathlib import Path
import sys
U=Path(__file__).resolve().parent
sys.path.insert(0,str(U.parent/'word100_learned_flow_role_20261102'))
import sync_flow
sync_flow.U=U
if __name__=='__main__':
    assert sync_flow.read(U/'INPUTS_VERIFIED.json')['known_queries']==576
    sync_flow.sync('가이드 행동발생 앞뒤 관계 지도입력 준비·검산',
        'V2학습176비교와Flow1학습88비교완료. Flow로맞은주요제외9→7이나88전체상태변화0/정상순서6·짧은역순1유지라미채택. 역할미세조정반복을멈추고가이드행동발생·앞뒤관계를직접학습할공통판정부로연결. 공개FIT9ID의27원본(3순서지원표현),변형및질문교환630중576관계확인/54혼재보류를독립코드로검산.4번째교대표현의기존관측/역할입력도유지하고순서정답은발명하지않음. PLAN.json준비,새관계신경망은아직학습하지않음.',
        'word100_guided_event_order_20261102/PLAN.json을따라공통가이드질의별발생신경망과앞뒤관계손실을구현·사전검증하고,조건/숫자통과기준을평가전동결한다. physical actor동결/DTW힌트는정답아님/같은영상A→B와B→A질의교환/짧은반대근거평균으로삭제금지. 발생지도만대조 vs직접순서손실추가2조건(동일80epoch42)계획,새모델만88조건씩1회. 완료Flow88/V176/L/J/K/F/E/D/U재실행없음. 입력576확인/54보류,사용자·멘토·H14/15금지. 실제데모전체연결·전체속도검증전완료아님. 현재실행중프로세스없음.',
        'GUIDED_EVENT_ORDER_INPUTS_VERIFIED',None)
