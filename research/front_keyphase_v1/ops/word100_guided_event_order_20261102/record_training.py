import sync_order
from sync_order import U,sync_flow
if __name__=='__main__':
    assert sync_flow.read(U/'TRAIN_PROGRESS.json')['status']=='TRAINING'
    sync_flow.sync('가이드별 행동발생·앞뒤 관계 신경망 2조건 학습',
        '가이드질의별발생관련성신경망 구현·사전검증통과. 기존raw actor고정,Source guide키고정,zero-init새질의부와역할층만학습. 같은영상질의교환시관측불변/순서확률상보,패딩불변,54모호관계순서손실제외,긴정상중1짧은반대근거는높은평균으로정상화금지검사. FIT252입력·576확인질의로발생지도만/직접순서손실추가2조건80epoch진행. 새성능미확인/기존웹미적용.',
        'train_order.py session16076 실제세션/TRAIN_PROGRESS확인후중복금지. 완료후evaluate_order.py새2모델x88조건176비교한번,verify_order.py별도저장배열검산. 결과기준FROZEN에고정:raw589 exact·주요유지>=582와대조·가족별유지·정상순서>6및대조·짧은역순>=2·거짓정상/정상역순0·누락보류. 정확대기vs준비구분은진단/대기를주요로오인은별도. 완료Flow88/V176/L/J/K/F/E/D/U재실행금지. C/T3F/기존웹/사용자/멘토/H14·15보존. 전체goal active.',
        'GUIDED_EVENT_ORDER_TRAINING',16076)
