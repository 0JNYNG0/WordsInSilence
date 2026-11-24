from pathlib import Path
D=Path(__file__).resolve().parent;N=D.parent/'frontal_scope_recovery_20261117'
s=(N/'thanks_runtime/worker.py').read_text('utf-8')
s=s.replace('from frontal_candidate import FrontalLesson as ThanksCandidate','from short_arrival import ShortArrivalLesson as ThanksCandidate')
(D/'thanks_runtime').mkdir(exist_ok=True)
(D/'thanks_runtime/worker.py').write_text(s,encoding='utf-8')
