from pathlib import Path
from prepare import read,write,sha
U=Path(__file__).resolve().parent
def main():
    assert not (U/'FROZEN.json').exists()
    assert not (U/'RUNNING.json').exists() and not (U/'RESULT_MANIFEST.json').exists()
    files=['AUTHORIZATION.json','PLAN_KO.txt','INPUTS.json','prepare.py','candidate.py','run.py','summarize.py','selftest.py','SELFTEST.json','review/METHOD_REVIEW.json','review/DATA_SCOPE_REVIEW.json','review/EVALUATION_LEDGER.json','review/PREPARATION_DATA_REVIEW.json','review/PREEXECUTION_REVIEW.json','review/IMPLEMENTATION_REVIEW.json']
    for n in files: assert (U/n).exists(),n
    st=read(U/'SELFTEST.json');assert st['status']=='PASS' and st['code_sha256']==sha(U/'candidate.py')
    old=U.parent/'word100_relational_sequence_compare_20261031/relational.py'
    pins={str((U/n).resolve()):sha(U/n) for n in files};pins[str(old.resolve())]=sha(old)
    write(U/'FROZEN.json',{'status':'FROZEN_BEFORE_PUBLIC_CANDIDATE_RESULTS','pins':pins,'candidate':'common ordered observed signed inter-hand wrist transition rectangles; no word conditions','scope':{'public_videos':16,'natural_pairs':12,'natural_panels':2,'retained_parent_intervals':39,'saved_controls':126,'new_controls':48},'primary':'stored original69/C13 paths versus identical cost with observed constraints','recovery':'stored partial303 overlay; new matched-input unconstrained E0 versus same-input constrained E1','new_controls':'16 each midpoint hold15 and same 2/3 copies with query speed075; all exact source and occurrence truth evaluator-only','gate':'all full axes, masks, old values preserved; no normal query/reference mean/max/peak/occurrence regression; no lost compared edited cells or short peak; at least one known-time total AND interior improvement; natural semantic time truth remains unavailable','no_training':True,'no_new_inference':True,'no_production_change':True,'all_panels_reported_no_favorable_selection':True,'on_failure':'Preserve first errors. Correct implementation bugs only. No word-specific threshold/weight search.'})
    print('FROZEN',sha(U/'FROZEN.json'))
if __name__=='__main__':main()
