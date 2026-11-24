"""Follow the frozen T3-F training-context code hashes, without model execution."""
from common import *

def main():
 ctx=read(U/'EVAL_CONTEXT.json');check(ctx['pins'])
 old=read(T/'TRAIN_CONTEXT.json')
 pins={p:h for p,h in old['pins'].items() if p.endswith('.py')}
 model_path=str(ROOT/'research/sign_model_pilot_v1/src/pilot/model_masked_v1.py')
 assert model_path in pins
 check(pins)
 write(U/'MODEL_SOURCE_AUDIT.json',{'at':now(),'status':'PASS_ORIGINAL_TRAINING_SOURCE_UNCHANGED','pins':pins,'parent_context_sha256':sha(T/'TRAIN_CONTEXT.json'),'training':0,'inference':0,'note':'Additional transitive check before heldout inference; no frozen input, model or evaluation code edited.'},new=True)
 print('PASS',len(pins),'original training-source hashes, including MaskedSignTCN')

if __name__=='__main__':main()
