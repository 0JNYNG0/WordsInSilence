"""Deliberate in-memory corruption checks, saved public scores only."""
from common import *
import copy
from verify import recount

c=read(SRC/'PREP_CONTEXT.json');r=read(SRC/'C1_20261013_RESULT.json')
recount(r,c)
def wrong_prediction(z):z['predictions']['sensor'][0]['prediction']=(z['predictions']['sensor'][0]['prediction']+1)%100
def wrong_label(z):z['predictions']['sensor'][0]['label_index']=(z['predictions']['sensor'][0]['label_index']+1)%100
def wrong_total(z):z['metrics']['sensor']['top1_correct']+=1
def wrong_ce(z):z['selected_ce']+=.001
def missing_row(z):z['predictions']['masked'].pop()
cases=[]
for mutate in (wrong_prediction,wrong_label,wrong_total,wrong_ce,missing_row):
 z=copy.deepcopy(r);mutate(z)
 try:recount(z,c)
 except AssertionError:cases.append({'case':mutate.__name__,'rejected':True})
 else:raise AssertionError('Corruption not detected: '+mutate.__name__)
write(U/'VERIFIER_SELF_CHECK.json',{'status':'PASS','at':now(),'valid_saved_scores_pass':True,'corruptions':cases,'input_files_unchanged':True,'model_inference':0},new=True)
print('PASS: valid reference and 5 deliberately corrupted copies checked.')
