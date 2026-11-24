from ranking_core import *
from collections import defaultdict
d=read(K/'TARGET_DIAGNOSIS.json');groups=defaultdict(lambda:[0,0]);rows=[]
for r in d['rows']:
 person=int(r['clip_id'].split('REAL')[1].split('_')[0]);cohort='probe08_09'if person in[8,9]else'added_fit03_07'if person in[3,7]else'original_fit02_04_05_06';trained=r['split']=='train'and not(r['condition']=='rank_original'and person in[3,7]);rr=dict(r,source_cohort=cohort,seen_by_this_head=trained);rows.append(rr);key=(r['condition'],r['seed'],cohort,trained,r['view'],r['reverse'],r['target']);groups[key][0]+=r['compatible'];groups[key][1]+=1
sm=[dict(condition=k[0],seed=k[1],source_cohort=k[2],seen_by_this_head=k[3],view=k[4],reverse=k[5],target=k[6],correct=v[0],n=v[1])for k,v in groups.items()]
write(K/'TARGET_DIAGNOSIS_COHORT_CORRECTION.json',dict(at=stamp(),rows=rows,summary=sm,initial=str(K/'TARGET_DIAGNOSIS.json'),correction='Initial source split=train included added FIT03/07 even for rank_original, which never trained on them. Predictions unchanged; original/added/probe cohorts now separate.',no_inference=True,normal_component_changes=d['normal_component_changes']))
print('Source-cohort metadata separated; scores/predictions unchanged')
