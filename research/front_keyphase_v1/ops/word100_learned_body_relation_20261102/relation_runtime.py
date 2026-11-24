from pathlib import Path
from functools import lru_cache
import sys,time,torch
V=Path(__file__).resolve().parent;O=V.parent;P=O/'word100_learned_guide_feedback_20261102';U=O/'word100_learned_feature_retention_20261102'
sys.path.insert(0,str(U))
from retention_engine import RetentionEngine
from relation_model import RelationNet,read,dense
from evaluate_relations import attach
class RelationsEngine(RetentionEngine):
 def __init__(self,condition='rgb_geometry'):
  super().__init__(42);assert condition in ['rgb_geometry','geometry_only'];self.relation_condition=condition
  self.relations=RelationNet(condition=='rgb_geometry');self.relations.load_state_dict(torch.load(V/'runs'/f'{condition}_42/model.pt',map_location='cpu',weights_only=True));self.relations.eval();self.relations.requires_grad_(False)
  self.relation_stats=read(V/'NORMALIZATION.json');self.relation_support=read(V/'GUIDE_RELATION_SUPPORT.json')
 def infer(self,arrays,word):
  p=super().infer(arrays,word);t=time.perf_counter();p['relation_probability']=dense(self.relations,arrays,self.relation_stats);p['relation_head_seconds']=time.perf_counter()-t;return p
 def compare(self,prediction,arrays,word,calibration=False):
  assert not calibration
  base=super().compare(prediction,arrays,word);out=attach(base,arrays,prediction['relation_probability'],self.relation_support)
  out.update(relation_condition=self.relation_condition,relation_seed=42,relation_head_seconds=prediction['relation_head_seconds'],
   relation_scope='3 physical relations, trained on public people01-09; not touch, whole movement, or educational correctness')
  return out
