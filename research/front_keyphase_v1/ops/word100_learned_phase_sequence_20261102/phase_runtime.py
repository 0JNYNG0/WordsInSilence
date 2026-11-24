"""Native-time phase evidence supplements, never overwrites, existing feedback."""
import copy,time,torch,numpy as np
from phase_model import *
from sequence_evidence import observe

class PhaseObserver:
    def __init__(self):
        self.model=PhaseNet('temporal_readout');path=S/'runs/temporal_readout_42/model.pt'
        assert sha(path)==read(path.parent/'COMPLETE.json')['sha256'];self.model.load_state_dict(torch.load(path,map_location='cpu',weights_only=True));self.model.eval();self.model.requires_grad_(False)
        self.graph=read(S/'LEARNED_STATE_GRAPH.json')['graphs'];self.groups=read(S/'PHYSICAL_STATE_TARGETS.json')['groups'];self.guides=read(P/'GUIDE_SUPPORT.json')['guides']
    @torch.inference_mode()
    def attach(self,result,arrays):
        out=copy.deepcopy(result);word=result['word_id'];a=arrays;n=len(a['latent']);start=time.perf_counter()
        row=dict(x=np.asarray(a['latent']),y=np.full(n,-1,np.int64),word_id=word);x,l,w,_=batch([row]);p=self.model(x,l,w).softmax(-1)[0].numpy()
        scope=sorted(set(i for s in out['stages'] for i in s['frame_ids']));scope=np.arange(min(scope),max(scope)+1)
        available=np.asarray(a.get('image_available',np.ones(n,bool)));times=a['timestamp_us']/1e6
        evidence=observe(p,available,scope,self.graph[word],times)
        for state in evidence['states']:
            g=self.groups[state['group']];state['description']=g['description_ko'];gs=[self.guides[k] for k in g['old_stage_aliases']]
            state['guide_ids']=[z['canonical_stage_id'] for z in gs];state['source_phrases']=sorted({c['text'] for z in gs for c in z.get('source_cards',[])})
            state['description_provenance']='source-linked visual configuration, not a claim that every linked phrase applies at each instant'
        for edge in evidence['precedences']:
            edge['before_description']=self.groups[edge['before']]['description_ko'];edge['after_description']=self.groups[edge['after']]['description_ko']
        evidence['head_seconds']=time.perf_counter()-start;evidence['classifier_conditioned_on_goal']=True
        evidence['independent_observers_unchanged']=True;evidence['automatic_main_window']='Preserved DTW provides coarse main bounds only; state ordering is measured on native time, not the DTW path.'
        evidence['limitations']=['Only9broad physical configurations in4supported expressions.','Some source stages are transitions; source descriptions are not dense labels.','A target-conditioned phase prediction alone cannot establish a handshape or whole-word educational error.','No fixed repetition count or exact contact/depth judgment.']
        out['learned_state_sequence']=evidence;out['whole_sign_correctness']=None
        assert out['path']==result['path'] and out['stages']==result['stages'] and out['motion_observations']==result['motion_observations']
        return out
