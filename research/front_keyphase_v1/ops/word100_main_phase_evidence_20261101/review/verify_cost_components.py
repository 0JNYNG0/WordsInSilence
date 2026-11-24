"""Independent saved-link arithmetic; no producer import, DTW or inference."""
from pathlib import Path
from collections import Counter
import datetime
import hashlib
import json
import math
import numpy as np

U = Path(__file__).resolve().parents[1]
OPS = U.parent
GROUPS = ['L_shape','R_shape','L_location','R_location','L_axis','R_axis',
          'L_arm','R_arm','hands_relation','L_to_R_forearm','R_to_L_forearm',
          'L_direction','R_direction']


class Verification:
    def __init__(self):
        self.checks = Counter()
        self.failures = []
        self.pins = {}
        self.arrays = {}
        self.maximum_cost_error = 0.

    def check(self, category, condition, detail=''):
        self.checks[category] += 1
        if not bool(condition):
            self.failures.append(dict(category=category, detail=detail))

    def pin(self, path):
        p = Path(path).resolve()
        value = hashlib.sha256(p.read_bytes()).hexdigest()
        self.pins[str(p)] = value
        return value

    def read(self, path):
        self.pin(path)
        return json.loads(Path(path).read_text(encoding='utf-8-sig'))

    def array(self, alias, path):
        self.pin(path)
        with np.load(path, allow_pickle=False) as z:
            a = {k:z[k].copy() for k in ['x','mask','frame_id','timestamp_us']}
        self.check('array_shape', a['x'].shape == a['mask'].shape and a['x'].shape[1] == 69, alias)
        self.check('mask_finite_source_contract', np.array_equal(a['mask'], np.isfinite(a['x'])) or
                   np.all(~a['mask'] | np.isfinite(a['x'])), alias)
        self.arrays[alias] = a

    def equal(self, got, expected, label, motion=False):
        if expected is None:
            self.check('null_value', got is None, label)
        elif isinstance(expected, (bool,int,np.bool_,np.integer)):
            self.check('exact_integer_or_flag', got == expected, label)
        else:
            error = abs(float(got)-float(expected))
            if not motion:
                self.maximum_cost_error = max(self.maximum_cost_error,error)
            # Producer diagnostics use np.linalg.norm on saved float32 arrays;
            # independent math.hypot uses float64. Costs are computed as float64.
            tol = (2e-7 + 2e-6*abs(float(expected))) if motion else 2e-12
            self.check('motion_diagnostic' if motion else 'cost_arithmetic', error <= tol, label)


def frame_groups(a, i):
    """Scalar source equations, independent of relational.features."""
    raw = [float(v) for v in a['x'][i]]
    ok = [bool(m) and math.isfinite(v) for m,v in zip(a['mask'][i],raw)]
    out = {}
    def cols(ids, scale=1.):
        return ([raw[k]/scale if ok[k] else 0. for k in ids], [ok[k] for k in ids])
    for side,base in [('L',0),('R',28)]:
        other = 28-base
        out[side+'_shape'] = cols(list(range(base+11,base+25)))
        out[side+'_location'] = cols([base,base+1,base+8,base+9])
        out[side+'_axis'] = cols([base+25,base+26],2.)
        dx,dy = raw[base+2]-raw[base+4],raw[base+3]-raw[base+5]
        length = math.hypot(dx,dy)
        valid = all(ok[base+2:base+6]) and length > 1e-8
        values,masks = cols([base+4,base+5,base+10])
        values += [dx/length/2,dy/length/2,length] if valid else [0.,0.,0.]
        masks += [valid]*3
        out[side+'_arm'] = values,masks
        vx,vy = raw[other+2]-raw[other+4],raw[other+3]-raw[other+5]
        wx,wy = raw[base]-raw[other+4],raw[base+1]-raw[other+5]
        norm2 = vx*vx+vy*vy
        valid = all(ok[k] for k in [base,base+1,other+2,other+3,other+4,other+5]) and norm2 > 1e-8
        relation = [(wx*vx+wy*vy)/norm2,(vx*wy-vy*wx)/norm2] if valid else [0.,0.]
        out[side+'_to_'+('R' if side=='L' else 'L')+'_forearm'] = relation,[valid]*2
        out[side+'_direction'] = cols([63,64] if side=='L' else [67,68],2.)
    out['hands_relation'] = cols([56,57,58])
    return out


def cell_components(q, r):
    out = {}
    for name in GROUPS:
        x,knownq = q[name]
        y,knownr = r[name]
        expected = sum(knownr)
        shared = sum(a and b for a,b in zip(knownq,knownr))
        residual = sum(min(abs(a-b),1.) for a,b,ka,kb in zip(x,y,knownq,knownr) if ka and kb)
        missing = sum(b and not a for a,b in zip(knownq,knownr))
        observed_cost = residual/expected if expected else 0.
        missing_cost = missing/expected if expected else 0.
        empty = 0. if expected else .5
        out[name] = dict(expected=expected,shared=shared,observed=observed_cost,
                         missing=missing_cost,empty=empty,total=observed_cost+missing_cost+empty)
    return out


def motion(a, i):
    out = {}
    for side,off,vi in [('L',0,61),('R',28,65)]:
        velocity_known = bool(a['mask'][i,vi:vi+2].all())
        speed = math.hypot(*map(float,a['x'][i,vi:vi+2])) if velocity_known else None
        adjacent = i > 0 and int(a['frame_id'][i])-int(a['frame_id'][i-1]) == 1
        known_step = adjacent and bool(a['mask'][[i-1,i],off:off+2].all())
        # Match the producer's stored-coordinate float32 subtraction, while
        # computing its magnitude independently in scalar double precision.
        delta = a['x'][i,off:off+2]-a['x'][i-1,off:off+2] if known_step else None
        displacement = math.hypot(*map(float,delta)) if known_step else None
        out[side] = dict(speed_shoulder_per_second=speed,
                         consecutive_displacement_shoulder=displacement,
                         direct_hand_position_valid=bool(a['mask'][i,off:off+2].all()),
                         pose_wrist_valid=bool(a['mask'][i,off+2:off+4].all()))
    return out


def verify_cells(v, document):
    cache = {}
    for key,cell in document['cells'].items():
        a,b,i,j = cell['query'],cell['reference'],cell['q'],cell['r']
        v.check('cell_key',key==f'{a}_{i}_{b}_{j}',key)
        for alias,frame in [(a,i),(b,j)]:
            if (alias,frame) not in cache:
                cache[alias,frame] = frame_groups(v.arrays[alias],frame)
        expected = cell_components(cache[a,i],cache[b,j])
        v.check('all_13_groups', set(cell['components'])==set(GROUPS),key)
        for group,fields in expected.items():
            for name,value in fields.items():
                v.equal(cell['components'][group][name],value,f'{key}/{group}/{name}')
        geometry = sum(x['total'] for k,x in expected.items() if not k.endswith('_direction'))/13
        direction = sum(x['total'] for k,x in expected.items() if k.endswith('_direction'))/13
        for name,value in [('geometry11',geometry),('direction2',direction),('full13',geometry+direction)]:
            v.equal(cell[name],value,key+'/'+name)
        for alias,frame,field in [(a,i,'query_motion'),(b,j,'reference_motion')]:
            expected_motion = motion(v.arrays[alias],frame)
            for side,values in expected_motion.items():
                for name,value in values.items():
                    v.equal(cell[field][side][name],value,key+'/'+field+'/'+side+'/'+name,motion=True)


def verify_parent_rows(v,document,inputs,whole_counterparts=False):
    pairs={(p['query'],p['reference']):p for p in inputs['natural']}
    parents={p['id']:p for p in inputs['parents']}
    paths={}
    for pair,record in pairs.items():
        paths[pair]={name:v.read(record[field])['path'] for name,field in [('B','global_path'),('C','same_cost_path')]}
    v.check('all39parents', len(document['parents'])==39 and set(p['parent_id'] for p in document['parents'])==set(parents))
    used=set()
    for row in document['parents']:
        parent=parents[row['parent_id']]
        a,b=row['query'],row['reference'];lo,hi=parent['reference_range']
        selected={k:[[i,j] for i,j in p if lo<=j<=hi] for k,p in paths[a,b].items()}
        qids=sorted({i for p in selected.values() for i,j in p})
        v.check('main_reference_scope',row['reference_range']==[lo,hi],row['parent_id'])
        v.check('all_union_query_rows',[x['q'] for x in row['links']]==qids,row['parent_id'])
        for link in row['links']:
            i=link['q'];expected_links={}
            for condition,pp in (paths[a,b] if whole_counterparts else selected).items():
                expected_links[condition]=[f'{a}_{i}_{b}_{j}' for qi,j in pp if qi==i]
                v.check('exact_saved_links',link['saved_links'][condition]==expected_links[condition],f"{row['parent_id']}/{i}/{condition}")
                used.update(expected_links[condition])
            if all(expected_links.values()):
                expected_means={k:{name:sum(document['cells'][s][name] for s in ids)/len(ids)
                                      for name in ['geometry11','direction2','full13']}
                                for k,ids in expected_links.items()}
                for k,fields in expected_means.items():
                    for name,value in fields.items():v.equal(link['local_component_mean'][k][name],value,f"{row['parent_id']}/{i}/{k}/{name}")
                df=expected_means['B']['full13']-expected_means['C']['full13']
                dg=expected_means['B']['geometry11']-expected_means['C']['geometry11']
                v.check('ordering_flip',link['direction_changes_local_ordering']==bool(df*dg<0),f"{row['parent_id']}/{i}")
            else:
                v.check('one_sided_comparison_null',link['local_component_mean'] is None and link['direction_changes_local_ordering'] is None)
    v.check('unique_cells_exact_union',set(document['cells'])==used)
    if 'summary' in document:
        expected=[dict(parent_id=p['parent_id'],q_rows=len(p['links']),
                       both_conditions_rows=sum(x['local_component_mean'] is not None for x in p['links']),
                       local_ordering_flip_rows=sum(x['direction_changes_local_ordering'] is True for x in p['links']))
                  for p in document['parents']]
        v.check('summary_exact',document['summary']==expected)


def main():
    v=Verification()
    scope=v.read(U/'COST_AUDIT_SCOPE.json')
    for p,h in scope['pins'].items():v.check('frozen_source_pin',v.pin(p)==h,p)
    inp=v.read(OPS/'word100_phase_correspondence_20261101/INPUTS.json')
    for alias,row in inp['inputs'].items():v.array(alias,row['original'])
    document=v.read(U/'COST_COMPONENTS.json')
    v.check('scope_hash',document['scope_sha256']==v.pin(U/'COST_AUDIT_SCOPE.json'))
    verify_cells(v,document)
    verify_parent_rows(v,document,inp)
    result=dict(status='PASS' if not v.failures else 'FAIL',at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                checks=dict(v.checks),check_total=sum(v.checks.values()),failures=v.failures,
                unique_cells=len(document['cells']),parents=len(document['parents']),
                local_ordering_flip_rows=sum(s['local_ordering_flip_rows'] for s in document['summary']),
                maximum_float64_cost_error=v.maximum_cost_error,
                source_pins=v.pins,verifier_sha256=v.pin(Path(__file__)),
                method='Independent scalar69 feature equations and saved-link accounting; no producer feature/cost/DTW/model imported or called.',
                scope='Initial main-reference-core diagnostic only; counterpart outside scope is absent by construction, not a negative result.',
                supplement_verified=False,new_paths=0,new_inference=0,old_scores_changed=False,global_causation_proven=False)
    out=U/'review/COST_COMPONENT_VERIFICATION.json'
    with out.open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps({k:result[k] for k in ['status','check_total','unique_cells','parents','local_ordering_flip_rows','maximum_float64_cost_error']}))


if __name__=='__main__':main()
