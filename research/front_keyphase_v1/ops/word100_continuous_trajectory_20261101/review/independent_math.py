"""Independent verification mathematics; never used by the candidate runner.

LP enumerates cell chains and optimizes crossing points, instead of using a
free-space interval dynamic program. Rational decision checks floating bounds.
Only synthetic/public stored geometry is allowed by the caller.
"""
from fractions import Fraction as F
import itertools
import numpy as np
from scipy.optimize import linprog


def rational_curve(a):
    return [[F(float(v)) for v in row] for row in np.asarray(a)]


def edge_interval(start, end, point, epsilon):
    low, high = F(0), F(1)
    for a, b, p in zip(start, end, point):
        slope, intercept = b-a, a-p
        if not slope:
            if abs(intercept) > epsilon:
                return None
            continue
        x, y = (-epsilon-intercept)/slope, (epsilon-intercept)/slope
        low, high = max(low, min(x, y)), min(high, max(x, y))
        if low > high:
            return None
    return (low, high)


def rational_decision(p, q, epsilon):
    """Exact feasibility for the binary-float coordinates and epsilon supplied.

All free edge intervals are constructed before propagation. The orientation
uses p as horizontal axis, q as vertical axis; no producer import occurs.
"""
    p, q = rational_curve(p), rational_curve(q)
    e = epsilon if isinstance(epsilon, F) else F(float(epsilon))
    n, m = len(p), len(q)
    if not n or not m or e < 0:
        return False
    if max(abs(a-b) for a,b in zip(p[0],q[0])) > e or max(abs(a-b) for a,b in zip(p[-1],q[-1])) > e:
        return False
    if n == 1:
        return all(max(abs(a-b) for a,b in zip(p[0],v)) <= e for v in q)
    if m == 1:
        return all(max(abs(a-b) for a,b in zip(q[0],v)) <= e for v in p)
    vertical = {(i,j):edge_interval(q[j],q[j+1],p[i],e) for i in range(n) for j in range(m-1)}
    horizontal = {(i,j):edge_interval(p[i],p[i+1],q[j],e) for i in range(n-1) for j in range(m)}
    rv, rh = {}, {}
    can_reach_vertex = True
    for j in range(m-1):
        free = vertical[0,j]
        rv[0,j] = free if can_reach_vertex and free is not None and free[0] == 0 else None
        can_reach_vertex = rv[0,j] is not None and rv[0,j][1] == 1
    can_reach_vertex = True
    for i in range(n-1):
        free = horizontal[i,0]
        rh[i,0] = free if can_reach_vertex and free is not None and free[0] == 0 else None
        can_reach_vertex = rh[i,0] is not None and rh[i,0][1] == 1
    def suffix(free, threshold):
        if free is None or threshold is None:
            return None
        lo = max(free[0],threshold)
        return (lo,free[1]) if lo <= free[1] else None
    for j in range(m-1):
        for i in range(n-1):
            left,bottom=rv[i,j],rh[i,j]
            rv[i+1,j]=suffix(vertical[i+1,j],F(0) if bottom is not None else left[0] if left is not None else None)
            rh[i,j+1]=suffix(horizontal[i,j+1],F(0) if left is not None else bottom[0] if bottom is not None else None)
    return (rv[n-1,m-2] is not None and rv[n-1,m-2][1] == 1) or (rh[n-2,m-1] is not None and rh[n-2,m-1][1] == 1)


def cell_chains(n, m):
    """Crossings between consecutive cells, including vertex crossings twice."""
    def walk(i,j,events):
        if i == n-2 and j == m-2:
            yield events
        if i < n-2:
            yield from walk(i+1,j,events+[('V',i,j)])
        if j < m-2:
            yield from walk(i,j+1,events+[('H',i,j)])
    yield from walk(0,0,[])


def chain_lp(p,q,events):
    p,q=np.asarray(p,float),np.asarray(q,float)
    k=len(events);variables=k+1;A=[];b=[]
    objective=np.zeros(variables);objective[-1]=1
    # A parameter coordinate is an affine vector (coefficients,constant).
    zero=np.zeros(variables)
    previous=((zero.copy(),0.),(zero.copy(),0.))
    endpoint=max(float(np.max(np.abs(p[0]-q[0]))),float(np.max(np.abs(p[-1]-q[-1]))))
    for t,(axis,i,j) in enumerate(events):
        coeff=np.zeros(variables);coeff[t]=1
        if axis=='V':
            delta=p[i+1]-q[j];slope=-(q[j+1]-q[j])
            position=((zero.copy(),float(i+1)),(coeff,float(j)))
        else:
            delta=p[i]-q[j+1];slope=p[i+1]-p[i]
            position=((coeff,float(i)),(zero.copy(),float(j+1)))
        for value,direction in zip(delta,slope):
            row=np.zeros(variables);row[t]=direction;row[-1]=-1
            A.append(row);b.append(-float(value))
            row=np.zeros(variables);row[t]=-direction;row[-1]=-1
            A.append(row);b.append(float(value))
        for (pc,pv),(nc,nv) in zip(previous,position):
            A.append(pc-nc);b.append(nv-pv)
        previous=position
    ending=((zero.copy(),float(len(p)-1)),(zero.copy(),float(len(q)-1)))
    for (pc,pv),(nc,nv) in zip(previous,ending):
        A.append(pc-nc);b.append(nv-pv)
    result=linprog(objective,A_ub=np.asarray(A),b_ub=np.asarray(b),bounds=[(0,1)]*k+[(endpoint,None)],method='highs',options={'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9})
    if not result.success:
        raise RuntimeError('Cell-chain LP failed: '+result.message)
    return float(result.fun)


def independent_lp_distance(p,q):
    p,q=np.asarray(p,float),np.asarray(q,float)
    if len(p)==1 or len(q)==1:
        return float(np.max(np.abs(p[:,None,:]-q[None,:,:]))),1
    values=[chain_lp(p,q,chain) for chain in cell_chains(len(p),len(q))]
    return min(values),len(values)


def discrete_distance(p,q):
    """Independent rolling-row bottleneck recurrence; bound only, not producer."""
    p,q=np.asarray(p,float),np.asarray(q,float)
    previous=np.full(len(q),np.inf)
    for i,x in enumerate(p):
        current=np.full(len(q),np.inf)
        for j,y in enumerate(q):
            cost=float(np.max(np.abs(x-y)))
            prior=0. if i==0 and j==0 else min(previous[j] if i else np.inf,current[j-1] if j else np.inf,previous[j-1] if i and j else np.inf)
            current[j]=max(cost,prior)
        previous=current
    return float(previous[-1])


def subdivide(curve, parts=4):
    curve=np.asarray(curve,float)
    if len(curve)==1:return curve.copy()
    return np.vstack([curve[i]+(curve[i+1]-curve[i])*a/parts for i in range(len(curve)-1) for a in range(parts)]+[curve[-1]])


def discrete_enclosure(p,q,parts=4):
    a,b=subdivide(p,parts),subdivide(q,parts)
    upper=discrete_distance(a,b)
    la=float(np.max(np.abs(np.diff(a,axis=0)))) if len(a)>1 else 0.
    lb=float(np.max(np.abs(np.diff(b,axis=0)))) if len(b)>1 else 0.
    # Conservative rounding of both parameters to sampled vertices costs at
    # most the sum of maximum individual edge lengths.
    return max(0.,upper-la-lb),upper

