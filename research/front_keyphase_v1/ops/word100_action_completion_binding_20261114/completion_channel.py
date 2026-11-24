"""Research-only channel separation; a completion score cannot hide an action.

This is not enabled in the web. It preserves localization/attribute inputs and
adds non-decisional diagnostics from an unadopted candidate model.
"""
import copy
def attach_extent_channel(located,extent_candidate):
    out=copy.deepcopy(located)
    candidates={c['card']:c for c in extent_candidate['cards']}
    assert len(candidates)==len(extent_candidate['cards'])
    for c in out['cards']:
        alternative=candidates.get(c['card'])
        matching=alternative is not None and alternative.get('family')==c.get('family')
        choice=alternative.get('chosen')if matching else None
        c['extent_assessment']=dict(
            state='research_extent_evidence'if choice else'insufficient_extent_evidence',
            diagnostic_span=choice['span']if choice else None,
            binding_verified=False,semantic_completion=None,educational_error=None,
            may_suppress_comparison=False,may_replace_localized_span=False,
            reference_scope='reviewed source extent; not required user duration',
            display_enabled=False)
    out['completion_channel_version']='research_isolation_v1'
    return out
