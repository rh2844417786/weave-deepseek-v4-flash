"""Page-rounded admission accounting for the fixed B32 DeepSeek V4 probe.

This checks token and request limits, not free KV memory or numerical accuracy.
Actual B32 forward/replay checks remain the evidence of full cohort execution.
"""
FIXED_PAGE_SIZE = 256
FIXED_PREFILL_TOKENS = 16384
SERVER_FIELDS = ('page_size', 'chunked_prefill_size', 'max_prefill_tokens',
                 'max_running_requests', 'prefill_max_requests')


def _positive_int(value, name):
    if type(value) is not int or value <= 0:
        raise ValueError('positive integer required for ' + name)
    return value


def admission_budget(lengths, *, page_size, chunked_prefill_size, max_prefill_tokens):
    if not isinstance(lengths, (tuple, list)) or len(lengths) != 32:
        raise ValueError('fixed-prefix budget requires all 32 input lengths')
    for value in lengths:_positive_int(value, 'input length')
    for name, value in (('page_size', page_size), ('chunked_prefill_size', chunked_prefill_size), ('max_prefill_tokens', max_prefill_tokens)):
        _positive_int(value, name)
    rounded=[((length+page_size-1)//page_size)*page_size for length in lengths]
    cumulative=[];total=0
    for value in rounded:
        total+=value;cumulative.append(total)
    limit=min(chunked_prefill_size,max_prefill_tokens)
    return dict(request_count=32,raw_input_tokens=sum(lengths),page_size=page_size,
        page_rounded_input_tokens=total,page_rounded_prefix_tokens=cumulative,
        chunked_prefill_size=chunked_prefill_size,max_prefill_tokens=max_prefill_tokens,
        rows_within_prefill_token_budget=sum(value<=limit for value in cumulative),
        whole_cohort_fits_token_budget=total<=limit,
        scope='page_rounded_token_budget_only_actual_forward_rows_checked_separately',
        numerical_policy='record_only')


def effective_request_limits(server_args):
    states=server_args.get('internal_states') if isinstance(server_args,dict) else None
    if not isinstance(states,list) or not states:
        raise ValueError('live scheduler request limits missing from /server_info')
    limits=[]
    for state in states:
        if not isinstance(state,dict):raise ValueError('invalid live scheduler state')
        limits.append(_positive_int(state.get('effective_max_running_requests_per_dp'),'effective_max_running_requests_per_dp'))
    return limits


def validate_effective_budget(lengths, server_args):
    if not isinstance(server_args,dict) or any(key not in server_args for key in SERVER_FIELDS):
        raise ValueError('effective prefill budget fields missing from pinned /server_info')
    actual={key:server_args[key] for key in SERVER_FIELDS}
    if actual['page_size']!=FIXED_PAGE_SIZE or type(actual['page_size'])is not int:
        raise ValueError('DeepSeek V4 CUDA page size differs from pinned 256-token contract')
    proof=admission_budget(lengths,**{key:actual[key] for key in SERVER_FIELDS[:3]})
    if not proof['whole_cohort_fits_token_budget']:
        raise ValueError('fixed-prefix prefill token budget cannot admit all 32 rows: requires '+str(proof['page_rounded_input_tokens']))
    if _positive_int(actual['max_running_requests'],'max_running_requests')<32:
        raise ValueError('effective running request limit cannot admit 32 rows')
    if actual['prefill_max_requests'] is not None and _positive_int(actual['prefill_max_requests'],'prefill_max_requests')<32:
        raise ValueError('effective prefill request limit cannot admit 32 rows')
    worker_limits=effective_request_limits(server_args)
    if any(limit<32 for limit in worker_limits):
        raise ValueError('actual scheduler request limit cannot admit 32 rows')
    return dict(proof,status='passed',effective_max_running_requests_per_dp=worker_limits,max_running_requests=actual['max_running_requests'],
        prefill_max_requests=actual['prefill_max_requests'],configuration_source='live_server_info')
