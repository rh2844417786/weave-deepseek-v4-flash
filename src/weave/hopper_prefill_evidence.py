"""Torch-free source and complete-matrix validation for prefill GPU evidence."""
import math

SYNTHETIC=(("b1_spread",1),("b33_empty",33),("b65_skewed",65),("b129_spread",129),
           ("b8035_spread",8035),("b8192_skewed",8192),("b1_shrink",1))

def expected_cases():
    return ([{'case':name,'layer':-1,'batch':batch} for name,batch in SYNTHETIC]
            +[{'case':'real_b129','layer':layer,'batch':129} for layer in range(43)]
            +[{'case':'real_large','layer':0,'batch':batch} for batch in (8035,8192)])

def validate_prefill_reports(reports,**identity):
    if len(reports)!=4 or {r.get('rank') for r in reports}!={0,1,2,3}:
        raise ValueError('four distinct prefill ranks required')
    expected={(x['case'],x['layer'],x['batch']) for x in expected_cases()}
    for report in reports:
        if (report.get('status')!='passed' or report.get('world_size')!=4 or report.get('numerical_policy')!='record_only'
                or any(report.get(k)!=v for k,v in identity.items())):
            raise ValueError('prefill source/checkpoint/image/runtime identity differs')
        cases=report.get('cases',[])
        if len(cases)!=len(expected) or {(x.get('case'),x.get('layer'),x.get('batch')) for x in cases}!=expected:
            raise ValueError('prefill declared matrix incomplete')
        for case in cases:
            if any(case.get(k) is not True for k in ('metadata_exact','finite','assembly_exact','stable_addresses','binding_alive')) or case.get('additional_slot_collectives')!=1:
                raise ValueError('prefill structural/runtime evidence incomplete')
            metrics=case.get('numerical',{})
            if set(metrics)!={'old_fp8','fp8_oracle','bf16_oracle','bf16_reference'}:raise ValueError('old FP8 and independent FP8/BF16/reference records required')
            for value in metrics.values():
                if value.get('finite') is not True or any(type(value.get(k)) not in (int,float) or not math.isfinite(value[k]) or value[k]<0 for k in ('relative_l2','max_abs')):
                    raise ValueError('prefill nonfinite or missing numeric record')
    return {'status':'passed','numerical_policy':'record_only','ranks':4,'cases_per_rank':len(expected),'identity':identity}
