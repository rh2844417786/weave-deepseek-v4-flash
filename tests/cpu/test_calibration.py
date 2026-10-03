"""Calibration identity, measured arithmetic and durable resume gates (CPU)."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

try:
    from src.weave import calibration as cal
except ImportError:
    cal = None


IDENTITY = {'kernel': 'tiled_mxfp4_tile_v1', 'source_sha256': 'a'*64,
            'device_key': 'H100;sm90;132', 'hidden_size': 4096,
            'intermediate_size': 2048, 'experts': 256, 'top_k': 6,
            'rank': 0, 'workers': 132, 'ep_size': 4,
            'checkpoint_content_sha256': 'b'*64, 'image_id': 'sha256:'+'c'*64,
            'runtime': {'python': '3.12.3', 'torch': '2.11.0+cu130', 'cuda': '13.0'},
            'measurement': {'buckets': [1,64,256,1024,2048],
                            'repetitions': 3, 'warmups': 1}}


def profile():
    # The required grid is literal, independent of the production generator.
    buckets = []
    for tokens in [2048,1024,256,64,1]:
        ks = [1] if tokens == 1 else [1,2,4,8]
        buckets.append({'max_local_tokens': tokens, 'candidates': [
            [c,k,200.,0. if c == 0 else 300.,.5,0. if c == 0 else .25]
            for c in [0,1,4,8,16,32,64,131] for k in ks]})
    return dict(copy.deepcopy(IDENTITY), schema_version=1, status='measured',
                raw_timings={'file': 'timings-rank0.json', 'sha256': 'd'*64,
                             'count': 374, 'source': 'cuda_event'}, buckets=buckets)


def point(tokens=1, c=0, k=1, mode='compute', samples=None):
    remote = mode != 'compute'
    groups = 6*sum((tokens//k + (j < tokens%k) + 15)//16 for j in range(k))
    tiles = groups*96
    return {'tokens': tokens, 'c': c, 'K': k, 'mode': mode,
            'samples_ms': samples or [1.,2.,3.], 'warmups': 1,
            'workload': [0 if remote else 6*tokens, 6*tokens if remote else 0,
                         tokens if remote else 0, tokens, 6*tokens, groups, tiles],
            'tile_counters': [tiles,0], 'completed_before_deadline': True}


class CalibrationTest(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(cal, 'measured calibration module is missing')

    def validate(self, bundle):
        return cal.validate_profile_bundle(bundle, **{k: IDENTITY[k] for k in
            ['kernel','source_sha256','device_key','hidden_size','intermediate_size','rank','workers']})

    def test_profile_normalizes_order_and_rejects_stale_or_missing_required_data(self):
        rows = self.validate(profile())
        self.assertEqual([b['max_local_tokens'] for b in rows], [1,64,256,1024,2048])
        self.assertEqual(rows[0]['candidates'][0], [0,1,200.,0.,.5,0.])
        mutations = [lambda b: b.update(status='partial'), lambda b: b.update(schema_version=True), lambda b: b.update(rank=1),
                     lambda b: b.update(source_sha256='e'*64),
                     lambda b: b.update(hidden_size=2048),
                     lambda b: b.update(checkpoint_content_sha256=''),
                     lambda b: b.update(raw_timings={}), lambda b: b.update(raw_timings=[]),
                     lambda b: b.update(buckets={}),
                     lambda b: b['buckets'].pop(0),
                     lambda b: b['buckets'][0]['candidates'].pop(),
                     lambda b: b['buckets'][0]['candidates'].append(b['buckets'][0]['candidates'][0])]
        for mutate in mutations:
            bundle = profile()
            mutate(bundle)
            with self.subTest(bundle=bundle.keys()), self.assertRaises(ValueError):
                self.validate(bundle)

    def test_profile_rejects_nonfinite_invalid_ranges_and_noninteger_plans(self):
        for column, bad in [(0,True),(0,132),(1,1.),(1,9),(2,0.),(2,float('nan')),
                            (3,1.),(4,0.),(4,1.01),(5,1.),(5,float('inf'))]:
            bundle = profile()
            bundle['buckets'][0]['candidates'][0][column] = bad
            with self.subTest(column=column,bad=bad), self.assertRaises(ValueError):
                self.validate(bundle)
        bundle = profile()
        bundle['buckets'][-1]['candidates'][1][3] = 0.
        with self.assertRaises(ValueError):
            self.validate(bundle)

    def test_grid_clips_workers_and_never_requests_remote_c0_modes(self):
        cases = cal.required_cases(5, [1,64])
        self.assertEqual(len(cases), 35)  # 3 c; 5 K total; c0=1 mode, other=3.
        self.assertEqual({(v['c'],v['K']) for v in cases if v['tokens']==1}, {(0,1),(1,1),(4,1)})
        self.assertFalse(any(v['c']==0 and v['mode']!='compute' for v in cases))
        self.assertEqual({v['K'] for v in cases if v['tokens']==64}, {1,2,4,8})

    def test_work_units_use_deduplicated_dispatch_and_fp32_return_bytes(self):
        p = point(tokens=2,c=1,mode='communication')
        units = cal.work_units(p,4096,2048)
        self.assertEqual(units['logical_flops'], 603979776)
        # Two valid rows still execute six complete M16 expert matrix tiles.
        self.assertEqual(units['padded_flops'], 4831838208)
        self.assertEqual(units['dispatch_bytes'], 16384)
        self.assertEqual(units['return_bytes'], 196608)
        self.assertEqual(units['remote_bytes'], 212992)
        p['workload'][2] = 3
        with self.assertRaises(ValueError):
            cal.work_units(p,4096,2048)

    def test_six_distinct_experts_cannot_claim_one_group(self):
        p=point()
        p['workload'][5:]=[1,96]
        p['tile_counters']=[96,0]
        with self.assertRaisesRegex(ValueError,'group'):
            cal.work_units(p,4096,2048)

    def test_overlap_is_conservative_and_records_clipping(self):
        fit = cal.fit_overlap([10.,12.],[5.,6.],[14.,15.],return_fraction=1.,chunks=1)
        # Fast primitive measurements give 10+5, slow mixed gives 15 -> overlap zero.
        self.assertEqual(fit['overlap'], 0.)
        fit = cal.fit_overlap([10.],[5.],[12.],return_fraction=1.,chunks=1)
        self.assertAlmostEqual(fit['overlap'], .6)
        fit = cal.fit_overlap([10.],[5.],[17.],return_fraction=1.,chunks=1)
        self.assertEqual(fit['overlap'], 0.)
        self.assertEqual(fit['clipped'], 'lower')
        fit = cal.fit_overlap([10.],[5.],[4.],return_fraction=1.,chunks=1)
        self.assertEqual(fit['overlap'], 0.)  # comm-dominated data cannot identify overlap.
        self.assertTrue(fit['unidentifiable'])
        fit = cal.fit_overlap([30.],[5.],[10.],return_fraction=1.,chunks=1)
        self.assertLess(fit['overlap'],1.)
        self.assertGreater(fit['overlap'],.999)
        self.assertEqual(fit['clipped'],'upper')

    def test_resume_rejects_identity_changes_corruption_and_late_or_duplicate_points(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'timings-rank0.json'
            session = cal.TimingStore(path, IDENTITY)
            session.append(point(), elapsed_seconds=1199., budget_seconds=1200.)
            resumed = cal.TimingStore(path, IDENTITY, resume=True)
            self.assertEqual(len(resumed.points),1)
            with self.assertRaises(ValueError):
                resumed.append(point(),elapsed_seconds=1.,budget_seconds=1200.)
            with self.assertRaises(TimeoutError):
                resumed.append(point(c=1),elapsed_seconds=1200.01,budget_seconds=1200.)
            self.assertEqual(len(cal.TimingStore(path,IDENTITY,resume=True).points),1)
            for field in ['source_sha256','checkpoint_content_sha256','device_key','image_id','rank','workers','runtime','measurement']:
                changed = copy.deepcopy(IDENTITY)
                changed[field] = 'different'
                with self.subTest(field=field), self.assertRaises(ValueError):
                    cal.TimingStore(path,changed,resume=True)
            with self.assertRaises(FileExistsError):
                cal.TimingStore(path,IDENTITY)
            raw = json.loads(path.read_text())
            raw['points'][0]['samples_ms'][0] = 999.
            path.write_text(json.dumps(raw))
            with self.assertRaisesRegex(ValueError,'hash'):
                cal.TimingStore(path,IDENTITY,resume=True)

    def test_banks_require_every_timing_and_derive_rates_from_actual_samples(self):
        identity = copy.deepcopy(IDENTITY)
        identity['workers'] = 2
        identity['device_key'] = 'H100;sm90;2'
        with tempfile.TemporaryDirectory() as folder:
            store = cal.TimingStore(Path(folder)/'timings-rank0.json',identity)
            with self.assertRaisesRegex(ValueError,'incomplete'):
                cal.build_profile_bundle(store)
            for case in cal.required_cases(2):
                # Hand chosen CPU fixture, never marked as actual server measurements.
                item = point(case['tokens'],case['c'],case['K'],case['mode'],
                             [10.,10.,10.] if case['K']==1 else [20.,20.,20.])
                store.append(item,elapsed_seconds=1.,budget_seconds=1200.)
            bank = cal.build_profile_bundle(store)
            a,b = bank['buckets'][1]['candidates'][:2]
            self.assertEqual(a[:2],[0,1])
            self.assertEqual(b[:2],[0,2])
            self.assertAlmostEqual(a[2],1932735283200.)
            self.assertEqual(a[4],1.)
            self.assertEqual(b[4],.5)
            self.assertEqual(a[3],0.)
            raw = (Path(folder)/'timings-rank0.json').read_bytes()
            self.assertEqual(bank['raw_timings']['sha256'],hashlib.sha256(raw).hexdigest())

    def test_interrupted_driver_persists_partial_and_does_not_publish_late_samples(self):
        with tempfile.TemporaryDirectory() as folder:
            store = cal.TimingStore(Path(folder)/'timings-rank0.json',IDENTITY)
            checks = iter([False,False,False,True])
            def measure(case):
                return point(case['tokens'],case['c'],case['K'],case['mode'])
            result = cal.run_calibration(store,measure,lambda:next(checks),lambda:1.,1200.)
            self.assertEqual(result['status'],'partial')
            saved = json.loads(store.path.read_text())
            self.assertEqual(len(saved['points']),1)
            self.assertEqual(saved['status'],'partial')
            self.assertEqual(saved['excluded_points'][0]['samples_ms'],[1.,2.,3.])
            self.assertFalse(saved['excluded_points'][0]['completed_before_deadline'])
            self.assertFalse((store.path.parent/'profile-bank-rank0.json').exists())
            # A signal arriving after events also excludes that just-finished case.
            resumed = cal.TimingStore(store.path,IDENTITY,resume=True)
            self.assertEqual(len(resumed.points),1)

    def test_rank_resume_intersection_preserves_orphaned_samples_for_audit(self):
        with tempfile.TemporaryDirectory() as folder:
            store = cal.TimingStore(Path(folder)/'timings-rank0.json',IDENTITY)
            store.append(point(),elapsed_seconds=1.,budget_seconds=1200.)
            store.reconcile_completed(set())
            self.assertEqual(store.points,[])
            raw = json.loads(store.path.read_text())
            self.assertEqual(raw['orphaned_points'][0]['samples_ms'],[1.,2.,3.])
            resumed = cal.TimingStore(store.path,IDENTITY,resume=True)
            self.assertEqual(resumed.orphaned_points,raw['orphaned_points'])

    def test_invalid_raw_points_and_excluded_hashes_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            store = cal.TimingStore(Path(folder)/'timings-rank0.json',IDENTITY)
            for mutate in [lambda p:p.update(samples_ms=[1.,2.]),
                           lambda p:p.update(samples_ms=[1.,float('inf'),3.]),
                           lambda p:p.update(completed_before_deadline=False),
                           lambda p:p.update(tile_counters=[0,0]),
                           lambda p:p.update(mode='mixed'),
                           lambda p:p.update(c=True),
                           lambda p:p.update(K=1.),
                           lambda p:p.update(tokens=1.)]:
                p = point()
                mutate(p)
                with self.assertRaises(ValueError):
                    store.append(p,elapsed_seconds=1.,budget_seconds=1200.)
            incomplete = point(c=1,mode='mixed',samples=[1.])
            store.record_excluded(incomplete,reason='deadline')
            raw = json.loads(store.path.read_text())
            raw['excluded_points'][0]['samples_ms'][0] = 2.
            store.path.write_text(json.dumps(raw))
            with self.assertRaisesRegex(ValueError,'hash'):
                cal.TimingStore(store.path,IDENTITY,resume=True)

    def test_cli_parser_rejects_overbudget_and_core_gate_rejects_absent_evidence(self):
        try:
            from scripts import calibrate_weave as cli
        except ImportError:
            cli = None
        self.assertIsNotNone(cli,'CUDA calibration command is missing')
        with self.assertRaises(SystemExit), patch('sys.stderr'):
            cli.parse_args(['--model-path','/model','--output-dir','/work/out',
                            '--core-run-dir','/work/core','--budget-seconds','1201'])
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ValueError,'core evidence'):
                cli.validate_core_evidence(Path(folder),Path(folder),'a'*64,{'resolved_image_id':IDENTITY['image_id']})

    def test_profile_loader_verifies_raw_hash_and_rederives_rates_without_rewriting(self):
        identity = copy.deepcopy(IDENTITY)
        identity['workers'],identity['device_key'] = 2,'H100;sm90;2'
        with tempfile.TemporaryDirectory() as folder:
            store = cal.TimingStore(Path(folder)/'timings-rank0.json',identity)
            for case in cal.required_cases(2):
                store.append(point(case['tokens'],case['c'],case['K'],case['mode']),
                             elapsed_seconds=1.,budget_seconds=1200.)
            bank = cal.build_profile_bundle(store)
            path = Path(folder)/'profile-bank-rank0.json'
            path.write_text(json.dumps(bank))
            kwargs = {k:identity[k] for k in ['kernel','source_sha256','device_key','hidden_size','intermediate_size','rank','workers']}
            before = (store.path.read_bytes(),store.path.stat().st_mtime_ns)
            rows = cal.load_verified_profile_bundle(path,**kwargs)
            self.assertEqual(rows[0]['max_local_tokens'],1)
            self.assertEqual((store.path.read_bytes(),store.path.stat().st_mtime_ns),before)
            bank['buckets'][0]['candidates'][0][2] *= 2
            path.write_text(json.dumps(bank))
            with self.assertRaisesRegex(ValueError,'derived'):
                cal.load_verified_profile_bundle(path,**kwargs)
            bank['buckets'][0]['candidates'][0][2] /= 2
            path.write_text(json.dumps(bank))
            store.path.write_bytes(before[0]+b' ')
            with self.assertRaisesRegex(ValueError,'hash'):
                cal.load_verified_profile_bundle(path,**kwargs)
            store.path.unlink()
            with self.assertRaises(ValueError):
                cal.load_verified_profile_bundle(path,**kwargs)
            self.assertFalse(store.path.exists())

    def test_core_evidence_binds_original_files_source_image_and_full_three_kernel_gate(self):
        from scripts.calibrate_weave import validate_core_evidence
        from src.weave.checkpoint_identity import checkpoint_identity
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            model,core = root/'model',root/'core'
            model.mkdir()
            core.mkdir()
            (model/'config.json').write_text('{}')
            (model/'model.safetensors.index.json').write_text(json.dumps({'weight_map':{'a':'weights.safetensors'}}))
            (model/'weights.safetensors').write_bytes(b'original fixture contents')
            identity = checkpoint_identity(model)
            (core/'checkpoint_identity.json').write_text(json.dumps(identity))
            runtime = {'runtime_status':'passed','checkpoint_validation_status':'passed',
                       'core_validation_status':'passed','child_exit_code':0,
                       'resolved_image_id':IDENTITY['image_id'],'kernel_source_sha256':'a'*64}
            (core/'runtime.json').write_text(json.dumps(runtime))
            cases = [{'layer':layer,'kernel':kernel,'case':case,'completed_stolen':1,
                      'metrics':{'relative_l2':.001},'workload':[32,0,0,32,32,2,192],
                      'tile_counters':[192,1]}
                     for layer in [-1]+list(range(43)) for kernel in ['scalar','wmma','tiled']
                     for case in ['remote_K2','steal_K2','graph_replay0','graph_replay1']]
            cases += [{'layer':-1,'kernel':'tiled','case':f'grouped_rows_K{k}',
                       'metrics':{'relative_l2':.001},'workload':[32,0,0,32,32,2,192],
                       'tile_counters':[192,1]} for k in [1,2,4,8]]
            for rank in range(4):
                report = {'status':'checkpoint_checks_passed','rank':rank,'layers':list(range(43)),
                          'kernels':['scalar','wmma','tiled'],'architecture':[4096,2048,256,6,43],
                          'source_sha256':'a'*64,'checkpoint_index_sha256':identity['files_sha256']['model.safetensors.index.json'],
                          'checkpoint_content_sha256':identity['sha256'],'cases':cases}
                (core/f'checkpoint-rank{rank}.json').write_text(json.dumps(report))
            gate,used = validate_core_evidence(core,model,'a'*64,{'resolved_image_id':IDENTITY['image_id']})
            self.assertEqual(gate['checkpoint_content_sha256'],identity['sha256'])
            self.assertEqual(used,identity)
            with self.assertRaises(ValueError):
                validate_core_evidence(core,model,'e'*64,{'resolved_image_id':IDENTITY['image_id']})
            with self.assertRaises(ValueError):
                validate_core_evidence(core,model,'a'*64,{'resolved_image_id':'sha256:'+'e'*64})
            (model/'weights.safetensors').write_bytes(b'changed checkpoint')
            with self.assertRaisesRegex(ValueError,'checkpoint files differ'):
                validate_core_evidence(core,model,'a'*64,{'resolved_image_id':IDENTITY['image_id']})


if __name__ == '__main__':
    unittest.main()
