import copy
from pathlib import Path
import tempfile
import unittest

try:
    from src.weave.replay_benchmark import ReplayStore,point_key,paired_summary
except ImportError:
    ReplayStore=point_key=paired_summary=None


def point(variant='persistent_raw_no_probe'):
    return {'layer':0,'sample':0,'variant':variant,'gpu_ms':[2.,4.],
            'host_enqueue_ms':[.1,.2],'correctness':{'status':'passed','relative_l2':0.},
            'workload':{'pairs':6,'groups':3},'stage_gpu_ms':{},'warmups':1}


class ReplayBenchmarkTest(unittest.TestCase):
    def test_resume_requires_same_identity_and_reconciles_only_common_points(self):
        self.assertIsNotNone(ReplayStore)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'rank0.json';identity={'rank':0,'source_sha256':'a'*64,'repetitions':2}
            store=ReplayStore(path,identity)
            store.append(point());key=point_key(point())
            self.assertEqual(ReplayStore(path,identity).keys(),{key})
            with self.assertRaises(ValueError):ReplayStore(path,dict(identity,source_sha256='b'*64))
            store.reconcile(set())
            self.assertEqual(store.keys(),set())
            self.assertEqual(len(ReplayStore(path,identity).value['orphaned_points']),1)

    def test_bad_or_duplicate_measurement_cannot_be_published(self):
        self.assertIsNotNone(ReplayStore)
        with tempfile.TemporaryDirectory() as tmp:
            store=ReplayStore(Path(tmp)/'rank0.json',{'rank':0,'source_sha256':'a'*64,'repetitions':2})
            for mutation in ('nan','short','failed','raw'):
                bad=point()
                if mutation=='nan':bad['gpu_ms'][0]=float('nan')
                if mutation=='short':bad['gpu_ms']=[1.]
                if mutation=='failed':bad['correctness']['status']='failed'
                if mutation=='raw':bad['ids']=[12]
                with self.subTest(mutation=mutation),self.assertRaises(ValueError):store.append(bad)
            store.append(point())
            with self.assertRaises(ValueError):store.append(point())

    def test_summary_uses_matching_layer_sample_and_max_rank_duration(self):
        self.assertIsNotNone(paired_summary)
        ranks=[]
        for rank in range(4):
            a=point();b=point('persistent_bound_no_probe')
            a['gpu_ms']=[2.+rank,4.+rank];b['gpu_ms']=[1.+rank,2.+rank]
            ranks.append({'identity':{'rank':rank},'points':[a,b]})
        result=paired_summary(ranks)
        self.assertEqual(result['cases'][0]['variants']['persistent_raw_no_probe']['max_rank_median_gpu_ms'],6.)
        self.assertEqual(result['cases'][0]['variants']['persistent_bound_no_probe']['max_rank_median_gpu_ms'],4.5)
        self.assertFalse(result['used_for_formal_performance'])
        ranks[3]['points'].pop()
        with self.assertRaises(ValueError):paired_summary(ranks)
