import copy
import unittest
from tests.cpu.test_graph_evidence import snapshots
from src.weave.fixed_prefix_evidence import validate_fixed_graph, normalize_fixed_launch

class FixedEvidenceTests(unittest.TestCase):
    def test_actual_events_and_buffers_not_enabled_flags(self):
        for arm,nodes,refs in [('native_a',0,0),('p0_full',1032,43),('p0_events',1032,0),('p0_buffers',0,43)]:
            a,b=snapshots()
            for row in b:
                row.update(successful_replays=9,raw_batch_histogram={'32':9},captured_batch_histogram={'32':9})
            if arm.startswith('p0'):
                for rows in (a,b):
                    for r in rows:r['stage_profile']=dict(kind='native_p0',profile_source_sha256='c'*64,rank=r['rank'],b32_seen=r['raw_batch_histogram']['32'],sampled_total=0,aborted_samples=0,intervention=dict(mode=arm.split('_')[1],readback=False,event_node_count=nodes,allocated_event_count=nodes,retained_grouping_layer_count=refs,retained_grouping_tensor_ref_count=refs*3),layer_metadata={str(i):{'captured_path':'forward_normal_dual_stream','captured_spans':(['s'+str(j) for j in range(12)] if nodes else [])} for i in range(43)})
            identity=dict(observer_source_sha256='b'*64,graph_sources={'p':'a'*64},native_p0_source_sha256='c'*64)
            self.assertEqual(validate_fixed_graph(a,b,arm=arm,steps=8,identity=identity)['replays'],7)
            if arm.startswith('p0'):
                b[0]['stage_profile']['intervention']['event_node_count']=99
                with self.assertRaises(ValueError):validate_fixed_graph(a,b,arm=arm,steps=8,identity=identity)

    def test_launch_normalization_keeps_compute_flags(self):
        a=['docker','--name','a','-e','WEAVE_FIXED_PREFIX_DIR=/work/a','--output-dir','/work/a','--tp-size','4']
        b=['docker','--name','b','-e','WEAVE_FIXED_PREFIX_DIR=/work/b','-e','WEAVE_NATIVE_P0=1','-e','WEAVE_NATIVE_P0_MODE=events','--output-dir','/work/b','--tp-size','4']
        self.assertEqual(normalize_fixed_launch(a),normalize_fixed_launch(b));b[-1]='2';self.assertNotEqual(normalize_fixed_launch(a),normalize_fixed_launch(b))

if __name__=='__main__':unittest.main()
