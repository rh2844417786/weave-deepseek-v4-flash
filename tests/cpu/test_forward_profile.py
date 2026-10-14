import unittest
from types import SimpleNamespace

try:
    from src.weave.forward_profile import ForwardSampler
except ImportError:
    ForwardSampler=None


class Event:
    def __init__(self,clock): self.clock=clock;self.time=None
    def record(self): self.time=self.clock[0]
    def elapsed_time(self,other): return (other.time-self.time)*1000


class ForwardProfileTest(unittest.TestCase):
    def test_sampling_is_bounded_and_preserves_actual_batch_and_host_gpu_boundaries(self):
        self.assertIsNotNone(ForwardSampler)
        clock=[0.]
        cuda=SimpleNamespace(Event=lambda **kw:Event(clock),is_current_stream_capturing=lambda:False)
        sampler=ForwardSampler(cuda,stride=2,capacity=2,clock=lambda:clock[0])
        for batch in [32,31,30,29,28]:
            token=sampler.begin('decode',batch,batch)
            clock[0]+=.01
            sampler.end(token)
            clock[0]+=.1
        report=sampler.snapshot()
        decode=report['phases']['decode']
        self.assertEqual(decode['calls'],5)
        self.assertEqual(decode['valid_rows_total'],150)
        self.assertEqual(decode['sampled_invocations'],3)
        self.assertEqual(decode['overwritten_samples'],1)
        self.assertEqual([x['sequence'] for x in decode['recent_samples']],[2,4])
        self.assertEqual([x['valid_rows'] for x in decode['recent_samples']],[30,28])
        self.assertAlmostEqual(decode['recent_samples'][0]['gpu_stream_ms'],10)
        self.assertAlmostEqual(decode['recent_samples'][0]['host_enqueue_ms'],10)
        sampler.reset()
        self.assertEqual(sampler.snapshot()['phases']['decode']['calls'],0)

    def test_unfinished_forward_and_padded_decode_cannot_claim_valid_timing(self):
        self.assertIsNotNone(ForwardSampler)
        cuda=SimpleNamespace(Event=lambda **kw:Event([0.]),is_current_stream_capturing=lambda:False)
        sampler=ForwardSampler(cuda,stride=1,capacity=1)
        with self.assertRaises(ValueError):sampler.begin('decode',32,31)
        sampler.begin('decode',32,32)
        with self.assertRaises(ValueError):sampler.snapshot()
