import json
from pathlib import Path
import tempfile
import unittest
from src.weave.experiment_export import export_run
from src.weave.service_acceptance import compare_first_token


class ExperimentExportTest(unittest.TestCase):
    def test_actual_native_diagnostics_export_only_aggregate_counts_and_deltas(self):
        def response(top):
            return {'input_sha256':'same', 'response':{'meta_info':{
                'output_token_logprobs':[[-.1,77,None]], 'output_top_logprobs':[top]}}}
        diagnostic=compare_first_token(response([[-.1,77],[-1.,88]]),
                                       response([[-.1,77],[-2.,99]]))
        diagnostic['matched_generated_tokens']=8
        with tempfile.TemporaryDirectory() as tmp:
            source=Path(tmp)/'run';source.mkdir();model=source/'model';model.mkdir()
            (source/'experiment.json').write_text(json.dumps({'status':'failed',
                'model_acceptance_dir':str(model)}))
            (model/'weave_eager-request.json').write_text(json.dumps({
                'status':'passed', 'native_comparison':diagnostic}))
            dest=Path(tmp)/'report';export_run(source,dest)
            public=json.loads((dest/'model/weave_eager-proof.json').read_text())['native_comparison']
            for key in ('token_id','weave_token_id','native_only_token_ids',
                        'weave_only_token_ids','common_logprob_deltas'):
                self.assertNotIn(key,public)
            self.assertEqual(public['native_only_count'],1)
            self.assertEqual(public['weave_only_count'],1)
            self.assertEqual(public['common_count'],1)
            self.assertEqual(public['matched_generated_tokens'],8)
            self.assertFalse(public['used_for_acceptance'])

    def test_token_level_field_is_rejected_even_outside_model_proof(self):
        with tempfile.TemporaryDirectory() as tmp:
            source=Path(tmp)/'run';source.mkdir();dest=Path(tmp)/'report'
            (source/'experiment.json').write_text(json.dumps({'status':'failed','token_id':77}))
            with self.assertRaisesRegex(ValueError,'token_id'): export_run(source,dest)
            self.assertFalse(dest.exists())

    def test_safe_partial_summary_exports_without_claiming_completion(self):
        with tempfile.TemporaryDirectory() as tmp:
            source=Path(tmp)/'run';source.mkdir()
            value={'status':'failed','performance':'not_run','steps':{},'scope':'formal_gpqa_budgeted_comparison'}
            (source/'experiment.json').write_text(json.dumps(value))
            dest=Path(tmp)/'report'
            manifest=export_run(source,dest)
            self.assertEqual(json.loads((dest/'experiment.json').read_text()),value)
            self.assertEqual(manifest['experiment_status'],'failed')
            self.assertFalse(manifest['raw_content_exported'])

    def test_decodable_payload_is_rejected_before_any_public_file_is_written(self):
        with tempfile.TemporaryDirectory() as tmp:
            source=Path(tmp)/'run';source.mkdir();dest=Path(tmp)/'report'
            (source/'experiment.json').write_text(json.dumps({'status':'failed','steps':{'output_ids':[1,2]}}))
            with self.assertRaisesRegex(ValueError,'output_ids'): export_run(source,dest)
            self.assertFalse(dest.exists())
