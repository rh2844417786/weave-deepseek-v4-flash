"""Build tools may mutate flag lists; the attested source manifest must not."""
import copy
import sys
import types
import unittest
from unittest.mock import patch
from src.weave import hopper_persistent as loader


class PaperLoaderTests(unittest.TestCase):
    def setUp(self):
        loader.extension.cache_clear()
        self.addCleanup(loader.extension.cache_clear)
        torch=types.ModuleType('torch');torch.__version__='2.11.0+cu130'
        torch.version=types.SimpleNamespace(cuda='13.0')
        torch.cuda=types.SimpleNamespace(is_available=lambda:True,get_device_capability=lambda:(9,0))
        self.cpp=types.ModuleType('torch.utils.cpp_extension')
        self.modules={'torch':torch,'torch.utils':types.ModuleType('torch.utils'),'torch.utils.cpp_extension':self.cpp}

    def test_builder_flag_mutation_does_not_modify_source_identity(self):
        original=loader.source_manifest();observed=[]
        def build(**kwargs):
            for key in ('extra_cuda_cflags','extra_cflags','extra_ldflags'):
                kwargs[key].append('builder-added-'+key)
            observed.append(kwargs)
            return types.SimpleNamespace()
        self.cpp.load=build
        with patch.dict(sys.modules,self.modules):module=loader.extension()
        self.assertEqual(len(observed),1)
        self.assertEqual(module.weave_source_manifest,original)
        self.assertEqual(module.weave_source_sha256,loader.source_sha256())
        self.assertEqual(loader.source_manifest(),original)
        self.assertEqual(original['ldflags'],['-lcuda'])

    def test_real_manifest_change_still_rejected_and_not_cached(self):
        original=loader.source_manifest();changed=copy.deepcopy(original)
        changed['files'][0]['sha256']='0'*64
        self.cpp.load=lambda **kwargs:types.SimpleNamespace()
        with patch.dict(sys.modules,self.modules),patch.object(loader,'source_manifest',side_effect=[original,changed]):
            with self.assertRaisesRegex(RuntimeError,'source changed during build'):loader.extension()
        self.assertEqual(loader.extension.cache_info().currsize,0)


if __name__=='__main__':unittest.main()
