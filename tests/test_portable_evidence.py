import unittest
from app.privacy import public_package

class PortableEvidenceTests(unittest.TestCase):
    def test_runtime_evidence_stays_out_of_portable_package(self):
        original={'processing':{'private':'runtime'},'words':[{'text':'going to','phrase_unit':True,'spoken_words':[{'text':'going'},{'text':'to'}],'timing_evidence':{'clip_file':'private-local-file'},'original_zero_diagnostics':[{'runtime':'private'}]}]}
        result=public_package(original)
        self.assertNotIn('processing',result)
        self.assertNotIn('timing_evidence',result['words'][0])
        self.assertNotIn('original_zero_diagnostics',result['words'][0])
        self.assertTrue(result['words'][0]['phrase_unit'])
        self.assertIn('timing_evidence',original['words'][0])
