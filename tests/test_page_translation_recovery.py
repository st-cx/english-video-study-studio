import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from app.layout import _page_translations
from app.providers import ProviderError


def package():
    return {'sentences':[{'id':'s0','text':'Learn slowly. Keep going.','translation':'慢慢学习，坚持下去。'}],
      'words':[{'id':'w0','text':'Learn'},{'id':'w1','text':'slowly.'},{'id':'w2','text':'Keep'},{'id':'w3','text':'going.'}],
      'layout':{'pages':[{'id':'page-0','sentence_id':'s0','lines':[['w0','w1']]},{'id':'page-1','sentence_id':'s0','lines':[['w2','w3']]}]}}


class PageTranslationRecoveryTests(unittest.TestCase):
    def test_missing_or_long_page_is_targeted_without_clipping(self):
        p=package();original=copy.deepcopy(p['words'])
        replies=[{'pages':[{'id':'page-0','zh':'过长'*40}]},{'pages':[{'id':'page-0','zh':'慢慢学习。'},{'id':'page-1','zh':'堅持下去。'}]}]
        with tempfile.TemporaryDirectory() as temporary,patch('app.providers.ask_deepseek',side_effect=replies):
            _page_translations(p,Path(temporary))
        self.assertEqual([x['translation'] for x in p['layout']['pages']],['慢慢学习。','坚持下去。'])
        self.assertEqual(p['words'],original)

    def test_exhausted_repairs_do_not_publish_placeholder_or_truncate(self):
        p=package()
        with tempfile.TemporaryDirectory() as temporary,patch('app.providers.ask_deepseek',return_value={'pages':[]}):
            with self.assertRaises(ProviderError):_page_translations(p,Path(temporary))
        self.assertFalse(any('translation' in p for p in p['layout']['pages']))
