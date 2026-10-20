"""Release boundaries: portable metadata, optional cards and actual CPU model."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock,patch
from app.general import _transcribe_chunks, _alignment_implementation_fingerprint
from app.pipeline import _annotate,make_notes
from app.privacy import public_package
from app.providers import ProviderError
from tests.test_annotation_recovery import package,answer


class PortableSourceTests(unittest.TestCase):
 def test_signed_source_urls_are_cleaned_without_mutating_private_input(self):
  url='https://example.com/audio.mp3?lang=en&access_token=SYNTHETIC&X-Amz-Signature=TEST&key=TEST&auth=TEST#token=TEST'
  original={'episode':{'source_url':url},'source':{'url':url},'api_key':'SYNTHETIC','processing':{'private_url':url}}
  result=public_package(original)
  self.assertEqual(result['episode']['source_url'],'https://example.com/audio.mp3?lang=en')
  self.assertEqual(result['source']['url'],'https://example.com/audio.mp3?lang=en')
  self.assertEqual(original['episode']['source_url'],url)
  self.assertNotIn('processing',result);self.assertNotIn('api_key',result)

 def test_public_video_identity_is_kept_while_userinfo_and_signatures_are_removed(self):
  source='https://name:pass@www.youtube.com/watch?v=PUBLIC_ID&list=PUBLIC_LIST&API_KEY=TEST&oauth_token=TEST'
  result=public_package({'episode':{'source_url':source}})
  self.assertEqual(result['episode']['source_url'],'https://www.youtube.com/watch?v=PUBLIC_ID&list=PUBLIC_LIST')

 def test_common_auth_and_sign_aliases_are_not_portable(self):
  source='https://example.com/media?sign=TEST&x-api-key=TEST&authSig=TEST&OSSAccessKeyId=TEST&hmac=TEST&jwt=TEST&t=45&keyword=english&author=someone&monkey=enabled&keyframe=20'
  self.assertEqual(public_package({'source':{'url':source}})['source']['url'],'https://example.com/media?t=45&keyword=english&author=someone&monkey=enabled&keyframe=20')

 def test_notes_do_not_export_signed_source_credentials(self):
  value=package();value['episode'].update(source_url='https://example.com/audio.mp3?signature=TEST&lang=en',duration_s=2)
  notes=make_notes(value)
  self.assertIn('https://example.com/audio.mp3?lang=en',notes);self.assertNotIn('signature=TEST',notes)


class OptionalCardBoundaryTests(unittest.TestCase):
 def test_empty_title_is_repaired_before_otherwise_complete_content_is_blocked(self):
  value=package();bad=answer();bad['cards']=[{'title':'','meaning':'学习方式','sentence_ids':['s0'],'example':'Learn slowly.'}]
  good=copy.deepcopy(bad);good['cards'][0]['title']='slowly'
  with patch('app.pipeline.ask_deepseek',side_effect=[bad,good]) as provider,patch('app.presentation.translate_excerpts'):
   _annotate(value,'',lambda *a:None)
  self.assertEqual(provider.call_count,2);self.assertEqual(value['cards'][0]['title'],'slowly')

 def test_repeated_empty_optional_title_is_omitted_without_making_up_title(self):
  value=package();bad=answer();bad['cards']=[{'title':' ','meaning':'学习方式','sentence_ids':['s0'],'example':'Learn slowly.'}]
  with patch('app.pipeline.ask_deepseek',side_effect=[copy.deepcopy(bad),copy.deepcopy(bad)]) as provider,patch('app.presentation.translate_excerpts'):
   _annotate(value,'',lambda *a:None)
  self.assertEqual(provider.call_count,2);self.assertEqual(value['cards'],[])
  self.assertTrue(value['processing']['omitted_card_suggestions'])
  self.assertEqual(value['sentences'][0]['translation'],'慢慢学习。')

 def test_missing_required_translation_still_stops_after_bounded_repair(self):
  value=package();bad=answer();bad['translations']=[]
  with patch('app.pipeline.ask_deepseek',side_effect=[bad,{},{}]),patch('app.presentation.translate_excerpts'):
   with self.assertRaises(ProviderError):_annotate(value,'',lambda *a:None)
  self.assertEqual(value['sentences'][0]['translation'],'')


class CpuFallbackIdentityTests(unittest.TestCase):
 def run_fallback(self,cpu_path):
  segment=SimpleNamespace(no_speech_prob=0,avg_logprob=0,compression_ratio=1,start=.1,
   words=[SimpleNamespace(word='Hello.',start=.1,end=.7,probability=.99)])
  model=Mock();model.transcribe_original.return_value=([segment],SimpleNamespace(language='en',language_probability=1))
  raw=[{'text':'Hello.','start_ms':100,'end_ms':700,'probability':.99,'whisper_segment':.1}]
  plan={'path':'large-fixture','model':'large-v3','snapshot':'large-snapshot','device':'cuda','compute_type':'int8_float16'}
  with tempfile.TemporaryDirectory() as temporary,patch('app.speech_runtime.speech_runtime_plan',return_value=plan),patch('app.pipeline._local_whisper_path',return_value=cpu_path),patch('stable_whisper.load_faster_whisper',side_effect=[RuntimeError('missing CUDA'),model]) as loader,patch('app.general._boundaries',return_value=[0,2]),patch('app.general.subprocess.run'),patch('app.alignment.align_words',return_value=(raw,{})):
   folder=Path(temporary);_,records=_transcribe_chunks(folder/'audio.mp3',2,folder,'fixture',lambda *a:None)
   self.assertEqual(loader.call_args_list[1].args[0],cpu_path)
   checkpoint=json.loads((folder/'.speech/0000.json').read_text('utf-8'))
   return records[0],checkpoint

 def test_downloaded_english_only_cpu_model_is_recorded_exactly(self):
  record,checkpoint=self.run_fallback('medium.en')
  self.assertEqual(record['runtime'],{'model':'medium.en','snapshot':'','device':'cpu','compute_type':'int8','alignment_implementation_sha256':_alignment_implementation_fingerprint()})
  self.assertEqual(record['backend'],'CPU');self.assertEqual(checkpoint['runtime'],record['runtime'])

 def test_cached_multilingual_cpu_model_keeps_snapshot_identity(self):
  model_path=str(Path('model-cache')/'models--Systran--faster-whisper-medium'/'snapshots'/'test-snapshot')
  record,_=self.run_fallback(model_path)
  self.assertEqual(record['runtime'],{'model':'medium','snapshot':'test-snapshot','device':'cpu','compute_type':'int8','alignment_implementation_sha256':_alignment_implementation_fingerprint()})


if __name__=='__main__':unittest.main()
