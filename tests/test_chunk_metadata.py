import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.general import PROCESSING_VERSION, _signature, _transcribe_chunks, _alignment_implementation_fingerprint


class ChunkMetadataTests(unittest.TestCase):
    def test_gpu_load_failure_caches_actual_cpu_model_identity(self):
        from types import SimpleNamespace
        from unittest.mock import Mock
        segment=SimpleNamespace(no_speech_prob=0,avg_logprob=0,compression_ratio=1,start=.1,
            words=[SimpleNamespace(word='Hello.',start=.1,end=.7,probability=.99)])
        model=Mock()
        model.transcribe_original.return_value=([segment],SimpleNamespace(language='en',language_probability=1))
        raw=[{'text':'Hello.','start_ms':100,'end_ms':700,'probability':.99,'whisper_segment':.1}]
        plan={'path':'large-fixture','model':'large-v3','snapshot':'large-snapshot','device':'cuda','compute_type':'int8_float16'}
        with tempfile.TemporaryDirectory() as temporary, \
             patch('app.speech_runtime.speech_runtime_plan',return_value=plan), \
             patch('stable_whisper.load_faster_whisper',side_effect=[RuntimeError('missing CUDA'),model]), \
             patch('app.general._boundaries',return_value=[0,2]), \
             patch('app.general.subprocess.run'), \
             patch('app.alignment.align_words',return_value=(raw,{})):
            folder=Path(temporary)
            words,records=_transcribe_chunks(folder/'audio.mp3',2,folder,'fixture',lambda *a:None)
            saved=json.loads((folder/'.speech/0000.json').read_text(encoding='utf-8'))
            self.assertEqual(saved['runtime']['device'],'cpu')
            self.assertEqual(saved['runtime']['model'],'medium')
            self.assertEqual(saved['key'],_signature([PROCESSING_VERSION,'fixture',0,2,saved['runtime']]))
            self.assertEqual(saved['backend'],'CPU')
            self.assertEqual(words[0]['start_ms'],100)

    def test_cached_local_diagnostics_become_global_once_without_retranscription(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            speech = folder / '.speech'
            speech.mkdir()
            (speech / 'cuts.json').write_text(json.dumps({'duration': 600, 'cuts': [0, 300, 600]}))
            from app.speech_runtime import speech_runtime_plan
            runtime={**{k:v for k,v in speech_runtime_plan().items() if k!='path'},'alignment_implementation_sha256':_alignment_implementation_fingerprint()}
            for index, start in enumerate([0, 300]):
                record = {
                    'key': _signature([PROCESSING_VERSION, 'same-media', start, start + 300, runtime]),
                    'runtime': runtime,
                    'words': [{'text': ['Hello.', 'Again.'][index], 'start_ms': start * 1000 + 100,
                               'end_ms': start * 1000 + 600, 'whisper_segment': 1.5, 'probability': .95}],
                    'backend': 'CPU', 'language': 'en', 'language_probability': 1,
                    'suspects': [], 'start_ms': start * 1000, 'end_ms': (start + 300) * 1000,
                    'alignment': {'alignment_disputes': [{'text': 'Again.', 'aligned_ms': [1500, 1900],
                                                         'asr_ms': [1600, 2000]}],
                                  'connected_speech_groups': [{'measured_ms': [1500, 1900]}],
                                  'alignment_window_audits': [{'start_ms': 1400, 'end_ms': 2200,
                                      'observed_words': [{'text': 'Again.', 'start_ms': 1500, 'end_ms': 1900}]}]},
                }
                (speech / f'{index:04}.json').write_text(json.dumps(record))
            with patch('faster_whisper.WhisperModel', side_effect=AssertionError('cached audio must stay cached')), patch('app.speech_repair.repair_speech_words', side_effect=lambda audio, words, records, folder, progress: (words, records)):
                words, records = _transcribe_chunks(folder / 'audio.mp3', 600, folder, 'same-media', lambda *a: None)
                again_words, again_records = _transcribe_chunks(folder / 'audio.mp3', 600, folder, 'same-media', lambda *a: None)
            self.assertEqual(words[1]['start_ms'], 300100)
            self.assertEqual(words[1]['whisper_segment'], 300)
            self.assertEqual(records[1]['alignment']['alignment_disputes'][0]['aligned_ms'], [300000, 300400])
            self.assertEqual(records[1]['alignment']['connected_speech_groups'][0]['measured_ms'], [300000, 300400])
            audit = records[1]['alignment']['alignment_window_audits'][0]
            self.assertEqual([audit['start_ms'], audit['end_ms']], [299900, 300700])
            self.assertEqual(audit['observed_words'][0]['start_ms'], 300000)
            self.assertEqual(records[1]['time_basis'], 'media')
            self.assertEqual(records[1]['clip_start_ms'], 298500)
            self.assertEqual((words, records), (again_words, again_records))


if __name__ == '__main__':
    unittest.main()
