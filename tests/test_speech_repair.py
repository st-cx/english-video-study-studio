from __future__ import annotations
import copy
from pathlib import Path
import sys,tempfile,unittest
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from app.speech_repair import repair_speech_words


def words():
 return [{'text':t,'start_ms':i*300,'end_ms':i*300+200} for i,t in enumerate(['we','know','the','word','today','now'])]


def records(source):
 # Local word_index points to the WRONG global word on purpose.
 return [{'time_basis':'media','start_ms':0,'alignment':{'alignment_disputes':[
  {'word_index':0,'text':'word','aligned_ms':[900,1100],'reason':'test_dispute'},
  {'word_index':3,'text':'the','aligned_ms':[600,800],'reason':'other_dispute'}], 'alignment_repairs':[], 'alignment_window_audits':[{'raw':'preserve'}]}}]


def engine(source,*,status='measured_transaction',target=3,changed=True):
 output=copy.deepcopy(source)
 p={'word_index':target,'text':source[target]['text'],'original_start_ms':source[target]['start_ms'],'original_end_ms':source[target]['end_ms'],
    'start_ms':source[target]['start_ms']+10,'end_ms':source[target]['end_ms']+10}
 if changed:output[target].update(start_ms=p['start_ms'],end_ms=p['end_ms'])
 measurements=[{'status':'measured','padding_ms':padding,'aligned_context':[{'index':target,'text':source[target]['text'],'measured_start_ms':p['start_ms'],'measured_end_ms':p['end_ms'],'raw_path_score_not_accuracy':.95}]} for padding in [2500,4500]]
 return output,{'status':'measured','rows':[{'word_index':target,'measurements':measurements,'transaction':{'status':status,'patch':[p]}}],'applied_transactions':1}


class SpeechRepairTests(unittest.TestCase):
 def setUp(self):
  self.temp=tempfile.TemporaryDirectory(dir=ROOT/'work');self.folder=Path(self.temp.name)
  self.audio=self.folder/'original.wav';self.audio.write_bytes(b'synthetic-audio-fingerprint-fixture')
  self.modelpatch=patch('app.speech_repair._model_cache',return_value=self.folder/'no-model')
  self.modelpatch.start()
 def tearDown(self):
  self.modelpatch.stop();self.temp.cleanup()
 def run_repair(self,source=None,recs=None):
  source=source or words();recs=recs if recs is not None else records(source)
  return repair_speech_words(self.audio,source,recs,self.folder,lambda _:None)
 def test_only_exact_global_target_resolves_and_raw_history_survives(self):
  source=words();recs=records(source);before=copy.deepcopy(recs)
  with patch('app.speech_repair.repair_with_ctc',return_value=engine(source)):
   output,result=self.run_repair(source,recs)
  alignment=result[0]['alignment']
  self.assertEqual(len(alignment['alignment_disputes']),1)
  self.assertEqual(alignment['alignment_disputes'][0]['text'],'the')
  self.assertEqual(alignment['resolved_history'][0]['original_dispute'],before[0]['alignment']['alignment_disputes'][0])
  self.assertEqual(alignment['resolved_history'][0]['resolution']['global_word_index'],3)
  self.assertFalse(alignment['resolved_history'][0]['resolution']['human_ground_truth'])
  self.assertEqual(recs,before)
  self.assertEqual(output[3]['start_ms'],910)
 def test_rejected_or_not_actually_applied_target_cannot_clear_dispute(self):
  source=words();recs=records(source)
  for result in [engine(source,status='rejected'),engine(source,changed=False)]:
   with patch('app.speech_repair.repair_with_ctc',return_value=result):
    _,new=self.run_repair(source,recs)
   self.assertEqual(new[0]['alignment']['alignment_disputes'],recs[0]['alignment']['alignment_disputes'])
   # Avoid legitimately hitting the earlier cached input in the next scenario.
   self.audio.write_bytes(self.audio.read_bytes()+b'x')
 def test_measured_changed_collateral_neighbor_can_resolve_exact_dispute(self):
  source=words();out,evidence=engine(source,target=4)
  p={'word_index':3,'text':'word','original_start_ms':900,'original_end_ms':1100,'start_ms':910,'end_ms':1110}
  evidence['rows'][0]['transaction']['patch'].append(p);out[3].update(start_ms=910,end_ms=1110)
  for m in evidence['rows'][0]['measurements']:
   m['aligned_context'].append({'index':3,'text':'word','measured_start_ms':910,'measured_end_ms':1110,'raw_path_score_not_accuracy':.95})
  with patch('app.speech_repair.repair_with_ctc',return_value=(out,evidence)):
   _,new=self.run_repair(source)
  self.assertEqual(len(new[0]['alignment']['alignment_disputes']),1)
  self.assertEqual(new[0]['alignment']['alignment_disputes'][0]['text'],'the')
  self.assertEqual(new[0]['alignment']['resolved_history'][0]['resolution']['global_word_index'],3)
 def test_unmeasured_unchanged_or_low_score_neighbor_never_resolves(self):
  source=words()
  for mode in ['unmeasured','unchanged','low_score','unstable']:
   out,evidence=engine(source,target=4)
   p={'word_index':3,'text':'word','original_start_ms':900,'original_end_ms':1100,'start_ms':910,'end_ms':1110}
   if mode=='unchanged':p.update(start_ms=900,end_ms=1100)
   evidence['rows'][0]['transaction']['patch'].append(p)
   out[3].update(start_ms=p['start_ms'],end_ms=p['end_ms'])
   if mode!='unmeasured':
    for m in evidence['rows'][0]['measurements']:
     m['aligned_context'].append({'index':3,'text':'word','measured_start_ms':p['start_ms'],'measured_end_ms':p['end_ms'],'raw_path_score_not_accuracy':.5 if mode=='low_score' else .95})
   if mode=='unstable':evidence['rows'][0]['measurements'][0]['aligned_context'][-1]['measured_start_ms']-=100
   with patch('app.speech_repair.repair_with_ctc',return_value=(out,evidence)):
    _,new=self.run_repair(source)
   self.assertEqual(len(new[0]['alignment']['alignment_disputes']),2,mode)
   self.audio.write_bytes(self.audio.read_bytes()+b'x')
 def test_legacy_local_or_ambiguous_time_mapping_preserves_dispute(self):
  source=words();recs=records(source);recs[0]['time_basis']='clip'
  with patch('app.speech_repair.repair_with_ctc',return_value=engine(source)):
   _,new=self.run_repair(source,recs)
  self.assertEqual(len(new[0]['alignment']['alignment_disputes']),2)
  source.append(copy.deepcopy(source[3]));recs=records(source)
  with patch('app.speech_repair.repair_with_ctc',return_value=engine(source)):
   _,new=self.run_repair(source,recs)
  self.assertEqual(len(new[0]['alignment']['alignment_disputes']),2)
 def test_cache_requires_matching_audio_words_records_and_complete_evidence(self):
  source=words();recs=records(source)
  with patch('app.speech_repair.repair_with_ctc',return_value=engine(source)) as model:
   first=self.run_repair(source,recs);second=self.run_repair(source,recs)
   self.assertEqual(first,second);self.assertEqual(model.call_count,1)
   self.audio.write_bytes(b'changed-audio');self.run_repair(source,recs);self.assertEqual(model.call_count,2)
   source[0]['text']='changed';self.run_repair(source,recs);self.assertEqual(model.call_count,3)
   recs[0]['new_diagnostic']=True;self.run_repair(source,recs);self.assertEqual(model.call_count,4)
   # An interrupted/deleted audit cannot be concealed by a cached success.
   evidence=list((self.folder/'.speech-repair').glob('*/evidence.json'))[-1]
   evidence.write_text('corrupt',encoding='utf-8')
   # Find the currently used evidence exactly by its cache result's records.
   _,current=self.run_repair(source,recs)
   current_evidence=Path(current[0]['alignment']['ctc_repair_audit']['evidence_file'])
   current_evidence.write_text('corrupt',encoding='utf-8')
   before=model.call_count;self.run_repair(source,recs);self.assertEqual(model.call_count,before+1)
 def test_pause_callback_propagates_original_exception_without_failure_cache(self):
  class JobPaused(RuntimeError):pass
  paused=JobPaused('synthetic pause requested by user')
  def progress(_):raise paused
  def model(audio,source,targets,report,*,cache_dir):
   try:report('local CTC measurement')
   except RuntimeError as error:raise RuntimeError('lower-layer wrapper') from error
   raise AssertionError('pause was swallowed')
  with patch('app.speech_repair.repair_with_ctc',side_effect=model):
   with self.assertRaises(JobPaused) as caught:
    repair_speech_words(self.audio,words(),records(words()),self.folder,progress)
  self.assertIs(caught.exception,paused)
  self.assertFalse((self.folder/'.speech-repair').exists())

 def test_missing_model_or_measurement_failure_never_clears_originals(self):
  source=words();recs=records(source)
  _,new=self.run_repair(source,recs)
  self.assertEqual(new[0]['alignment']['alignment_disputes'],recs[0]['alignment']['alignment_disputes'])
  self.audio.write_bytes(b'another-audio')
  with patch('app.speech_repair.repair_with_ctc',side_effect=RuntimeError('synthetic failure')):
   out,new=self.run_repair(source,recs)
  self.assertEqual(out,source);self.assertEqual(new[0]['alignment']['alignment_disputes'],recs[0]['alignment']['alignment_disputes'])

if __name__=='__main__':unittest.main()
