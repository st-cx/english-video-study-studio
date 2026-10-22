import copy,unittest
from app.timing_revision import propose_timing_revision,retime_package
from app.sync_audit import digest

def fixture():
 words=[{'id':str(i),'text':text,'sentence_id':'s','start_ms':1000+i*400,'end_ms':1200+i*400} for i,text in enumerate(['we','read','good','books','today'])]
 windows=[]
 for offset in (0,10):
  windows.append({'start_ms':offset,'end_ms':10000+offset,'greedy_unprompted_text':'WE READ GOOD BOOKS TODAY','aligned':[{'index':i,'id':w['id'],'text':w['text'],'start_ms':w['start_ms']+100,'end_ms':w['end_ms']+100,'path_score_not_accuracy':.99} for i,w in enumerate(words)]})
 return words,{'audio_sha256':'a'*64,'input_timing_sha256':digest(words),'model_sha256':'b'*64,'windows':windows}

class TimingRevisionTests(unittest.TestCase):
 def test_measured_revision_keeps_text_and_ids(self):
  words,s=fixture();patch=propose_timing_revision(words,s,'a'*64)
  self.assertEqual(len(patch['patches']),5);self.assertFalse(patch['human_ground_truth'])
  self.assertEqual(words[1]['start_ms'],1400)
 def test_stale_audio_or_words_rejected(self):
  words,s=fixture()
  with self.assertRaises(ValueError):propose_timing_revision(words,s,'c'*64)
  words[0]['text']='changed'
  with self.assertRaises(ValueError):propose_timing_revision(words,s,'a'*64)
 def test_same_window_cannot_be_counted_twice(self):
  words,s=fixture();s['windows']=[s['windows'][0],s['windows'][0]]
  self.assertFalse(propose_timing_revision(words,s,'a'*64)['patches'])
 def test_low_score_unstable_or_missing_greedy_support_retained(self):
  for mode in ['low','unstable','unsupported','nonfinite']:
   words,s=fixture()
   if mode=='low':s['windows'][1]['aligned'][2]['path_score_not_accuracy']=.2
   if mode=='unstable':s['windows'][1]['aligned'][2]['start_ms']+=100
   if mode=='unsupported':
    for win in s['windows']:win['greedy_unprompted_text']='WE SELL OLD HOUSES TODAY'
   if mode=='nonfinite':s['windows'][0]['aligned'][2]['end_ms']=float('nan')
   changes=propose_timing_revision(words,s,'a'*64)['patches']
   self.assertNotIn('2',[p['word_id'] for p in changes],mode)
 def test_candidate_cannot_overlap_unmeasured_neighbor(self):
  words,s=fixture()
  for win in s['windows']:
   win['aligned'][2]['start_ms']=2200;win['aligned'][2]['end_ms']=2350
   win['aligned'][3]['path_score_not_accuracy']=.1
  self.assertNotIn('2',[p['word_id'] for p in propose_timing_revision(words,s,'a'*64)['patches']])
 def test_retime_updates_every_view_and_invalidates_video_and_old_audit(self):
  words,s=fixture();pkg={'words':words,'sentences':[{'id':'s','text':'we read good books today','word_ids':[w['id'] for w in words],'start_ms':1000,'end_ms':2800}],
   'cards':[{'id':'c','word_ids':['2'],'sentence_ids':['s'],'start_ms':1800,'end_ms':3000}],
   'layout':{'pages':[{'sentence_id':'s','lines':[['0','1'],['2','3','4']],'start_ms':1000,'end_ms':2800}]},'artifacts':{'video':'ready'},'render':{'status':'ready'},'sync_review':{'page':'old'},'processing':{},'episode':{'duration_s':10}}
  before=copy.deepcopy(pkg);after=retime_package(pkg,propose_timing_revision(words,s,'a'*64))
  self.assertEqual(pkg,before);self.assertEqual(after['sentences'][0]['start_ms'],1100)
  self.assertEqual(after['layout']['pages'][0]['start_ms'],1100);self.assertEqual(after['cards'][0]['start_ms'],1900)
  self.assertEqual(after['artifacts']['video'],'not_requested');self.assertNotIn('sync_review',after)
 def test_tampered_patch_rejected_before_mutation(self):
  words,s=fixture();proposal=propose_timing_revision(words,s,'a'*64);proposal['patches'][0]['before_ms']=[0,0]
  with self.assertRaises(ValueError):retime_package({'words':words},proposal)

if __name__=='__main__':unittest.main()
