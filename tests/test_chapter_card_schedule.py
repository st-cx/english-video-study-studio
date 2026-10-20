import copy
import unittest
from app.general import _merge_chapter


class ChapterCardScheduleTests(unittest.TestCase):
    def test_cards_respect_chapter_reading_time_and_keep_every_note_and_source_span(self):
        sentence={'id':'s1','card_ids':['a','b','c','late','notes']}
        package={'episode':{},'sentences':[copy.deepcopy(sentence)],'words':[], 'cards':[], 'phrases':[]}
        chapter={'id':'ch1','start_ms':10000,'end_ms':30000}
        cards=[{'id':cid,'type':'vocab','title':cid,'meaning':'保留内容','sentence_ids':['s1'],
                'word_ids':['w'+cid],'start_ms':start,'end_ms':60000,'display':display}
               for cid,start,display in [('a',10000,True),('b',13500,True),('c',18500,True),('late',25220,True),('notes',15000,False)]]
        part={'episode':{},'sentences':[copy.deepcopy(sentence)],'words':[],'cards':copy.deepcopy(cards)}
        _merge_chapter(package,part,chapter)
        visible=sorted([c for c in package['cards'] if c.get('display')],key=lambda c:c['start_ms'])
        self.assertTrue(all(c['end_ms']-c['start_ms']>=8000 for c in visible))
        self.assertTrue(all(b['start_ms']-a['start_ms']>=8000 and a['end_ms']<=b['start_ms'] for a,b in zip(visible,visible[1:])))
        self.assertTrue(all(chapter['start_ms']<=c['start_ms']<c['end_ms']<=chapter['end_ms'] for c in visible))
        byid={c['id']:c for c in package['cards']}
        self.assertFalse(byid['ch1-b']['display'])
        self.assertFalse(byid['ch1-late']['display'])
        self.assertEqual([c['start_ms'] for c in visible],[10000,18500])
        self.assertFalse(byid['ch1-notes']['display'])
        self.assertEqual(len(package['cards']),len(cards))
        for original in cards:
            result=byid['ch1-'+original['id']]
            self.assertEqual((result['original_display_start_ms'],result['original_display_end_ms']), (original['start_ms'],original['end_ms']))
            for field in ('title','meaning','word_ids','sentence_ids'):self.assertEqual(result[field],original[field])
            if result['display']:self.assertEqual(result['start_ms'],max(chapter['start_ms'],original['start_ms']))

    def test_short_chapter_keeps_cards_for_notes_without_flashing_them(self):
        package={'episode':{},'sentences':[{'id':'s1'}],'words':[],'cards':[]}
        part={'episode':{},'sentences':[{'id':'s1'}],'words':[],
              'cards':[{'id':'c','sentence_ids':['s1'],'start_ms':0,'end_ms':90000,'display':True}]}
        _merge_chapter(package,part,{'id':'ch1','start_ms':0,'end_ms':4780})
        self.assertFalse(package['cards'][0]['display'])
        self.assertEqual(package['cards'][0]['original_display_end_ms'],90000)


if __name__=='__main__':unittest.main()
