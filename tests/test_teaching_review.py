import copy
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.providers import ProviderError
from app.teaching_review import review_teaching


def package():
    return {
        'sentences': [{'id': 's1', 'text': 'Are you going to try?',
                       'translation': '你打算试试吗？', 'speaker': ''}],
        'words': [{'id': 'w1', 'text': 'to', 'sentence_id': 's1',
                   'gloss_zh': '（不定式标记）', 'start_ms': 100, 'end_ms': 200}],
        'phrases': [], 'cards': [], 'processing': {},
    }


class TeachingReviewTests(unittest.TestCase):
    def test_identical_correction_requires_explicit_verdict_without_fake_history(self):
        data = package()
        noop = {'type': 'word', 'id': 'w1', 'field': 'gloss_zh',
                'value': '（不定式标记）', 'reason': '原义正确，无需修改'}
        answers = [
            {'ok': False, 'corrections': [noop], 'issues': []},
            {'ok': True, 'corrections': [], 'issues': []},
        ]
        with patch('app.teaching_review.ask_deepseek', side_effect=answers) as ask:
            review_teaching(data, lambda *a: None)
        self.assertEqual(data['teaching_review']['status'], 'passed')
        self.assertEqual(data['teaching_review']['corrections'], [])
        self.assertEqual(ask.call_count, 2)
        adjudication = json.loads(ask.call_args_list[1].args[1])
        self.assertIn('unchanged_corrections', adjudication)
        self.assertEqual(data['words'][0]['start_ms'], 100)

    def test_every_real_correction_is_rechecked_even_with_ok_true(self):
        data = package()
        def change(value):
            return {'ok': True, 'corrections': [{'type': 'word', 'id': 'w1',
                    'field': 'gloss_zh', 'value': value, 'reason': '语法作用'}], 'issues': []}
        answers = [change('不定式标记'), change('不定式符号'),
                   {'ok': True, 'corrections': [], 'issues': []}]
        with patch('app.teaching_review.ask_deepseek', side_effect=answers) as ask:
            review_teaching(data, lambda *a: None)
        self.assertEqual(ask.call_count, 3)
        self.assertEqual(len(data['teaching_review']['corrections']), 2)

    def test_unchanged_points_in_simplified_chinese_are_not_corrections(self):
        data = package()
        data['cards'] = [{'id': 'c1', 'sentence_ids': ['s1'],
                          'points': [{'label': '结构', 'text': '不定式作宾语'}]}]
        answer = {'ok': True, 'issues': [], 'corrections': [{
            'type': 'card', 'id': 'c1', 'field': 'points',
            'value': [{'text': '不定式作賓語', 'label': '結構'}], 'reason': '原义正确'}]}
        with patch('app.teaching_review.ask_deepseek', return_value=answer) as ask:
            review_teaching(data, lambda *a: None)
        self.assertEqual(ask.call_count, 1)
        self.assertEqual(data['teaching_review']['corrections'], [])

    def test_noop_without_explicit_pass_stays_pending(self):
        data = package()
        answer = {'ok': False, 'issues': [], 'corrections': [{
            'type': 'word', 'id': 'w1', 'field': 'gloss_zh',
            'value': '（不定式标记）', 'reason': '原义正确'}]}
        with patch('app.teaching_review.ask_deepseek', return_value=answer) as ask:
            with self.assertRaises(ProviderError):
                review_teaching(data, lambda *a: None)
        self.assertEqual(ask.call_count, 2)
        self.assertEqual(data['teaching_review']['status'], 'needs_review')
        self.assertEqual(data['teaching_review']['corrections'], [])

    def test_genuine_issue_remains_pending_after_adjudication(self):
        data = package()
        answer = {'ok': False, 'issues': [{'sentence_id': 's1',
                  'reason': '数字与原句不一致，需确认来源'}], 'corrections': []}
        with patch('app.teaching_review.ask_deepseek', return_value=answer):
            with self.assertRaises(ProviderError):
                review_teaching(data, lambda *a: None)
        self.assertEqual(data['teaching_review']['issues'], answer['issues'])

    def test_cross_batch_noop_cannot_bypass_scope_validation(self):
        data = package()
        for i in range(2, 9):
            data['sentences'].append({'id': f's{i}', 'text': 'Hello.',
                                      'translation': '你好。', 'speaker': ''})
        answer = {'ok': True, 'issues': [], 'corrections': [{
            'type': 'sentence', 'id': 's8', 'field': 'translation',
            'value': '你好。', 'reason': '无需修改'}]}
        before = copy.deepcopy(data)
        with patch('app.teaching_review.ask_deepseek', return_value=answer):
            with self.assertRaises(ProviderError):
                review_teaching(data, lambda *a: None)
        self.assertEqual(data['sentences'], before['sentences'])

    def test_invalid_value_does_not_partially_apply_review(self):
        data = package()
        before = copy.deepcopy(data)
        answer = {'ok': True, 'issues': [], 'corrections': [
            {'type': 'word', 'id': 'w1', 'field': 'gloss_zh', 'value': '新解释'},
            {'type': 'sentence', 'id': 's1', 'field': 'translation', 'value': 123},
        ]}
        with patch('app.teaching_review.ask_deepseek', return_value=answer):
            with self.assertRaises(ProviderError):
                review_teaching(data, lambda *a: None)
        self.assertEqual(data['words'], before['words'])

    def test_continuing_real_corrections_stays_pending_with_target_ids(self):
        data = package()
        answers = [{'ok': True, 'issues': [], 'corrections': [{
            'type': 'word', 'id': 'w1', 'field': 'gloss_zh',
            'value': value, 'reason': '语法作用仍有问题'}]}
            for value in ('不定式标记', '不定式符号', '不定式标记')]
        with patch('app.teaching_review.ask_deepseek', side_effect=answers):
            with self.assertRaises(ProviderError):
                review_teaching(data, lambda *a: None)
        review = data['teaching_review']
        self.assertEqual(review['status'], 'needs_review')
        self.assertEqual(review['issues'][0]['targets'][0]['id'], 'w1')
        self.assertEqual(len(review['corrections']), 3)

    def test_schema_repair_is_rechecked_and_invalid_changes_never_apply(self):
        data=package()
        bad={'ok':True,'issues':[],'corrections':[{'type':'sentence','id':'s1','field':'translation','value':123}]}
        repaired={'ok':True,'issues':[],'corrections':[{'type':'sentence','id':'s1','field':'translation','value':'你打算试试吗？'}]}
        with patch('app.teaching_review.ask_deepseek',side_effect=[bad,repaired]) as ask:
            review_teaching(data,lambda *a:None)
        self.assertEqual(ask.call_count,2)
        self.assertEqual(data['teaching_review']['status'],'passed')
        self.assertEqual(data['sentences'][0]['translation'],'你打算试试吗？')
        self.assertEqual(data['teaching_review']['corrections'],[])

    def test_retry_keeps_real_correction_history_and_removes_old_noops(self):
        data = package()
        change = {'type': 'word', 'id': 'w1', 'field': 'gloss_zh',
                  'value': '（不定式标记）', 'before': '去', 'reason': '语法错误'}
        noop = {**change, 'before': '（不定式标记）'}
        data['teaching_review'] = {'status': 'needs_review',
                                   'corrections': [change, noop]}
        with patch('app.teaching_review.ask_deepseek', return_value={
                'ok': True, 'corrections': [], 'issues': []}):
            review_teaching(data, lambda *a: None)
        self.assertEqual(data['teaching_review']['corrections'], [change])

    def test_issue_adjudication_does_not_discard_pending_real_corrections(self):
        data = package()
        data['sentences'][0]['translation'] = '你不打算试试。'
        answers = [
            {'ok': False, 'corrections': [{'type': 'sentence', 'id': 's1',
                'field': 'translation', 'value': '你打算试试吗？', 'reason': '原句无否定'}],
             'issues': [{'sentence_id': 's1', 'reason': '需核对口语省略'}]},
            {'ok': True, 'corrections': [], 'issues': []},
            {'ok': True, 'corrections': [], 'issues': []},
        ]
        with patch('app.teaching_review.ask_deepseek', side_effect=answers) as ask:
            review_teaching(data, lambda *a: None)
        self.assertEqual(data['sentences'][0]['translation'], '你打算试试吗？')
        self.assertEqual(ask.call_count, 3)

    def test_issue_adjudication_can_explicitly_retract_a_correction(self):
        data = package()
        original = data['sentences'][0]['translation']
        change = {'type': 'sentence', 'id': 's1', 'field': 'translation',
                  'value': '错误改法', 'reason': '待裁决'}
        answers = [
            {'ok': False, 'corrections': [change],
             'issues': [{'sentence_id': 's1', 'reason': '需核对'}]},
            {'ok': True, 'corrections': [{**change, 'value': original,
                                       'reason': '原义正确，撤回修改'}], 'issues': []},
        ]
        with patch('app.teaching_review.ask_deepseek', side_effect=answers):
            review_teaching(data, lambda *a: None)
        self.assertEqual(data['sentences'][0]['translation'], original)
        self.assertEqual(data['teaching_review']['corrections'], [])


if __name__ == '__main__':
    unittest.main()
