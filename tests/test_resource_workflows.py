from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from app import publishing, workflows
from app.sources import checked_url


class ResourceWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.patch = patch.object(workflows, 'data_root', return_value=self.root)
        self.patch.start()
        self.enqueue = Mock(side_effect=lambda spec: 'job-' + str(len(self.enqueue.call_args_list)))

    def tearDown(self):
        self.patch.stop()
        self.temporary.cleanup()

    def source(self, **options):
        return workflows.save_source({'url': 'https://example.org/feed.xml', 'mode': 'rss',
                                      'name': 'Learning feed', 'auto_generate': True, **options})

    def entries(self, *ids):
        return [{'url': f'https://example.org/{identifier}.mp4', 'title': f'Video {identifier}'} for identifier in ids]

    def scan(self, source, entries):
        with patch.object(workflows, 'discover', return_value=entries):
            return workflows.scan_source(source['id'], self.enqueue)

    def automatic_source(self):
        return self.source(auto_publish=True, publication={'rights_confirmed': True, 'tid': 208, 'copyright': 2})

    def complete(self, identifier):
        return {'status': 'complete', 'episode_id': identifier}

    def test_first_scan_only_latest_then_new_items_are_deduplicated(self):
        source = self.source()
        first = self.scan(source, self.entries(3, 2, 1))
        self.assertEqual(len(first['items']), 1)
        self.assertEqual(first['items'][0]['title'], 'Video 3')
        self.assertEqual(len(first['seen']), 3)
        again = self.scan(source, self.entries(3, 2, 1))
        self.assertEqual(len(again['items']), 1)
        updated = self.scan(source, self.entries(5, 4, 3, 2, 1))
        self.assertEqual(len(updated['items']), 3)
        self.assertEqual(self.enqueue.call_count, 3)
        self.assertEqual(len({row['id'] for row in updated['items']}), 3)

    def test_later_burst_preserves_overflow_for_next_scan(self):
        source = self.source()
        self.scan(source, self.entries(1))
        burst = self.entries(*range(9, 0, -1))
        first_batch = self.scan(source, burst)
        self.assertEqual(len(first_batch['items']), 6)
        self.assertEqual(len(first_batch['seen']), 6)
        second_batch = self.scan(source, burst)
        self.assertEqual(len(second_batch['items']), 9)
        self.assertEqual(self.enqueue.call_count, 9)

    def test_manual_discovery_and_repeated_generate_share_job(self):
        source = self.source(auto_generate=False)
        found = self.scan(source, self.entries(1))
        self.assertFalse(self.enqueue.called)
        item = found['items'][0]
        one = workflows.enqueue_item(source['id'], item['id'], self.enqueue)
        two = workflows.enqueue_item(source['id'], item['id'], self.enqueue)
        self.assertEqual(one['job_id'], two['job_id'])
        self.assertEqual(self.enqueue.call_count, 1)

    def test_edit_preserves_history_but_changes_revision(self):
        source = self.source()
        scanned = self.scan(source, self.entries(1))
        updated = self.source(id=source['id'], name='Renamed', enabled=False)
        self.assertNotEqual(source['revision'], updated['revision'])
        self.assertEqual(updated['seen'], scanned['seen'])
        self.assertEqual(updated['items'], scanned['items'])
        with self.assertRaisesRegex(ValueError, '新建'):
            workflows.save_source({**updated, 'url': 'https://example.org/changed.xml'})
        with patch.object(workflows, 'discover') as discover:
            workflows.scan_source(source['id'], self.enqueue, force=False)
            discover.assert_not_called()

    def test_edit_during_discovery_cannot_enqueue_old_revision(self):
        source = self.source()
        def change(_):
            self.source(id=source['id'], enabled=False)
            return self.entries(1)
        with patch.object(workflows, 'discover', side_effect=change):
            with self.assertRaises(ValueError):
                workflows.scan_source(source['id'], self.enqueue)
        self.assertFalse(self.enqueue.called)
        self.assertFalse(workflows.list_sources()[0]['enabled'])

    def test_failure_is_persistent_safe_and_keeps_previous_items(self):
        source = self.source()
        previous = self.scan(source, self.entries(1))
        with patch.object(workflows, 'discover', side_effect=RuntimeError('private-token-must-not-leak')):
            with self.assertRaises(ValueError):
                workflows.scan_source(source['id'], self.enqueue)
        after = workflows.list_sources()[0]
        self.assertEqual(after['items'], previous['items'])
        self.assertTrue(after['error'])
        self.assertNotIn('private-token', json.dumps(after))
        self.assertGreater(after['last_checked'], 0)

    def test_rss_atom_relative_urls_and_entity_rejection(self):
        source = self.source()
        content = b'<feed xmlns="http://www.w3.org/2005/Atom"><entry><title>Episode 1</title><link rel="enclosure" href="../audio.mp3"/></entry></feed>'
        with patch.object(workflows, '_read_url', return_value=(content, 'https://example.org/feeds/rss', {})):
            rows = workflows.discover(source)
            self.assertEqual(rows, [{'title': 'Episode 1', 'url': 'https://example.org/audio.mp3'}])
        with patch.object(workflows, '_read_url', return_value=(b'<!DOCTYPE x [<!ENTITY secret SYSTEM "file:///private">]><rss/>', source['url'], {})):
            with self.assertRaisesRegex(ValueError, '外部实体'):
                workflows.discover(source)

    def test_immediate_url_and_id_validation_does_not_need_dns(self):
        for url in ('http://127.0.0.1/x', 'http://10.0.0.1/x', 'http://[::1]/x', 'http://[::ffff:127.0.0.1]/x', 'http://localhost./x', 'http://printer.local/x', 'file:///x', 'https://user:pass@example.org/x'):
            with self.subTest(url=url), self.assertRaises(ValueError):
                self.source(url=url)
        for identifier in ('bad-id', ['a'], 'A' * 32):
            with self.subTest(identifier=identifier), self.assertRaises(ValueError):
                self.source(id=identifier)
        self.assertEqual(workflows.list_sources(), [])
        with patch('app.sources.socket.getaddrinfo', side_effect=AssertionError('DNS must not run while saving')):
            self.source(url='https://unresolved-example.org/rss')

    def test_actual_public_url_validation_rejects_private_dns_answer(self):
        with patch('app.sources.socket.getaddrinfo', return_value=[(2, 1, 6, '', ('127.0.0.1', 443))]):
            with self.assertRaisesRegex(ValueError, '内网'):
                checked_url('https://example.org/feed')

    def test_invalid_auto_publish_is_not_saved_or_model_reconfigured(self):
        for options in ({'auto_publish': True, 'auto_generate': False, 'publication': {'rights_confirmed': True}},
                        {'auto_publish': True, 'publication': {}}, {'interval_minutes': True}, {'interval_minutes': 1}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.source(**options)
        self.assertFalse((self.root / 'settings.json').exists())
        self.assertFalse((self.root / 'deepseek.key.dpapi').exists())

    def test_blocked_publication_is_bound_but_never_submitted(self):
        source = self.automatic_source()
        self.scan(source, self.entries(1))
        with patch.object(publishing, 'prepare_publication', return_value={'id': 'p1', 'status': 'blocked'}) as prepare, patch.object(publishing, 'submit_publication') as submit:
            workflows.handoff_completed(self.complete, lambda episode_id: ({}, self.root))
            submit.assert_not_called()
            self.assertEqual(prepare.call_args.args[2]['source'], 'https://example.org/1.mp4')
            self.assertIn('简体中文', prepare.call_args.args[2]['description'])
        item = workflows.list_sources()[0]['items'][0]
        self.assertEqual(item['publication_id'], 'p1')
        self.assertEqual(item['publication_revision'], source['revision'])
        self.assertTrue(item['error'])

    def test_login_can_resume_bound_draft_without_new_draft(self):
        source = self.automatic_source()
        self.scan(source, self.entries(1))
        with patch.object(publishing, 'prepare_publication', return_value={'id': 'p1', 'status': 'draft'}) as prepare, patch.object(publishing, 'submit_publication', side_effect=[ValueError('not logged in'), {'status': 'submitting'}]) as submit, patch.object(publishing, 'list_publications', return_value=[{'id': 'p1', 'status': 'draft'}]):
            workflows.handoff_completed(self.complete, lambda episode_id: ({}, self.root))
            self.assertTrue(workflows.list_sources()[0]['items'][0]['error'])
            workflows.handoff_completed(self.complete, lambda episode_id: ({}, self.root))
            self.assertEqual(prepare.call_count, 1)
            self.assertEqual(submit.call_count, 2)
            self.assertEqual(workflows.list_sources()[0]['items'][0]['error'], '')

    def test_uncertain_and_disabled_or_edited_source_never_resubmits(self):
        source = self.automatic_source()
        self.scan(source, self.entries(1))
        with patch.object(publishing, 'prepare_publication', return_value={'id': 'p1', 'status': 'uncertain'}), patch.object(publishing, 'submit_publication') as submit:
            workflows.handoff_completed(self.complete, lambda episode_id: ({}, self.root))
            submit.assert_not_called()
        with patch.object(publishing, 'list_publications', return_value=[{'id': 'p1', 'status': 'uncertain'}]), patch.object(publishing, 'submit_publication') as submit:
            workflows.handoff_completed(self.complete, lambda episode_id: ({}, self.root))
            submit.assert_not_called()
        self.source(id=source['id'], auto_publish=True, publication={'rights_confirmed': True}, name='Edited')
        with patch.object(publishing, 'list_publications', return_value=[{'id': 'p1', 'status': 'draft'}]), patch.object(publishing, 'submit_publication') as submit:
            workflows.handoff_completed(self.complete, lambda episode_id: ({}, self.root))
            submit.assert_not_called()
        self.source(id=source['id'], enabled=False, auto_publish=True, publication={'rights_confirmed': True})
        with patch.object(publishing, 'prepare_publication') as prepare, patch.object(publishing, 'submit_publication') as submit:
            workflows.handoff_completed(self.complete, lambda episode_id: ({}, self.root))
            prepare.assert_not_called()
            submit.assert_not_called()

    def test_one_failed_item_does_not_starve_next_item(self):
        source = self.automatic_source()
        self.scan(source, self.entries(1))
        self.scan(source, self.entries(2, 1))
        with patch.object(publishing, 'prepare_publication', side_effect=[ValueError('private-token'), {'id': 'p2', 'status': 'draft'}]) as prepare, patch.object(publishing, 'submit_publication') as submit:
            workflows.handoff_completed(self.complete, lambda episode_id: ({}, self.root))
            self.assertEqual(prepare.call_count, 2)
            submit.assert_called_once()
        state = workflows.list_sources()[0]
        self.assertNotIn('private-token', json.dumps(state))
        self.assertTrue(any(item.get('error') for item in state['items']))
        self.assertTrue(any(item.get('publication_id') == 'p2' for item in state['items']))

    def test_disable_while_loading_package_never_creates_orphan_auto_draft(self):
        source = self.automatic_source()
        self.scan(source, self.entries(1))
        def provider(_):
            self.source(id=source['id'], enabled=False, auto_publish=True, publication={'rights_confirmed': True})
            return {}, self.root
        with patch.object(publishing, 'prepare_publication') as prepare, patch.object(publishing, 'submit_publication') as submit:
            workflows.handoff_completed(self.complete, provider)
            prepare.assert_not_called()
            submit.assert_not_called()


if __name__ == '__main__':
    unittest.main()
