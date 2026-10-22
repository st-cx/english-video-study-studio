"""Verify real background-loop wiring without profiles, accounts or publication."""
from __future__ import annotations

import threading
import unittest
from unittest.mock import Mock, patch

from app import publishing, workflows


class _OneCycleStop:
    def __init__(self):
        self.stopped = False
        self.waits = []

    def is_set(self):
        return self.stopped

    def wait(self, timeout):
        self.waits.append(timeout)
        self.stopped = True
        return True


class _InlineThread:
    def __init__(self, target, daemon, name):
        self.target, self.daemon, self.name = target, daemon, name
        self.finished = False

    def start(self):
        self.target()
        self.finished = True


class PublicationBackgroundTests(unittest.TestCase):
    def _one_cycle(self, *, handoff_error=False, source_error=False):
        stop = _OneCycleStop()
        enqueue, lookup, provider = Mock(), Mock(), Mock()
        with patch.object(workflows, 'list_sources', side_effect=RuntimeError('source failure') if source_error else None,
                          return_value=[]), \
                patch.object(workflows, 'handoff_completed', side_effect=RuntimeError('handoff failure') if handoff_error else None) as handoff, \
                patch.object(publishing, 'process_ready_publications') as dispatch, \
                patch.object(workflows.threading, 'Thread', side_effect=_InlineThread) as thread:
            result = workflows.run_background(enqueue, lookup, provider, stop=stop)
        self.assertIs(result, stop)
        self.assertEqual(stop.waits, [60])
        self.assertTrue(stop.is_set())
        dispatch.assert_called_once_with(provider)
        self.assertTrue(thread.call_args.kwargs['daemon'])
        self.assertEqual(thread.call_args.kwargs['name'], 'english-resources')
        return handoff, lookup, provider

    def test_each_background_cycle_dispatches_independent_publication_drafts(self):
        handoff, lookup, provider = self._one_cycle()
        handoff.assert_called_once_with(lookup, provider)

    def test_handoff_failure_does_not_starve_independent_publication_dispatch(self):
        handoff, _, _ = self._one_cycle(handoff_error=True)
        handoff.assert_called_once()

    def test_source_listing_failure_does_not_starve_independent_publication_dispatch(self):
        handoff, _, _ = self._one_cycle(source_error=True)
        handoff.assert_not_called()

    def test_real_background_thread_exits_when_dispatcher_sets_stop(self):
        stop = threading.Event()
        real_thread = threading.Thread
        threads = []
        provider = Mock()

        def create_thread(*args, **kwargs):
            worker = real_thread(*args, **kwargs)
            threads.append(worker)
            return worker

        def dispatch(package_provider):
            self.assertIs(package_provider, provider)
            stop.set()

        with patch.object(workflows, 'list_sources', return_value=[]), \
                patch.object(workflows, 'handoff_completed'), \
                patch.object(publishing, 'process_ready_publications', side_effect=dispatch) as ready, \
                patch.object(workflows.threading, 'Thread', side_effect=create_thread):
            try:
                self.assertIs(workflows.run_background(Mock(), Mock(), provider, stop=stop), stop)
                threads[0].join(timeout=2)
                self.assertFalse(threads[0].is_alive())
                ready.assert_called_once_with(provider)
            finally:
                stop.set()
                for worker in threads:
                    worker.join(timeout=2)

    def test_already_stopped_loop_performs_no_source_or_publication_work(self):
        stop = _OneCycleStop()
        stop.stopped = True
        with patch.object(workflows, 'list_sources') as sources, \
                patch.object(workflows, 'handoff_completed') as handoff, \
                patch.object(publishing, 'process_ready_publications') as ready, \
                patch.object(workflows.threading, 'Thread', side_effect=_InlineThread):
            workflows.run_background(Mock(), Mock(), Mock(), stop=stop)
        sources.assert_not_called()
        handoff.assert_not_called()
        ready.assert_not_called()
        self.assertEqual(stop.waits, [])


if __name__ == '__main__':
    unittest.main()
