"""Compare mounted widgets and rendering work against another MIRA checkout.

Run with: conda run -n mira python tests/manual/benchmark_grouped_transcript.py CHECKOUT
Uses deterministic model-free fixtures, 250 saved records and 200 public deltas.
Outputs three samples; it does not assert machine-dependent timing thresholds.
"""

import argparse
import asyncio
import json
from pathlib import Path
import sys
import time
from unittest.mock import patch

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('checkout', type=Path, nargs='?', default=Path(__file__).resolve().parents[2])
arguments = parser.parse_args()
sys.path.insert(0, str(arguments.checkout.resolve()))
from textual.widget import Widget
from mira.api import MessageEvent, RuntimeEvent
from tests.test_textual_app import make_app, wait_until
from ui.textual.widgets.chat_log import ChatLog


async def main():
    measurements = []
    for repeat in range(3):
        app = make_app()
        async with app.run_test(size=(110, 40)) as pilot:
            await pilot.pause()
            chat = app.query_one(ChatLog)
            chat.clear_log()
            await pilot.pause()
            events = []
            for turn in range(50):
                for kind, data in [('user', {'text': 'question'}),
                                   ('reasoning', {'text': 'thinking ' * 10}),
                                   ('tool_call', {'name': 'read_file', 'args': {'file_path': 'a'}, 'call_id': f'call-{turn}'}),
                                   ('tool_result', {'name': 'read_file', 'output': 'result', 'call_id': f'call-{turn}', 'duration_ms': 20}),
                                   ('assistant', {'text': 'Answer ' * 10})]:
                    events.append({'type': kind, 'id': len(events)+1, 'turn_id': f'turn-{turn}',
                                   'created_at': '2026-11-27T01:00:00+00:00', **data})
            operations = {'mount': 0, 'remove': 0}
            mount, remove = Widget.mount, Widget.remove
            def counted_mount(owner, *widgets, **kwargs):
                operations['mount'] += len(widgets)
                return mount(owner, *widgets, **kwargs)
            def counted_remove(owner, *args, **kwargs):
                operations['remove'] += 1
                return remove(owner, *args, **kwargs)
            async def settled():
                await wait_until(lambda: all(turn._composed and not turn._moving and
                    all(widget.is_mounted for widget in turn.rows) for turn in chat.query('.assistant-turn')), timeout=20)
                await pilot.pause()
            with patch.object(Widget, 'mount', counted_mount), patch.object(Widget, 'remove', counted_remove):
                start = time.perf_counter()
                chat.restore_session({'events': events})
                await settled()
                restore = time.perf_counter()-start
                widgets = len(chat.query('*'))
                restore_operations = dict(operations)
                start = time.perf_counter()
                identity = dict(session_id=app.core_session.id, turn_id='benchmark-stream')
                frontend = app.application.frontend
                frontend.emit(RuntimeEvent(kind='session', state='turn_started', **identity))
                frontend.emit(MessageEvent(phase='user', text='stream', **identity))
                for _ in range(200):
                    frontend.emit(MessageEvent(phase='content', text='fragment ', message_id='stream', **identity))
                    await asyncio.sleep(0)
                frontend.emit(MessageEvent(phase='end', message_id='stream', **identity))
                frontend.emit(RuntimeEvent(kind='session', state='ready', **identity))
                await settled()
                streaming = time.perf_counter()-start
            measurements.append({'records': len(events), 'widgets': widgets, 'restore_s': round(restore, 4),
                'restore_operations': restore_operations, 'stream200_s': round(streaming, 4),
                'stream_added_mounts': operations['mount']-restore_operations['mount'],
                'stream_removals': operations['remove']-restore_operations['remove']})
    print(json.dumps(measurements))


asyncio.run(main())
