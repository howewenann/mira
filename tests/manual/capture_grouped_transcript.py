"""Export actual Textual frames for the grouped transcript manual review.

Run: conda run -n mira python tests/manual/capture_grouped_transcript.py
The deterministic public events below exercise presentation, not model quality.
"""

import asyncio
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
# The automation shell sets NO_COLOR; export the actual colour terminal design.
os.environ.pop('NO_COLOR', None)
os.environ['TERM'] = 'xterm-256color'
os.environ['COLORTERM'] = 'truecolor'
os.environ['FORCE_COLOR'] = '1'

from mira.api import MessageEvent, RuntimeEvent, ToolEvent
from tests.test_grouped_transcript import settle
from tests.test_textual_app import make_app
from ui.textual.widgets.assistant_turn import AssistantTurn
from ui.textual.widgets.chat_log import ChatLog


async def main():
    destination = ROOT / 'tests/manual/artifacts/phase3'
    destination.mkdir(exist_ok=True)
    app = make_app()
    async with app.run_test(size=(120, 44)) as pilot:
        await pilot.pause()
        chat = app.query_one(ChatLog)
        chat.clear_log()
        await pilot.pause()
        emit = app.application.frontend.emit
        identity = dict(session_id=app.core_session.id, turn_id='visual-review')
        emit(RuntimeEvent(kind='session', state='turn_started', **identity))
        emit(MessageEvent(phase='user', text='Read the project and run its focused checks.', **identity))
        emit(MessageEvent(phase='reasoning', text='\n'.join(f'Thinking line {i}: inspect the existing project.' for i in range(400)), message_id='first', **identity))
        emit(MessageEvent(phase='content', text='I will read the project and run the checks.', message_id='first', **identity))
        emit(MessageEvent(phase='end', message_id='first', **identity))
        emit(ToolEvent(phase='start', name='read_file', arguments={'file_path': '/README.md'}, tool_call_id='read', **identity))
        emit(ToolEvent(phase='completed_result', name='read_file', result='# Project\nExisting project instructions.', duration_ms=1000, tool_call_id='read', **identity))
        emit(MessageEvent(phase='reasoning', text='Check the focused tests next.', message_id='second', **identity))
        emit(MessageEvent(phase='end', message_id='second', **identity))
        command = 'python -m pytest tests/core/' + 'very_long_test_path/' * 10 + '\n--verbose\n--capture=no'
        emit(ToolEvent(phase='start', name='execute', arguments={'command': command}, tool_call_id='execute', **identity))
        await settle(app, pilot)
        turn = chat.query_one(AssistantTurn)

        async def capture(name, target=None):
            await pilot.pause()
            if target is not None:
                if target is turn.details:
                    target = turn.close_button
                chat.scroll_to_widget(target, top=True, animate=False, immediate=True)
            await pilot.pause()
            exported = app.export_screenshot()
            (destination / f'{name}.svg').write_text('\n'.join(line.rstrip() for line in exported.splitlines()) + '\n', encoding='utf-8')

        await capture('running', turn.summary)
        emit(ToolEvent(phase='completed_result', name='execute', result='\n'.join(f'Test output line {i}' for i in range(300)) + '\nAll checks passed.', duration_ms=8000, tool_call_id='execute', **identity))
        emit(MessageEvent(phase='content', text='The project instructions are available and the focused checks passed.', message_id='final', **identity))
        emit(MessageEvent(phase='end', message_id='final', **identity))
        emit(RuntimeEvent(kind='session', state='ready', **identity))
        await settle(app, pilot)
        await capture('completed', turn.summary)
        await capture('final-spacing', turn.summary)
        turn.toggle_history()
        await settle(app, pilot)
        await capture('history', turn.summary)
        thought = next(row for row in turn.completed_rows if row.category == 'Thinking')
        turn.select_activity(thought)
        await settle(app, pilot)
        await capture('thinking', turn.summary)
        tool = next(row for row in turn.completed_rows if getattr(row.widget, 'tool_name', '') == 'execute')
        turn.select_activity(tool)
        await settle(app, pilot)
        tool.widget.args_collapsible.collapsed = False
        tool.widget.output_collapsible.collapsed = False
        await settle(app, pilot)
        await capture('details', turn.details)
        turn.history.index = list(turn.history.children).index(tool)
        turn.history.focus(scroll_visible=False)
        await capture('highlighted', turn.summary)
        await pilot.hover(tool, offset=(5, 0))
        await capture('hovered', turn.summary)
        await pilot.resize_terminal(60, 40)
        await settle(app, pilot)
        await capture('narrow', turn.summary)
    print('Exported running, completed, history, thinking, details, highlighted, hovered and narrow frames.')


asyncio.run(main())
