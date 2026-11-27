"""Native keyline integration probe; run from probe/ with the MIRA environment.

Interactive: conda run -n mira python transcript_keyline_probe.py
Checks/exports: conda run -n mira python transcript_keyline_probe.py --check
No production MIRA module is imported or modified by this probe.
"""

import argparse
import asyncio
from collections import Counter
from importlib.metadata import version
import json
import os
from pathlib import Path
import sys

os.environ.pop('NO_COLOR', None)
os.environ['TERM'] = 'xterm-256color'
os.environ['COLORTERM'] = 'truecolor'
os.environ['FORCE_COLOR'] = '1'
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rich.text import Text
from textual import on
from textual.app import App
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import Button, Collapsible, Input, Label, ListItem, ListView, Static
from tests.probes.keyline_probe import KeylineProbe


STEPS = [
    ('thinking', '', '00:12', '#B8C2C9'),
    ('read_file', '/README.md', '00:01', '#8FA3B8'),
    ('message', '', '00:02', '#5BB8B1'),
    ('execute', 'python -m pytest tests/core/' + 'long_path/' * 25 + '\n--verbose', '00:08', '#8FA3B8'),
    ('rubric', '', '00:02', '#C58FD6'),
]


def activity(index):
    name, preview, duration, color = STEPS[index % len(STEPS)]
    glyph = Label('▸', classes='glyph')
    glyph.styles.color = color
    label = Label(name, classes='name')
    label.styles.color = color
    return ListItem(Horizontal(glyph, label,
        Label(' '.join(preview.split()), markup=False, classes='preview'),
        Label(duration, classes='duration'), classes='activity-content'))


class Summary(Static, can_focus=True):
    BINDINGS = [('enter,space', 'toggle', 'Activity history')]

    def on_click(self, event):
        event.stop()
        self.focus(scroll_visible=False)
        self.action_toggle()

    def action_toggle(self):
        self.parent.toggle()


class ProbeTurn(Vertical):
    def __init__(self, rows, *, id):
        super().__init__(classes='turn', id=id)
        self.expanded = False
        self.selected = None
        self.summary = Summary(classes='summary')
        self.activities = ListView(*(activity(i) for i in range(rows)), initial_index=None, classes='activities')
        self.details = Vertical(classes='details')
        self.active = Static('Running read_file /README.md', classes='live-activity')
        self.running = False
        self.final = Static('The project instructions are available and the focused checks passed.', classes='final')

    def compose(self):
        yield self.summary
        yield self.activities
        yield self.details
        yield self.active
        yield self.final

    def on_mount(self):
        self.activities.display = self.details.display = False
        self.active.display = False
        self.update_header()

    def update_header(self):
        header = Text(('▾' if self.expanded else '▸') + f' {"Working" if self.running else "Worked"} · {len(self.activities.children)} steps')
        # Native docked Label keeps the clock flush right on the first line.
        counts = Text('\n  ')
        categories = {'thinking': 'Thinking', 'read_file': 'Tools', 'message': 'Messages', 'execute': 'Tools', 'rubric': 'Rubric'}
        totals = Counter(categories[STEPS[j % 5][0]] for j in range(len(self.activities.children)))
        for name, color in [('Thinking', '#B8C2C9'), ('Tools', '#8FA3B8'), ('Messages', '#5BB8B1'), ('Rubric', '#C58FD6')]:
            count = totals[name]
            if count:
                if len(counts.plain) > 3:
                    counts.append(' · ')
                counts.append(f'{count} {name[:-1] if count == 1 and name.endswith("s") else name}', style=color)
        header.append_text(counts)
        self.summary.update(header)

    def toggle(self):
        self.expanded = not self.expanded
        self.activities.display = self.expanded
        if not self.expanded and self.selected is not None:
            self.run_worker(self.select(None))
        self.update_header()

    @on(ListView.Selected)
    async def activity_selected(self, event):
        event.stop()
        await self.select(event.item if self.selected is not event.item else None)

    @on(ListView.Highlighted)
    def keep_keyboard_cursor_visible(self, event):
        # An auto-height ListView has no internal overflow to scroll. Use the
        # native ancestor scroll path so keyboard movement reveals its row.
        if event.item is not None and event.list_view.has_focus:
            event.item.scroll_visible(animate=False, immediate=True)

    async def select(self, item):
        if self.selected is not None:
            self.selected.remove_class('selected')
            self.selected.query_one('.glyph', Label).update('▸')
        self.selected = item
        await self.details.remove_children()
        self.details.display = item is not None
        if item is None:
            return
        item.add_class('selected')
        item.query_one('.glyph', Label).update('▾')
        index = list(self.activities.children).index(item)
        name, preview, _, color = STEPS[index % 5]
        close = Button('Close', compact=True, classes='close')
        if name in {'read_file', 'execute'}:
            body = Vertical(
                Collapsible(Static(json.dumps({'command' if name == 'execute' else 'file_path': preview}, indent=2)), title='Arguments', collapsed=True),
                Collapsible(Static('\n'.join(f'Output line {i}' for i in range(350))), title='Output', collapsed=True),
                classes='tool-detail')
            body.border_title = name
        else:
            body = Static('\n'.join(f'Thinking line {i}: full retained detail' for i in range(400))
                          if name == 'thinking' else 'Full retained activity content.', classes='body')
            body.border_title = name
        body.styles.border = ('solid', color)
        await self.details.mount(close, body)

    @on(Button.Pressed, '.close')
    async def close_detail(self, event):
        event.stop()
        await self.select(None)

    async def insert_activity(self):
        self.running = self.active.display = True
        self.final.display = False
        self.update_header()
        await asyncio.sleep(.15)
        await self.activities.append(activity(len(self.activities.children)))
        self.running = self.active.display = False
        self.final.display = True
        self.update_header()


class TranscriptProbe(App):
    CSS = KeylineProbe.CSS.replace('#activities', '.activities') + '''
    #conversation { height: 1fr; padding: 1 2; }
    #composer { dock: bottom; height: 3; }
    .turn { height: auto; border: solid #5bb8b1; padding: 0 1; margin-bottom: 1; }
    .summary { height: 2; width: 1fr; background: #17313a; color: #e8edef; text-style: none; pointer: pointer; }
    .summary:hover, .summary:focus { background: #1b3036; text-style: none; }
    .summary > .clock { dock: right; width: 5; height: 1; color: #82909a; }
    .activities > ListItem.selected { background: #17313a; text-style: none; }
    .details, .body, .tool-detail { height: auto; padding: 0; margin: 0; }
    .details { width: 1fr; }
    .close { height: 1; min-height: 1; border: none; width: auto; min-width: 0; padding: 0 1; margin: 0; }
    .body, .tool-detail { padding: 0 1; }
    .live-activity { height: auto; border: solid #8fa3b8; padding: 0 1; }
    .final { height: auto; margin: 1 0 0 0; padding: 0; }
    Collapsible { height: auto; margin: 0; border: none; padding: 0; background: transparent; }
    CollapsibleTitle { height: auto; padding: 0; text-style: none; }
    CollapsibleTitle:focus { text-style: none; }
    Collapsible > Contents { padding: 0 0 0 3; }
    '''
    BINDINGS = [('a', 'add', 'Simulate completed activity'), ('q', 'quit', 'Quit')]

    def compose(self):
        with VerticalScroll(id='conversation'):
            yield Static('Earlier conversation and completed interaction.')
            yield ProbeTurn(5, id='short-turn')
            yield Static('Inspect the project and run its checks.')
            yield ProbeTurn(50, id='long-turn')
        yield Input(placeholder='Prompt composer | A adds an activity when not typing', id='composer')

    async def on_mount(self):
        for summary in self.query(Summary):
            await summary.mount(Label('01:20', classes='clock'))
        self.query_one(Input).focus()

    async def action_add(self):
        await self.query_one('#long-turn', ProbeTurn).insert_activity()


async def check():
    destination = Path('artifacts')
    destination.mkdir(exist_ok=True)
    app = TranscriptProbe()
    async with app.run_test(size=(110, 44)) as pilot:
        await pilot.pause()
        chat = app.query_one('#conversation', VerticalScroll)
        short = app.query_one('#short-turn', ProbeTurn)
        turn = app.query_one('#long-turn', ProbeTurn)
        observations = {'textual': version('textual')}
        async def capture(name, target):
            await pilot.pause()
            if target is turn.details and target.children:
                target = target.children[0]
            chat.scroll_to_widget(target, top=True, animate=False, immediate=True)
            await pilot.pause()
            Path(destination / f'{name}.svg').write_text(app.export_screenshot(), encoding='utf-8')
        await capture('collapsed-spacing', short)
        observations['collapsed_final_gap'] = short.final.region.y - short.summary.region.bottom
        short.running = short.active.display = True
        short.final.display = False
        short.update_header()
        await capture('running', short)
        short.running = short.active.display = False
        short.final.display = True
        short.update_header()
        turn.toggle()
        await pilot.pause()
        observations['list_height'] = turn.activities.region.height
        observations['expected_list_height'] = 2 * len(turn.activities.children) + 1
        observations['list_scroll_y'] = turn.activities.scroll_y
        observations['list_max_scroll_y'] = turn.activities.max_scroll_y
        await capture('expanded-50', turn.summary)
        first = turn.activities.children[0]
        turn.activities.focus(scroll_visible=False)
        await pilot.press('down', 'enter')
        await pilot.pause()
        observations['keyboard_selected'] = turn.selected is first
        observations['glyph_width_before_hover'] = first.query_one('.glyph').region.width
        glyph_region = first.query_one('.glyph').region
        await pilot.hover(first.query_one('.name'))
        await pilot.pause()
        observations['glyph_geometry_stable_on_hover'] = first.query_one('.glyph').region == glyph_region
        observations['selected_detail_below_rows'] = turn.details.region.y >= turn.activities.region.bottom
        await capture('large-thinking', turn.details)
        observations['thinking_height'] = turn.details.query_one('.body').region.height
        observations['thinking_internal_scroll_allowed'] = turn.details.query_one('.body').allow_vertical_scroll
        observations['selected_final_gap'] = turn.final.region.y - turn.details.region.bottom
        middle = turn.activities.children[20]
        await capture('middle-before-select', middle)
        app.query_one(Input).focus(scroll_visible=False)
        await pilot.pause()
        before = chat.scroll_y
        await pilot.click(middle.query_one('.name'))
        await pilot.pause()
        observations['mouse_selection_scroll_before'] = before
        observations['mouse_selection_scroll_after'] = chat.scroll_y
        observations['mouse_selected'] = turn.selected is middle
        await capture('selected-row', middle)
        await pilot.press(*(['down'] * 35))
        await pilot.pause()
        highlighted = turn.activities.highlighted_child
        observations['keyboard_navigation_keeps_highlight_visible'] = chat.content_region.overlaps(highlighted.region)
        observations['keyboard_list_scroll_y'] = turn.activities.scroll_y
        observations['keyboard_main_scroll_y'] = chat.scroll_y
        await capture('highlighted', middle)
        tool = turn.activities.children[3]
        await capture('tool-before-select', tool)
        await pilot.click(tool.query_one('.name'))
        await pilot.pause()
        for disclosure in turn.details.query(Collapsible):
            disclosure.collapsed = False
        await pilot.pause()
        await capture('tool-details', turn.details)
        chat.scroll_to_widget(turn.activities.children[25], top=True, animate=False, immediate=True)
        await pilot.pause()
        before = chat.scroll_y
        await turn.insert_activity()
        await pilot.pause()
        observations['insertion_scroll_before'] = before
        observations['insertion_scroll_after'] = chat.scroll_y
        observations['duration_right_edges'] = sorted(set(item.query_one('.duration').region.right for item in turn.activities.children))
        await pilot.resize_terminal(60, 40)
        await capture('narrow', turn.activities.children[3])
        observations['narrow_duration_right_edges'] = sorted(set(item.query_one('.duration').region.right for item in turn.activities.children))
        observations['narrow_one_line'] = all(item.region.height == 1 for item in turn.activities.children)
        preview = tool.query_one('.preview')
        preview_text = ''.join(strip.text for strip in preview.render_lines(preview.region.at_offset((0, 0))))
        observations['narrow_preview_text'] = preview_text
        observations['narrow_preview_truncated'] = '…' in preview_text or '...' in preview_text
        await pilot.resize_terminal(150, 44)
        await capture('wide', turn.activities.children[3])
        await turn.select(None)
        await pilot.pause()
        observations['expanded_final_gap'] = turn.final.region.y - turn.activities.region.bottom
        turn.expanded = False
        turn.activities.display = False
        await pilot.pause()
        observations['collapsed_after_detail_gap'] = turn.final.region.y - turn.summary.region.bottom
        assert observations['list_height'] == observations['expected_list_height'] == 101
        assert observations['list_max_scroll_y'] == observations['keyboard_list_scroll_y'] == 0
        assert observations['keyboard_selected'] and observations['mouse_selected']
        assert observations['keyboard_navigation_keeps_highlight_visible']
        assert observations['glyph_geometry_stable_on_hover']
        assert observations['mouse_selection_scroll_before'] == observations['mouse_selection_scroll_after']
        assert observations['insertion_scroll_before'] == observations['insertion_scroll_after']
        assert observations['thinking_height'] >= 400 and not observations['thinking_internal_scroll_allowed']
        assert len(observations['duration_right_edges']) == len(observations['narrow_duration_right_edges']) == 1
        assert observations['narrow_one_line'] and observations['narrow_preview_truncated']
        assert all(observations[name] == 1 for name in ('collapsed_final_gap', 'expanded_final_gap', 'selected_final_gap', 'collapsed_after_detail_gap'))
    Path(destination / 'observations.json').write_text(json.dumps(observations, indent=2), encoding='utf-8')
    print(json.dumps(observations, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    arguments = parser.parse_args()
    if arguments.check:
        asyncio.run(check())
    else:
        TranscriptProbe().run()
