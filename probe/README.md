# Transcript keyline integration probe

This probe was approved for production integration. The approved reference
was found at [tests/probes/keyline_probe.py](../tests/probes/keyline_probe.py).
Its source is unchanged. Both probes were run from this directory using MIRA's
pinned **Textual 8.2.8** environment:

```powershell
conda run -n mira python ../tests/probes/keyline_probe.py --check
conda run -n mira python transcript_keyline_probe.py --check
conda run -n mira python transcript_keyline_probe.py
```

The integration probe uses native ListView, ListItem, Horizontal, Label and
parent keylines. It imports no production MIRA modules. Enter/click selects or
closes a detail, Up/Down moves the native list cursor, and A simulates another
completed activity when the composer is not focused. The composer accepts text.
The simulated duration values are fixtures, not measurements of agent execution.

## Observed behavior

- Fifty completed activities occupy **101 terminal rows** (2N+1). Each activity
  has one content row; native keylines supply shared borders and junctions.
- The list has **zero internal scroll range**. The main conversation scrolls
  through all activities and the 400-line thinking / 350-line tool details.
- Native keyboard selection and mouse selection work. An auto-height ListView's
  native index watcher cannot reveal an offscreen cursor through its own scroll
  range. One Highlighted handler calls native `scroll_visible()` on the item,
  allowing the conversation to reveal it without a custom selection framework.
- Clicking a visible row after focusing the composer preserved the conversation
  offset (52 before and after). Inserting an activity while reading also
  preserved the measured offset and selected detail.
- Glyph width stays at two cells, and its region stays unchanged during hover.
  Label text styles remain fixed; selection changes the glyph direction and
  background inside the keyline boundaries.
- Duration right edges agree across every row at widths 60, 110 and 150. Long
  previews normalize newlines and ellipsize; narrow rows remain one line.
- The selected detail is below the entire activity list. Changing selection
  replaces it. Its close action and collapsing the summary hide it.
- A targeted final-response margin gives **exactly one blank row** with collapsed
  history, expanded history, and an open detail. No final response means no
  final-response margin.

The check mode asserts the recorded layout/interaction outcomes. Raw geometry
is retained in [observations.json](artifacts/observations.json).

## Actual rendered frames

These are Textual screen exports rasterized with headless Chrome, not browser
mockups. The PNGs and original SVGs are stored together. They were visually
reviewed for border alignment, density, highlighting, durations and spacing.

- [Collapsed summary and final spacing](artifacts/collapsed-spacing.png)
- [Expanded 50-row history](artifacts/expanded-50.png)
- [Selected row](artifacts/selected-row.png)
- [Large thinking detail](artifacts/large-thinking.png)
- [Long tool arguments and output](artifacts/tool-details.png)
- [Running interaction](artifacts/running.png)
- [Narrow terminal](artifacts/narrow.png)
- [Wide terminal](artifacts/wide.png)

Production uses this approved arrangement. The probe verifies native layout;
the final production results are recorded in
[the Phase 3 validation report](../tests/manual/phase3-validation.md).

## Existing correction validation

Before this keyline checkpoint, the working correction passed **242 API,
identity, inspection, SQLite and runner tests**. A broader UI run executed 444
tests with five failures. Two new test assumptions (lazy Inspector-button
mounting and delayed argument-editor layout) were corrected; the focused rerun
passed all 11 grouped tests plus the two intermittent UI paths. Git-dialog focus
still failed both on this checkout and unchanged `f55cd08`. Subsequent production
validation is recorded in the linked Phase 3 report.

Native execution on `f55cd08` reproduced missing **read-result content**: the
native tool captured 2,163 characters including `READ_END`, while the tool widget
exposed only a truncated 119-character presentation. The intermediate assistant
message and call identity were present. The correction shows the complete
captured result in grouped details, leaving Inspector presentation unchanged.
Seven parallel native `read_file` calls and their preceding assistant message
passed both live and restored checks. A configured Gemma/LM Studio run also
exposed its full 6,017-character read result after restoration. Complete loss of
a read-call event was not reproduced, so the truncation finding is not a claim
that every instance of the original missing-call report has been explained.
