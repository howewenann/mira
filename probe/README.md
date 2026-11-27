# Standalone Textual transcript probes

Run from this directory with an activated MIRA environment (`cd probe`). These
scripts use Textual and Rich, with no production MIRA imports. The integration
probe imports only its sibling `keyline_probe.py`. Contributor checks use
`conda run -n mira python ...` in place of `python ...`. Results below were
checked with **Textual 8.2.8**.

## Superseded activity-box experiments

[textual_activity_box_probe.py](textual_activity_box_probe.py) compares separate
full borders, native partial borders, manually joined separators, and flat rows.
Five entries measure 15, 11, 11 and 5 terminal rows respectively. Selection,
keyboard toggling and right-aligned durations pass at widths 60 and 110, but
geometry alone does not establish correct border junctions or hover rendering.

[textual_activity_box_probe_v2.py](textual_activity_box_probe_v2.py) adds manually
rendered vertical border cells and explicit hover/focus refresh to the joined
variant. It preserves the same geometry and duration checks while experimenting
with keeping border cells out of the selected background. Both versions remain
historical experiments; their custom separators/rendering are superseded by
native parent keylines and are not production recommendations.

```powershell
python textual_activity_box_probe.py --mode native
python textual_activity_box_probe_v2.py --mode joined
python textual_activity_box_probe.py --check --out ../.phase3-followup-validation/activity-v1
python textual_activity_box_probe_v2.py --check --out ../.phase3-followup-validation/activity-v2
```

## Validated native keyline solution

[keyline_probe.py](keyline_probe.py) is the approved minimal reference, moved
here from `tests/probes/`. Native ListView, ListItem, Horizontal, Label and parent
keylines give five one-line activities an 11-row joined outline. Native keyboard
selection works and duration right edges agree at widths 60 and 110.

[transcript_keyline_probe.py](transcript_keyline_probe.py) extends that reference
with natural conversation scrolling, larger histories, details and final-response
spacing. Enter/click selects or closes a detail, Up/Down moves the native list
cursor, and A simulates another completion when the composer is not focused.
The composer accepts text. Duration values are fixtures, not execution timings.

```powershell
python keyline_probe.py --rows 50
python keyline_probe.py --check --out ../.phase3-followup-validation/keyline
python transcript_keyline_probe.py
python transcript_keyline_probe.py --check --out ../.phase3-followup-validation/integration
```

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

## Summary focus overlay reproduction

[summary_focus_probe.py](summary_focus_probe.py) compares the historical
`f98e83b` two-row summary with a focus outline (A) against the same widget with
only that outline removed (B). Textual's outline paints over both content rows;
height and content size still report two rows. B keeps the text visible with
the same background-based hover/focus feedback and click/Enter/Space behavior.
Checks exercise both variants at widths 60 and 110 and export their real frames.
The historical A variant deliberately retains the bug as a reference.

```powershell
python summary_focus_probe.py
python summary_focus_probe.py --check --out ../.phase3-followup-validation/summary
```

Check modes export SVGs (and integration observations). Use `--out` to keep new
exports outside the reference artifact directories. Generated caches and repeat
screenshots are not source changes; retain the existing approved frames rather
than committing another copy. Production regressions and limitations are in
[the Phase 3 follow-up report](../tests/manual/phase3-followup-validation.md).
