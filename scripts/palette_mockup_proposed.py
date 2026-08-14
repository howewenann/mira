"""Run the MIRA palette mockup with the proposed shared lifecycle colors."""

try:
    from scripts.palette_mockup_original import PaletteMockup
except ModuleNotFoundError:  # Direct execution adds scripts/, not the repository root.
    from palette_mockup_original import PaletteMockup


class ProposedPaletteMockup(PaletteMockup):
    """Option 2 shared by header and tool lifecycle labels through TCSS."""

    TITLE = "MIRA · proposed palette"

    CSS = PaletteMockup.CSS + """
    /* Muted Option 2 shared by header and tool lifecycle labels. */
    .lifecycle.starting,
    .lifecycle.preparing {
        color: #9b91b8;
    }

    .lifecycle.running { color: #5f9fc7; }

    .lifecycle.ready,
    .lifecycle.completed {
        color: #6fa884;
    }

    .lifecycle.cancelling,
    .lifecycle.cancelled {
        color: #b89b59;
    }

    .lifecycle.error,
    .lifecycle.failed {
        color: #d77d79;
    }

    /* Muted filled controls: visibly clickable without dominating the header. */
    #artifact-status-button,
    .artifact-control.goal-control {
        background: #46354d;
        color: #e0b5ea;
    }

    .artifact-control.plan-control {
        background: #394329;
        color: #d0e19a;
    }

    #mcp-status-button,
    .mcp-control {
        background: #386b69;
        color: #d2f0ed;
    }

    #header-control-separator,
    .control-separator {
        display: none;
    }

    #artifact-status-button:hover,
    #artifact-status-button:focus,
    .artifact-control:hover,
    .artifact-control:focus {
        background: #46354d;
        color: #e0b5ea;
        tint: transparent;
    }

    .artifact-control.plan-control:hover,
    .artifact-control.plan-control:focus {
        background: #394329;
        color: #d0e19a;
        tint: transparent;
    }

    #mcp-status-button:hover,
    #mcp-status-button:focus,
    .mcp-control:hover,
    .mcp-control:focus {
        background: #386b69;
        color: #d2f0ed;
        tint: transparent;
    }
    """


if __name__ == "__main__":
    ProposedPaletteMockup().run()
