#!/usr/bin/env python
"""Run the maintained live Rubric probe, including --dry-run and --synthetic.

    python tests/probes/probe_mira_rubric_offload_live.py --synthetic --dry-run

Both historical commands exercise the same MIRA verifier/grader and native
checkpoint reconstruction. The implementation lives beside this entry point.
"""

from mira_rubric_live_probe_fixed import main


if __name__ == "__main__":
    raise SystemExit(main())
