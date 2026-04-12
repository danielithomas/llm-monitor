"""Entry point for ``python -m clawmeter`` and the ``clawmeter`` script.

Handles SIGPIPE, BrokenPipeError, and KeyboardInterrupt gracefully.
"""

import os
import signal
import sys


def main() -> None:
    # SIGPIPE: silent exit (for piping to head, grep -q, etc.)
    try:
        signal.signal(signal.SIGPIPE, signal.SIG_DFL)
    except (AttributeError, OSError):
        pass  # Windows doesn't have SIGPIPE

    from clawmeter.cli import cli

    try:
        cli(standalone_mode=False)
    except SystemExit as e:
        sys.exit(e.code)
    except BrokenPipeError:
        # Suppress traceback on broken pipe
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, sys.stdout.fileno())
        sys.exit(0)
    except KeyboardInterrupt:
        sys.exit(130)


if __name__ == "__main__":
    main()
