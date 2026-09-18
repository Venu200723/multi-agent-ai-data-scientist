"""Allow `python -m multi_agent_ds ...` to delegate to the CLI."""

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
