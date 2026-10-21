"""Lets `python -m mafdet` work like the `mafdet` command."""
import sys

from .cli import main

sys.exit(main())
