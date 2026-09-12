"""``python -m anime_tools.exclude``. The CLI itself is
:mod:`anime_tools.exclude._cli`, so importing the package does not run it."""

from anime_tools.exclude._cli import main

raise SystemExit(main())
