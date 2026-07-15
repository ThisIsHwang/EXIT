#!/usr/bin/env python3
"""Compatibility entry point for the historical misspelled filename.

Use ``train/evaluate.py`` for new integrations.
"""

from __future__ import annotations

import warnings

try:  # ``python -m train.evalutate``
    from .evaluate import *  # noqa: F401,F403
    from .evaluate import main
except ImportError:  # ``python train/evalutate.py``
    from evaluate import *  # type: ignore  # noqa: F401,F403
    from evaluate import main  # type: ignore


if __name__ == "__main__":
    warnings.warn(
        "train/evalutate.py is deprecated; use train/evaluate.py",
        DeprecationWarning,
        stacklevel=2,
    )
    main()
