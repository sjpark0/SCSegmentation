#!/usr/bin/env python3
"""Check method: predicts nothing.  The scorer must give J=F=0 (lower bound)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from methods import common  # noqa: E402


def empty(scene, args):
    return {"kind": "empty"}


if __name__ == "__main__":
    common.run_all(common.parse_args(__doc__), empty)
