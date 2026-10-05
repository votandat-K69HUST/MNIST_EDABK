"""Sinh dataset theo config.json (xem README.md). Chay: python generate.py [--n-train N] [--workers W] ..."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from common import engine  # noqa: E402

if __name__ == "__main__":
    engine.cli(os.path.join(HERE, "config.json"))
