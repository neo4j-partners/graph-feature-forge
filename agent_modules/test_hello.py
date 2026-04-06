"""Sanity-check script — prints env + argv from the remote side."""

from __future__ import annotations

import os
import sys


def main() -> None:
    print("=== semantic_auth / test_hello ===")
    print(f"python:   {sys.version}")
    print(f"cwd:      {os.getcwd()}")
    print(f"argv:     {sys.argv!r}")
    print("databricks env keys:")
    for key in sorted(os.environ):
        if key.startswith("DATABRICKS_"):
            print(f"  {key}=<set>")
    print("=== end ===")


if __name__ == "__main__":
    main()
