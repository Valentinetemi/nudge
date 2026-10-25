"""Put the repo's scripts directory on the import path for tests."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for extra in (ROOT / "scripts", ROOT / "aws" / "src"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))
