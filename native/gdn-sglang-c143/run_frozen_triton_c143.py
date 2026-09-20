# SPDX-License-Identifier: AGPL-3.0-only
#!/usr/bin/env python3
"""Default-off entry point for the frozen Triton c143 GPU microgate."""

from frozen_triton_executor.main import main


if __name__ == "__main__":
    raise SystemExit(main())
