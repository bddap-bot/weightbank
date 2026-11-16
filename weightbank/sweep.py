from __future__ import annotations

import subprocess
import sys

from weightbank.configs import CONFIGS, SEEDS
from weightbank.train import run_path


def main() -> None:
    for seed in SEEDS:
        for config in CONFIGS:
            if run_path(config, seed).exists():
                continue
            subprocess.run(
                [sys.executable, "-m", "weightbank.train", config, "--seed", str(seed)], check=True
            )


if __name__ == "__main__":
    main()
