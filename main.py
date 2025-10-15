"""Project entrypoint for makkah_time_series

Commands:
    python main.py analyze   # runs exploratory analysis
    python main.py train     # trains model and saves model bundle
    python main.py result    # runs result generation (predictions, plots, metrics)

This simple wrapper imports and executes the scripts in `scripts/`.
"""

import sys
import subprocess
import os

ROOT = os.path.dirname(__file__)
SCRIPTS_DIR = os.path.join(ROOT, "scripts")

COMMANDS = {
    "analyze": os.path.join(SCRIPTS_DIR, "linear_analysis.py"),
    "train": os.path.join(SCRIPTS_DIR, "training.py"),
    "result": os.path.join(SCRIPTS_DIR, "result.py"),
}


SEQUENCE = ["analyze", "train", "result"]


def run_script(path):
    if not os.path.exists(path):
        print(f"Script not found: {path}")
        return 2
    # run with same python interpreter
    return subprocess.call([sys.executable, path])


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python main.py [analyze|train|result|all]")
        sys.exit(1)

    cmd = sys.argv[1]
    if cmd == "all":
        for step in SEQUENCE:
            print(f"== Running: {step} ==")
            rc = run_script(COMMANDS[step])
            if rc != 0:
                print(f"Step '{step}' failed with exit code {rc}. Aborting.")
                sys.exit(rc)
        print("All steps completed successfully.")
        sys.exit(0)

    if cmd not in COMMANDS:
        print("Usage: python main.py [analyze|train|result|all]")
        sys.exit(1)

    script_path = COMMANDS[cmd]
    rc = run_script(script_path)
    sys.exit(rc)
