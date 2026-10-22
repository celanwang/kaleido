"""Prove that a copied delivery installs without the checkout or its project environment."""

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from kaleido.experiments import write_json
from kaleido.repro.integrity import file_info


def check(delivery):
    delivery = Path(delivery).resolve()
    cache = subprocess.check_output(["uv", "cache", "dir"], text=True).strip()
    with tempfile.TemporaryDirectory(prefix="kaleido-consumer-") as directory:
        directory = Path(directory)
        copied = directory / "delivery"
        shutil.copytree(delivery, copied)
        command = [
            str(Path(sys.executable).resolve()),
            "-I",
            str(copied / "install.py"),
            "--env",
            str(directory / "venv"),
            "--offline",
            "--cache-dir",
            cache,
        ]
        process = subprocess.run(command, cwd=directory, text=True, capture_output=True)
        if process.returncode:
            raise RuntimeError(process.stdout + process.stderr)
        report = json.loads(process.stdout.splitlines()[-1])
        if not report.get("exact_token_replay") or report.get("requests", 0) < 1:
            raise ValueError("consumer did not report a non-empty successful replay")
        return {
            **report,
            "outside_checkout": True,
            "base_interpreter": command[0],
            "delivery_manifest": file_info(delivery / "manifest.json"),
            "checker": file_info(Path(__file__)),
            "method": "copied kit; base Python -I; fresh venv; pylock artifact URLs; offline",
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--delivery", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    result = check(args.delivery)
    write_json(args.report, result)
    print(json.dumps({"status": "completed", "report": str(args.report.resolve())}))
