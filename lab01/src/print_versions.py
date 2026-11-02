"""Print the Python version and the version of every package in requirements.txt.

The output is also written to lab01/results/versions.txt.
"""
import platform
import sys
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS = Path(__file__).resolve().parents[1] / "results"


def read_requirements(path):
    packages = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name, _, pinned = line.partition("==")
        packages.append((name.strip(), pinned.strip()))
    return packages


def main():
    lines = [
        f"Python: {sys.version.split()[0]}",
        f"Implementation: {platform.python_implementation()}",
        f"Platform: {platform.platform()}",
        f"Processor: {platform.processor() or platform.machine()}",
        "",
        f"{'package':<20} {'installed':<28} {'pinned':<28}",
    ]
    for name, pinned in read_requirements(ROOT / "requirements.txt"):
        try:
            installed = metadata.version(name)
        except metadata.PackageNotFoundError:
            installed = "NOT INSTALLED"
        flag = "" if installed == pinned else "  <-- mismatch"
        lines.append(f"{name:<20} {installed:<28} {pinned:<28}{flag}")

    text = "\n".join(lines)
    print(text)
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "versions.txt").write_text(text + "\n")


if __name__ == "__main__":
    main()
