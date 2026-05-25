# Local checks before submission

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TEXT_SUFFIXES = {
    # Only read text files Model files are checked by filename only
    ".csv",
    ".html",
    ".md",
    ".py",
    ".txt",
}

SKIP_DIRS = {
    # Skip generated folders and editor folders during normal scans
    ".git",
    ".idea",
    ".mplcache",
    "venv",
    ".venv",
    "__pycache__",
}

REQUIRED_FILES = [
    # Files that should exist in the final project
    "README.md",
    "requirements.txt",
    "player.py",
    "main.py",
    "outputs/traininghistory.csv",
    "outputs/milestones.csv",
    "outputs/comparisonsummary.csv",
    "outputs/comparisongames.csv",
    "outputs/models/afterstate/twoplayer.pth",
    "outputs/models/afterstate/multiplayer.pth",
]

PLOT_ARTIFACT_DIRS = [
    # Report images are kept in the report folder, not in this project
    "outputs/figures",
    "outputs/plots",
]

STALE_TEXT = [
    # Old names from development They should not appear in the final code
    "projectsummary.md",
    "scripts/resume.py",
    "scripts/smoke_routing.py",
    "scripts/checkrouting.py",
    "scripts/multiplayer.py",
    "scripts/compare_models.py",
    "scripts/train_multiplayer.py",
    "scripts/diagnose.py",
    "scripts/evaluate.py",
    "scripts/makeplots.py",
    "scripts/makereportplots.py",
    "outputs/validation.md",
    "best.pth",
    "challenger.pth",
    "multiplayer_challenger_best.pth",
    "RL agent",
    "Tournament Player starting",
    "Locked route:",
    "Watch our model",
    "our model",
    "will play randomly",
    "Afterstate agent not available",
    "fallback model loaded",
]


def iter_project_files():
    # All checks use the same file walk
    for path in PROJECT_ROOT.rglob("*"):
        rel_parts = path.relative_to(PROJECT_ROOT).parts
        if any(part in SKIP_DIRS for part in rel_parts):
            continue
        yield path


def check_required_files():
    # Fail if important project files are missing
    missing = [name for name in REQUIRED_FILES if not (PROJECT_ROOT / name).exists()]
    if missing:
        raise AssertionError("Missing required files:\n  " + "\n  ".join(missing))


def check_clean_names():
    # Do not rename files inside server code from the course
    bad = []
    for path in iter_project_files():
        rel = path.relative_to(PROJECT_ROOT)
        if rel.parts and rel.parts[0] == "server":
            continue
        # Project filenames should use the new clean names
        if "_" in path.name or "-" in path.name:
            bad.append(str(rel))
    if bad:
        raise AssertionError("Unexpected underscore/dash filenames:\n  " + "\n  ".join(sorted(bad)))


def check_no_cache_dirs():
    # Do not skip __pycache__ here This check is meant to find it
    ignored = SKIP_DIRS - {"__pycache__"}
    caches = []
    for path in PROJECT_ROOT.rglob("__pycache__"):
        rel_parts = path.relative_to(PROJECT_ROOT).parts
        if any(part in ignored for part in rel_parts):
            continue
        # Use short paths so the error is easy to read
        caches.append(str(path.relative_to(PROJECT_ROOT)))
    if caches:
        raise AssertionError("Remove generated cache folders:\n  " + "\n  ".join(sorted(caches)))


def check_no_plot_artifacts():
    # Keep CSV data and models here Keep report plots in the report folder
    plot_files = []
    for dirname in PLOT_ARTIFACT_DIRS:
        directory = PROJECT_ROOT / dirname
        if not directory.exists():
            continue
        plot_files.extend(
            str(path.relative_to(PROJECT_ROOT))
            for path in directory.rglob("*")
            if path.is_file()
        )
    if plot_files:
        raise AssertionError("Remove generated plot artifacts:\n  " + "\n  ".join(sorted(plot_files)))


def check_no_stale_text():
    # Skip this files own list of old words
    hits = []
    for path in iter_project_files():
        if not path.is_file() or path.suffix not in TEXT_SUFFIXES:
            continue
        if path.name == "checksubmission.py":
            continue
        text = path.read_text(errors="ignore")
        for needle in STALE_TEXT:
            if needle in text:
                # Show both the file and the old word that was found
                hits.append(f"{path.relative_to(PROJECT_ROOT)}: {needle}")
    if hits:
        raise AssertionError("Stale references found:\n  " + "\n  ".join(sorted(hits)))


def check_python_syntax():
    # Check Python syntax without running the files
    failures = []
    for path in iter_project_files():
        if not path.is_file() or path.suffix != ".py":
            continue
        rel = path.relative_to(PROJECT_ROOT)
        try:
            source = path.read_text()
            compile(source, str(rel), "exec")
        except Exception as exc:
            failures.append(f"{rel}: {exc}")
    if failures:
        raise AssertionError("Python syntax failures:\n  " + "\n  ".join(failures))


def run_routing_check():
    # Run route checks like a normal terminal command
    result = subprocess.run(
        [sys.executable, "-B", "scripts/checkroute.py"],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        details = (result.stdout + result.stderr).strip()
        raise AssertionError(f"Routing check failed:\n{details}")


def run_core_logic_check():
    # Run small board and reward checks
    result = subprocess.run(
        [sys.executable, "-B", "scripts/checklogic.py"],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        details = (result.stdout + result.stderr).strip()
        raise AssertionError(f"Core logic check failed:\n{details}")


def main():
    # Print checks in order so failures are easy to find
    checks = [
        ("required files", check_required_files),
        ("filename cleanup", check_clean_names),
        ("cache cleanup", check_no_cache_dirs),
        ("plot cleanup", check_no_plot_artifacts),
        ("stale references", check_no_stale_text),
        ("python syntax", check_python_syntax),
        ("routing rules", run_routing_check),
        ("core logic", run_core_logic_check),
    ]

    for label, check in checks:
        # Run one check at a time for a clear OK or fail message
        check()
        print(f"OK: {label}")

    print("Submission checks passed.")


if __name__ == "__main__":
    main()
