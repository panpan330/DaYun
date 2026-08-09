from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PROJECTS = (
    ("ai-service", REPO_ROOT / "projects" / "ai-service", "python"),
    ("java-business-service", REPO_ROOT / "projects" / "java-business-service", "maven"),
)


def main() -> int:
    for project_name, project_dir, project_type in PROJECTS:
        if not project_dir.exists():
            print(f"[regression] missing project directory: {project_dir}", flush=True)
            return 1

        if project_type == "maven":
            commands = (
                ("run maven tests", ["mvn", "-q", "test"]),
            )
        else:
            commands = (
                ("sync dependencies", ["uv", "sync", "--frozen"]),
                (
                    "compile python files",
                    [
                        "uv",
                        "run",
                        "python",
                        "-m",
                        "compileall",
                        "-q",
                        "-x",
                        ".venv|__pycache__",
                        ".",
                    ],
                ),
                ("run pytest", ["uv", "run", "pytest"]),
            )
        for label, command in commands:
            exit_code = run_command(project_name, label, command, project_dir)
            if exit_code != 0:
                return exit_code
    print("[regression] all checks passed", flush=True)
    return 0


def run_command(
    project_name: str,
    label: str,
    command: list[str],
    cwd: Path,
) -> int:
    print(f"[regression] {project_name}: {label}", flush=True)
    executable = command[0]
    resolved = shutil.which(executable)
    if resolved is None:
        print(
            f"[regression] {executable} was not found. Install it before running regression.",
            flush=True,
        )
        return 127
    if resolved.lower().endswith((".cmd", ".bat")):
        # Windows: CreateProcess 不能直接执行 .cmd/.bat，需要经 cmd /c 包装
        command = ["cmd", "/c", *command]
    try:
        completed = subprocess.run(command, cwd=cwd, check=False)
    except OSError:
        print(
            f"[regression] {project_name}: {label} failed to start "
            f"(command: {' '.join(command)})",
            flush=True,
        )
        return 126
    if completed.returncode != 0:
        print(
            f"[regression] {project_name}: {label} failed with "
            f"exit code {completed.returncode}",
            flush=True,
        )
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
