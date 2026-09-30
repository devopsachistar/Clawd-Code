"""Cross-platform shell command execution.

POSIX keeps the historical behaviour: ``bash -lc <command>``.

Windows needs explicit shell discovery. ``subprocess`` resolves a bare ``bash``
through the Win32 search order, which checks ``C:\\Windows\\System32`` *before*
``PATH``. That finds the WSL launcher (``System32\\bash.exe``), so commands
silently run inside a Linux VM against ``/mnt/c/...`` instead of on Windows.
WSL also holds a handle on the working directory after the command exits. This
module therefore:

* prefers Git Bash (Git for Windows), found by absolute path and never the WSL
  launcher, so bash syntax keeps working natively on Windows
* falls back to PowerShell (``pwsh`` then Windows PowerShell), then ``cmd``
* always captures stdout/stderr as UTF-8, closes stdin (no hanging prompts),
  enforces the timeout, and kills the whole process tree when it expires

Override discovery with ``INCRESCENT_BASH`` / ``CLAWD_BASH`` (path to a bash
executable).
"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Mapping, Optional

from ..branding import env_lookup

ShellKind = Literal["bash", "sh", "powershell", "cmd"]

IS_WINDOWS = os.name == "nt"
POWERSHELL_UTF8_PREFIX = "[Console]::OutputEncoding=[System.Text.Encoding]::UTF8; $OutputEncoding=[System.Text.Encoding]::UTF8; "
TIMEOUT_EXIT_CODE = 124  # conventional "timed out" exit status (GNU timeout)


@dataclass(frozen=True)
class ShellSpec:
    kind: ShellKind
    executable: str

    def build_argv(self, command: str) -> list[str]:
        if self.kind == "bash":
            return [self.executable, "-lc", command]
        if self.kind == "sh":
            return [self.executable, "-c", command]
        if self.kind == "powershell":
            return [
                self.executable, "-NoLogo", "-NoProfile", "-NonInteractive",
                "-Command", POWERSHELL_UTF8_PREFIX + command,
            ]
        return [self.executable, "/d", "/s", "/c", command]


@dataclass(frozen=True)
class CommandResult:
    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool
    shell: ShellKind
    duration_ms: int


def is_wsl_launcher(path: str | Path) -> bool:
    """True for the WSL ``bash.exe`` shims in System32 / WindowsApps."""
    p = str(path).replace("/", "\\").lower()
    system_root = os.environ.get("SystemRoot", r"C:\Windows").lower()
    return (
        p.startswith(system_root + "\\system32\\")
        or p.startswith(system_root + "\\sysnative\\")
        or "\\windowsapps\\" in p
    )


def _git_bash_candidates() -> list[Path]:
    candidates: list[Path] = []
    git = shutil.which("git")
    if git:
        git_dir = Path(git).resolve().parent  # ...\Git\cmd or ...\Git\bin or ...\Git\mingw64\bin
        for base in (git_dir.parent, git_dir.parent.parent):
            candidates.append(base / "bin" / "bash.exe")
            candidates.append(base / "usr" / "bin" / "bash.exe")
    for env_name in ("ProgramFiles", "ProgramW6432", "ProgramFiles(x86)"):
        root = os.environ.get(env_name)
        if root:
            candidates.append(Path(root) / "Git" / "bin" / "bash.exe")
            candidates.append(Path(root) / "Git" / "usr" / "bin" / "bash.exe")
    local = os.environ.get("LOCALAPPDATA")
    if local:
        candidates.append(Path(local) / "Programs" / "Git" / "bin" / "bash.exe")
    return candidates


def find_bash() -> Optional[str]:
    """Locate a native bash (never the WSL launcher on Windows)."""
    override, _ = env_lookup("BASH")
    if override and Path(override).is_file() and not (IS_WINDOWS and is_wsl_launcher(override)):
        return str(Path(override))
    if not IS_WINDOWS:
        return shutil.which("bash")
    for candidate in _git_bash_candidates():
        if candidate.is_file() and not is_wsl_launcher(candidate):
            return str(candidate)
    found = shutil.which("bash")
    if found and not is_wsl_launcher(found):
        return found
    return None


def find_powershell() -> Optional[str]:
    return shutil.which("pwsh") or shutil.which("powershell")


def find_cmd() -> Optional[str]:
    comspec = os.environ.get("ComSpec")
    if comspec and Path(comspec).is_file():
        return comspec
    return shutil.which("cmd")


def detect_shell(preferred: ShellKind | None = None) -> ShellSpec:
    """Pick the shell for command execution on this platform."""
    if preferred == "powershell":
        ps = find_powershell()
        if ps:
            return ShellSpec("powershell", ps)
        raise RuntimeError("PowerShell (pwsh/powershell) was not found on PATH")
    if preferred == "cmd":
        cmd = find_cmd()
        if cmd:
            return ShellSpec("cmd", cmd)
        raise RuntimeError("cmd.exe was not found")

    bash = find_bash()
    if bash:
        return ShellSpec("bash", bash)
    if not IS_WINDOWS:
        sh = shutil.which("sh") or "/bin/sh"
        return ShellSpec("sh", sh)
    ps = find_powershell()
    if ps:
        return ShellSpec("powershell", ps)
    cmd = find_cmd()
    if cmd:
        return ShellSpec("cmd", cmd)
    raise RuntimeError("No usable shell found (install Git for Windows or PowerShell)")


def _decode(data: bytes | None) -> str:
    if not data:
        return ""
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        import locale

        return data.decode(locale.getpreferredencoding(False) or "utf-8", errors="replace")


def _kill_process_tree(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    try:
        if IS_WINDOWS:
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=15,
            )
        else:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except Exception:
        pass
    try:
        proc.kill()
    except Exception:
        pass


def run_shell_command(
    command: str,
    *,
    cwd: str | Path,
    timeout_s: float,
    shell: ShellSpec | None = None,
    env: Mapping[str, str] | None = None,
) -> CommandResult:
    """Run ``command`` and capture stdout, stderr and exit code.

    Never raises on timeout: returns ``timed_out=True`` and exit code 124 after
    killing the full process tree.
    """
    spec = shell or detect_shell()
    child_env = dict(os.environ if env is None else env)
    if spec.kind in ("bash", "sh"):
        child_env.setdefault("CHERE_INVOKING", "1")  # MSYS login shells: stay in cwd
    popen_kwargs: dict = {}
    if not IS_WINDOWS:
        popen_kwargs["start_new_session"] = True  # own process group -> killable as a tree

    started = time.monotonic()
    proc = subprocess.Popen(
        spec.build_argv(command),
        cwd=str(cwd),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=child_env,
        **popen_kwargs,
    )
    timed_out = False
    try:
        out, err = proc.communicate(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        timed_out = True
        _kill_process_tree(proc)
        try:
            out, err = proc.communicate(timeout=10)
        except Exception:
            out, err = b"", b""
    finally:
        for stream in (proc.stdout, proc.stderr):
            try:
                if stream is not None:
                    stream.close()
            except Exception:
                pass

    exit_code = proc.returncode if not timed_out else TIMEOUT_EXIT_CODE
    if exit_code is None:
        exit_code = -1
    return CommandResult(
        exit_code=exit_code,
        stdout=_decode(out),
        stderr=_decode(err),
        timed_out=timed_out,
        shell=spec.kind,
        duration_ms=int((time.monotonic() - started) * 1000),
    )


def shell_description() -> str:
    """Short human description of the shell the Bash tool will use here."""
    try:
        spec = detect_shell()
    except RuntimeError as e:
        return f"unavailable ({e})"
    return f"{spec.kind} ({spec.executable})"


__all__ = [
    "CommandResult",
    "IS_WINDOWS",
    "ShellSpec",
    "TIMEOUT_EXIT_CODE",
    "detect_shell",
    "find_bash",
    "find_cmd",
    "find_powershell",
    "is_wsl_launcher",
    "run_shell_command",
    "shell_description",
]
