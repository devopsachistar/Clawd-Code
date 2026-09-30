"""Cross-platform command execution (Bash / PowerShell tools)."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

from src.tool_system.context import ToolContext
from src.tool_system.shell import (
    IS_WINDOWS,
    TIMEOUT_EXIT_CODE,
    ShellSpec,
    detect_shell,
    find_bash,
    find_powershell,
    is_wsl_launcher,
    run_shell_command,
)
from src.tool_system.tools.bash import BashTool, _try_extract_cd
from src.tool_system.tools.misc import PowerShellTool

PYTHON = f'"{sys.executable}"'


class ShellDetectionTests(unittest.TestCase):
    def test_detect_shell_returns_usable_shell(self):
        spec = detect_shell()
        self.assertIn(spec.kind, {"bash", "sh", "powershell", "cmd"})
        self.assertTrue(Path(spec.executable).exists() or spec.executable)

    def test_wsl_launcher_detection(self):
        self.assertTrue(is_wsl_launcher(r"C:\Windows\System32\bash.exe"))
        self.assertTrue(is_wsl_launcher(r"C:\Users\me\AppData\Local\Microsoft\WindowsApps\bash.exe"))
        self.assertFalse(is_wsl_launcher(r"C:\Program Files\Git\bin\bash.exe"))

    @unittest.skipUnless(IS_WINDOWS, "Windows-specific: bash must never resolve to the WSL launcher")
    def test_windows_bash_is_never_wsl(self):
        bash = find_bash()
        if bash is not None:
            self.assertFalse(is_wsl_launcher(bash))
        spec = detect_shell()
        self.assertFalse(is_wsl_launcher(spec.executable))

    def test_argv_construction(self):
        self.assertEqual(ShellSpec("bash", "bash").build_argv("ls"), ["bash", "-lc", "ls"])
        self.assertEqual(ShellSpec("cmd", "cmd.exe").build_argv("dir"), ["cmd.exe", "/d", "/s", "/c", "dir"])
        ps = ShellSpec("powershell", "pwsh").build_argv("Get-Location")
        self.assertEqual(ps[:4], ["pwsh", "-NoLogo", "-NoProfile", "-NonInteractive"])
        self.assertTrue(ps[-1].endswith("Get-Location"))


class RunShellCommandTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name).resolve()

    def tearDown(self) -> None:
        self.tmp.cleanup()  # must not fail: no process may keep the directory open

    def test_captures_stdout_stderr_and_exit_code(self):
        result = run_shell_command(
            f"{PYTHON} -c \"import sys; print('out'); print('err', file=sys.stderr); sys.exit(3)\"",
            cwd=self.root, timeout_s=60,
        )
        self.assertEqual(result.exit_code, 3)
        self.assertIn("out", result.stdout)
        self.assertIn("err", result.stderr)
        self.assertFalse(result.timed_out)

    def test_runs_in_requested_directory(self):
        result = run_shell_command(f"{PYTHON} -c \"import os; print(os.getcwd())\"", cwd=self.root, timeout_s=60)
        self.assertEqual(Path(result.stdout.strip()).resolve(), self.root)

    def test_unicode_output(self):
        result = run_shell_command(
            f"{PYTHON} -c \"import sys; sys.stdout.buffer.write('h\\u00e9llo \\u4f60\\u597d'.encode('utf-8'))\"",
            cwd=self.root, timeout_s=60,
        )
        self.assertIn("héllo 你好", result.stdout)

    def test_timeout_kills_process_and_reports(self):
        result = run_shell_command(f"{PYTHON} -c \"import time; time.sleep(30)\"", cwd=self.root, timeout_s=2)
        self.assertTrue(result.timed_out)
        self.assertEqual(result.exit_code, TIMEOUT_EXIT_CODE)
        self.assertLess(result.duration_ms, 25_000)

    def test_stdin_is_closed(self):
        result = run_shell_command(f"{PYTHON} -c \"import sys; print(repr(sys.stdin.read()))\"", cwd=self.root, timeout_s=30)
        self.assertEqual(result.stdout.strip(), "''")

    @unittest.skipUnless(IS_WINDOWS, "cmd.exe exists only on Windows")
    def test_cmd_shell_on_windows(self):
        result = run_shell_command("echo %CD%", cwd=self.root, timeout_s=30, shell=detect_shell("cmd"))
        self.assertEqual(result.exit_code, 0)
        self.assertEqual(Path(result.stdout.strip()).resolve(), self.root)

    @unittest.skipUnless(find_powershell(), "PowerShell not installed")
    def test_powershell_shell(self):
        result = run_shell_command(
            "Write-Output 'ps-ok'; Write-Error 'ps-err'; exit 4",
            cwd=self.root, timeout_s=60, shell=detect_shell("powershell"),
        )
        self.assertEqual(result.exit_code, 4)
        self.assertIn("ps-ok", result.stdout)
        self.assertIn("ps-err", result.stderr)


class BashToolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name).resolve()
        (self.root / "sub").mkdir()
        self.ctx = ToolContext(workspace_root=self.root)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_echo_reports_shell_and_exit_code(self):
        out = BashTool().run({"command": "echo hello"}, self.ctx).output
        self.assertEqual(out["exit_code"], 0)
        self.assertIn("hello", out["stdout"])
        self.assertIn(out["shell"], {"bash", "sh", "powershell", "cmd"})

    def test_native_absolute_cwd_is_accepted(self):
        out = BashTool().run(
            {"command": f"{PYTHON} -c \"import os; print(os.getcwd())\"", "cwd": str(self.root / "sub")}, self.ctx
        ).output
        self.assertEqual(Path(out["stdout"].strip()).resolve(), self.root / "sub")

    def test_relative_cwd_rejected(self):
        with self.assertRaises(Exception):
            BashTool().run({"command": "echo x", "cwd": "sub"}, self.ctx)

    def test_cwd_outside_workspace_rejected(self):
        with self.assertRaises(Exception):
            BashTool().run({"command": "echo x", "cwd": str(Path(self.tmp.name).resolve().parent)}, self.ctx)

    def test_timeout_is_an_error_result_not_an_exception(self):
        result = BashTool().run({"command": f"{PYTHON} -c \"import time; time.sleep(20)\"", "timeout_s": 1}, self.ctx)
        self.assertTrue(result.is_error)
        self.assertTrue(result.output["timed_out"])

    def test_bare_cd_changes_session_directory(self):
        out = BashTool().run({"command": "cd sub"}, self.ctx).output
        self.assertEqual(Path(out["cwd"]), self.root / "sub")
        self.assertEqual(self.ctx.cwd, self.root / "sub")

    def test_cd_with_following_command_runs_everything(self):
        # Previously "cd sub && <cmd>" silently dropped "<cmd>".
        out = BashTool().run(
            {"command": f"cd sub && {PYTHON} -c \"open('made.txt','w').write('x')\""}, self.ctx
        ).output
        self.assertEqual(out["exit_code"], 0)
        self.assertTrue((self.root / "sub" / "made.txt").exists())
        self.assertEqual(self.ctx.cwd, self.root)  # a compound command does not move the session

    def test_extract_cd_only_for_bare_cd(self):
        self.assertEqual(_try_extract_cd("cd sub"), Path("sub"))
        self.assertIsNone(_try_extract_cd("cd sub && rm -rf x"))
        self.assertIsNone(_try_extract_cd("cd sub; ls"))
        self.assertIsNone(_try_extract_cd("cdx"))

    def test_blocked_command_raises_even_when_called_directly(self):
        with self.assertRaises(Exception):
            BashTool().run({"command": "sudo echo nope"}, self.ctx)


@unittest.skipUnless(find_powershell(), "PowerShell not installed")
class PowerShellToolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name).resolve()
        self.ctx = ToolContext(workspace_root=self.root)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_runs_powershell(self):
        result = PowerShellTool().run({"command": "(Get-Location).Path"}, self.ctx)
        self.assertFalse(result.is_error)
        self.assertEqual(result.output["shell"], "powershell")
        self.assertEqual(Path(result.output["stdout"].strip()).resolve(), self.root)

    def test_blocked_powershell_command(self):
        with self.assertRaises(Exception):
            PowerShellTool().run({"command": "Stop-Computer -Force"}, self.ctx)


if __name__ == "__main__":
    unittest.main()
