# reasonforge/codeverify/sandbox.py
"""Sandboxed Python code execution engine for ReasonForge code verification.

Executes untrusted Python code in isolated subprocesses with timeout
enforcement, stdout/stderr capture, and an import-blocking meta-path hook
that prevents access to dangerous standard-library modules.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import tempfile
import time
from dataclasses import dataclass
from typing import List, Optional, Set

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Blocked imports – modules that must not be reachable inside the sandbox.
# ---------------------------------------------------------------------------

BLOCKED_IMPORTS: List[str] = [
    "os",
    "subprocess",
    "shutil",
    "pathlib",
    "socket",
    "http",
    "urllib",
    "requests",
    "ctypes",
    "multiprocessing",
]

# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class SandboxResult:
    """Outcome of a sandboxed code execution."""

    stdout: str
    stderr: str
    exit_code: int
    timed_out: bool
    execution_time_ms: int
    success: bool


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_import_hook(blocked: Set[str]) -> str:
    """Return Python source for a ``sys.meta_path`` import blocker.

    The generated hook is prepended to every script executed in the sandbox
    so that any attempt to ``import <blocked_module>`` raises an
    ``ImportError`` immediately.
    """
    blocked_repr = repr(blocked)
    return (
        "import sys\n"
        "class _BlockedImportFinder:\n"
        f"    _blocked = {blocked_repr}\n"
        "    def find_module(self, name, path=None):\n"
        "        if name.split('.')[0] in self._blocked:\n"
        "            raise ImportError(f\"Import of '{name}' is blocked in sandbox\")\n"
        "        return None\n"
        "sys.meta_path.insert(0, _BlockedImportFinder())\n"
        "for _mod in list(sys.modules):\n"
        f"    if _mod.split('.')[0] in {blocked_repr}:\n"
        "        del sys.modules[_mod]\n"
    )


def _make_sandbox_result(
    stdout: str,
    stderr: str,
    exit_code: int,
    timed_out: bool,
    elapsed_ms: int,
) -> SandboxResult:
    """Construct a :class:`SandboxResult` with the derived ``success`` flag."""
    return SandboxResult(
        stdout=stdout,
        stderr=stderr,
        exit_code=exit_code,
        timed_out=timed_out,
        execution_time_ms=elapsed_ms,
        success=(exit_code == 0 and not timed_out),
    )


# ---------------------------------------------------------------------------
# Sandbox executor
# ---------------------------------------------------------------------------


class SandboxExecutor:
    """Execute Python code in a subprocess with resource restrictions.

    Parameters
    ----------
    timeout:
        Maximum wall-clock seconds allowed for a single execution.
    blocked_imports:
        Optional override for the set of blocked top-level module names.
        Defaults to :data:`BLOCKED_IMPORTS`.
    python_executable:
        Path to the Python interpreter used for subprocess execution.
        Defaults to the currently running interpreter (``sys.executable``).
    """

    def __init__(
        self,
        timeout: int = 30,
        blocked_imports: Optional[List[str]] = None,
        python_executable: Optional[str] = None,
    ) -> None:
        self.timeout = timeout
        self._blocked: Set[str] = set(
            blocked_imports if blocked_imports is not None else BLOCKED_IMPORTS
        )
        self._python: str = python_executable or sys.executable
        self._import_hook_source: str = _build_import_hook(self._blocked)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _prepend_hook(self, code: str) -> str:
        """Prepend the import-blocking hook to *code*."""
        return self._import_hook_source + "\n" + code

    def _write_temp_script(self, code: str, filename: str = "script.py") -> str:
        """Write *code* to a temporary ``.py`` file and return its path.

        A new temporary directory is created for each call so that files
        from different executions never collide.
        """
        tmp_dir = tempfile.mkdtemp(prefix="rfbox_")
        script_path = os.path.join(tmp_dir, filename)
        with open(script_path, "w", encoding="utf-8") as fh:
            fh.write(code)
        return script_path

    async def _run_subprocess(
        self,
        args: List[str],
        cwd: Optional[str] = None,
    ) -> SandboxResult:
        """Spawn *args* as an async subprocess and collect output.

        Enforces :attr:`timeout` and captures stdout, stderr, exit code.
        """
        start = time.monotonic()
        timed_out = False

        proc = await asyncio.create_subprocess_exec(
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd,
        )

        try:
            raw_stdout, raw_stderr = await asyncio.wait_for(
                proc.communicate(),
                timeout=self.timeout,
            )
        except asyncio.TimeoutError:
            timed_out = True
            proc.kill()
            # Drain whatever output was produced before the timeout.
            raw_stdout, raw_stderr = await proc.communicate()

        elapsed_ms = int((time.monotonic() - start) * 1000)

        stdout_text = raw_stdout.decode("utf-8", errors="replace") if raw_stdout else ""
        stderr_text = raw_stderr.decode("utf-8", errors="replace") if raw_stderr else ""
        exit_code: int = proc.returncode if proc.returncode is not None else -1

        logger.debug(
            "Subprocess finished: exit=%s timed_out=%s elapsed=%dms",
            exit_code,
            timed_out,
            elapsed_ms,
        )

        return _make_sandbox_result(
            stdout=stdout_text,
            stderr=stderr_text,
            exit_code=exit_code,
            timed_out=timed_out,
            elapsed_ms=elapsed_ms,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def execute(self, code: str) -> SandboxResult:
        """Execute *code* in an isolated subprocess.

        The import-blocking hook is prepended automatically.  The code is
        written to a temporary file and run with the configured Python
        interpreter.
        """
        sandboxed_code = self._prepend_hook(code)
        script_path = self._write_temp_script(sandboxed_code)
        try:
            return await self._run_subprocess([self._python, script_path])
        finally:
            _cleanup_path(script_path)

    async def execute_with_tests(
        self,
        code: str,
        test_code: str,
    ) -> SandboxResult:
        """Execute *code* followed by inline *test_code* assertions.

        The two fragments are concatenated (with two blank lines separating
        them) into a single script that is then run through :meth:`execute`.
        This is the simplest way to run ``assert``-based test suites against
        a piece of code.
        """
        combined = code + "\n\n" + test_code
        return await self.execute(combined)

    async def run_pytest(
        self,
        code: str,
        test_code: str,
        code_filename: str = "solution.py",
        test_filename: str = "test_solution.py",
    ) -> SandboxResult:
        """Run *pytest* on *code* and *test_code* inside a temporary directory.

        1. Create a temp directory.
        2. Write *code* (with import hook) to ``<code_filename>``.
        3. Write *test_code* (with import hook) to ``<test_filename>``.
        4. Invoke ``python -m pytest --tb=short -q`` in that directory.
        5. Return the captured :class:`SandboxResult`.
        """
        tmp_dir = tempfile.mkdtemp(prefix="rfbox_pytest_")
        code_path = os.path.join(tmp_dir, code_filename)
        test_path = os.path.join(tmp_dir, test_filename)

        sandboxed_code = self._prepend_hook(code)
        sandboxed_tests = self._prepend_hook(test_code)

        with open(code_path, "w", encoding="utf-8") as fh:
            fh.write(sandboxed_code)
        with open(test_path, "w", encoding="utf-8") as fh:
            fh.write(sandboxed_tests)

        try:
            return await self._run_subprocess(
                [self._python, "-m", "pytest", "--tb=short", "-q", test_filename],
                cwd=tmp_dir,
            )
        finally:
            _cleanup_path(code_path)
            _cleanup_path(test_path)
            _cleanup_dir(tmp_dir)


# ---------------------------------------------------------------------------
# File cleanup utilities
# ---------------------------------------------------------------------------


def _cleanup_path(path: str) -> None:
    """Best-effort removal of a single file."""
    try:
        os.remove(path)
    except OSError:
        logger.debug("Could not remove temp file: %s", path)


def _cleanup_dir(path: str) -> None:
    """Best-effort removal of an empty directory."""
    try:
        os.rmdir(path)
    except OSError:
        logger.debug("Could not remove temp dir: %s", path)
