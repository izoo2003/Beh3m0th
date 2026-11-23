"""Docker sandbox and the restricted subprocess fallback. Never uses exec()."""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Protocol

from beh3m0th.config import Settings

logger = logging.getLogger(__name__)

_SECRET_SUFFIXES = ("_API_KEY", "_TOKEN", "_SECRET", "_PASSWORD")


class SandboxResult:
    """Process outcome from either sandbox backend."""

    def __init__(self, *, exit_code: int, stdout: str, stderr: str, sandbox_mode: str) -> None:
        self.exit_code = exit_code
        self.stdout = stdout
        self.stderr = stderr
        self.sandbox_mode = sandbox_mode


class SandboxRunner(Protocol):
    async def run_python(
        self,
        code: str,
        *,
        workspace: Path,
        timeout_sec: float,
    ) -> SandboxResult: ...

    async def run_pytest(
        self,
        target: Path,
        *,
        workspace: Path,
        timeout_sec: float,
    ) -> SandboxResult: ...


def build_docker_argv(
    *,
    image: str,
    workspace: Path,
    command: list[str],
    memory: str,
    cpus: str,
    pids_limit: int,
    container_name: str,
    docker_bin: str = "docker",
) -> list[str]:
    """Argv for an ephemeral container. Unit tests inspect this without starting Docker."""
    root = str(workspace.resolve())
    return [
        docker_bin,
        "run",
        "--rm",
        "--network=none",
        "--read-only",
        "--tmpfs",
        "/tmp:rw,noexec,nosuid,size=64m",
        f"--memory={memory}",
        f"--cpus={cpus}",
        f"--pids-limit={pids_limit}",
        "--name",
        container_name,
        "--workdir",
        "/workspace",
        "-v",
        f"{root}:/workspace:rw",
        image,
        *command,
    ]


class DockerSandbox:
    """Primary backend. ``docker kill`` runs in ``finally`` even after a timeout."""

    def __init__(
        self,
        *,
        image: str = "beh3m0th-sandbox:latest",
        memory: str = "256m",
        cpus: str = "0.5",
        pids_limit: int = 64,
        docker_bin: str = "docker",
    ) -> None:
        self.image = image
        self.memory = memory
        self.cpus = cpus
        self.pids_limit = pids_limit
        self.docker_bin = docker_bin

    async def run_python(
        self,
        code: str,
        *,
        workspace: Path,
        timeout_sec: float,
    ) -> SandboxResult:
        return await self._run(
            ["python", "-c", code],
            workspace=workspace,
            timeout_sec=timeout_sec,
        )

    async def run_pytest(
        self,
        target: Path,
        *,
        workspace: Path,
        timeout_sec: float,
    ) -> SandboxResult:
        relative = target.resolve().relative_to(workspace.resolve()).as_posix() or "."
        return await self._run(
            ["python", "-m", "pytest", relative, "-q", "--noconftest"],
            workspace=workspace,
            timeout_sec=timeout_sec,
        )

    async def _run(
        self,
        command: list[str],
        *,
        workspace: Path,
        timeout_sec: float,
    ) -> SandboxResult:
        name = f"beh3m0th-sbx-{uuid.uuid4().hex[:12]}"
        argv = build_docker_argv(
            image=self.image,
            workspace=workspace,
            command=command,
            memory=self.memory,
            cpus=self.cpus,
            pids_limit=self.pids_limit,
            container_name=name,
            docker_bin=self.docker_bin,
        )
        process = await asyncio.create_subprocess_exec(
            *argv,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout_sec)
        except TimeoutError:
            process.kill()
            await process.wait()
            return SandboxResult(
                exit_code=124,
                stdout="",
                stderr="timeout",
                sandbox_mode="docker",
            )
        finally:
            await _docker_kill(self.docker_bin, name)
        return SandboxResult(
            exit_code=process.returncode if process.returncode is not None else 1,
            stdout=stdout.decode("utf-8", errors="replace"),
            stderr=stderr.decode("utf-8", errors="replace"),
            sandbox_mode="docker",
        )


class SubprocessRestrictedSandbox:
    """Fallback when Docker is unavailable. Isolated temp cwd and a stripped environment."""

    async def run_python(
        self,
        code: str,
        *,
        workspace: Path,
        timeout_sec: float,
    ) -> SandboxResult:
        del workspace
        logger.info("SANDBOX_MODE=fallback")
        temp = Path(tempfile.mkdtemp(prefix="beh3m0th-sbx-"))
        try:
            script = temp / "main.py"
            await asyncio.to_thread(script.write_text, code, "utf-8")
            return await _run_subprocess(
                [sys.executable, "main.py"],
                cwd=temp,
                timeout_sec=timeout_sec,
            )
        finally:
            await asyncio.to_thread(_remove_tree, temp)

    async def run_pytest(
        self,
        target: Path,
        *,
        workspace: Path,
        timeout_sec: float,
    ) -> SandboxResult:
        logger.info("SANDBOX_MODE=fallback")
        temp = Path(tempfile.mkdtemp(prefix="beh3m0th-pytest-"))
        try:
            ini = temp / "pytest.ini"
            await asyncio.to_thread(ini.write_text, "[pytest]\n", "utf-8")
            argv = [
                sys.executable,
                "-m",
                "pytest",
                str(target),
                "-q",
                "--noconftest",
                "-p",
                "no:cacheprovider",
                "-c",
                str(ini),
                f"--rootdir={workspace}",
            ]
            return await _run_subprocess(argv, cwd=workspace, timeout_sec=timeout_sec)
        finally:
            await asyncio.to_thread(_remove_tree, temp)


def sandbox_from_settings(settings: Settings) -> SandboxRunner:
    """Pick Docker or the restricted subprocess from ``Settings``."""
    if settings.mcp_sandbox_backend == "docker":
        return DockerSandbox(
            image=settings.mcp_sandbox_image,
            memory=settings.mcp_sandbox_mem_limit,
        )
    return SubprocessRestrictedSandbox()


def stripped_env() -> dict[str, str]:
    """Copy the process environment without API keys, tokens, secrets, or pytest flags."""
    cleaned: dict[str, str] = {}
    for key, value in os.environ.items():
        upper = key.upper()
        if upper.startswith("PYTEST_"):
            continue
        if any(upper.endswith(suffix) for suffix in _SECRET_SUFFIXES):
            continue
        cleaned[key] = value
    return cleaned


async def _run_subprocess(argv: list[str], *, cwd: Path, timeout_sec: float) -> SandboxResult:
    process = await asyncio.create_subprocess_exec(
        *argv,
        cwd=str(cwd),
        env=stripped_env(),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout_sec)
    except TimeoutError:
        process.kill()
        await process.wait()
        return SandboxResult(
            exit_code=124,
            stdout="",
            stderr="timeout",
            sandbox_mode="fallback",
        )
    return SandboxResult(
        exit_code=process.returncode if process.returncode is not None else 1,
        stdout=stdout.decode("utf-8", errors="replace"),
        stderr=stderr.decode("utf-8", errors="replace"),
        sandbox_mode="fallback",
    )


async def _docker_kill(docker_bin: str, name: str) -> None:
    try:
        process = await asyncio.create_subprocess_exec(
            docker_bin,
            "kill",
            name,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
    except OSError:
        logger.info("SANDBOX_MODE=docker kill skipped")
        return
    try:
        await asyncio.wait_for(process.wait(), 5)
    except TimeoutError:
        process.kill()
        await process.wait()


def _remove_tree(path: Path) -> None:
    shutil.rmtree(path, ignore_errors=True)
