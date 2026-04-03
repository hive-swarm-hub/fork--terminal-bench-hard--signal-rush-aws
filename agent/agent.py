"""
Meta-Harness — built on Terminus-KIRA (KRAFTON AI), which extends Harbor's Terminus2.

Adds environment bootstrapping: gathers a sandbox snapshot (working directory, file
listing, available languages/tools, package managers) before the agent loop starts
and injects it into the initial prompt.
"""

import asyncio
import json
import shlex
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

import litellm
from anthropic_caching import add_anthropic_caching
from harbor.agents.terminus_2 import Terminus2
from harbor.agents.terminus_2.terminus_2 import Command
from harbor.agents.terminus_2.tmux_session import TmuxSession
from harbor.environments.base import BaseEnvironment
from harbor.llms.base import (
    ContextLengthExceededError,
    LLMResponse,
    OutputLengthExceededError,
)
from harbor.llms.chat import Chat
from harbor.models.agent.context import AgentContext
from harbor.models.metric import UsageInfo
from harbor.models.trajectories import (
    Metrics,
    Observation,
    ObservationResult,
    Step,
    ToolCall,
)
from litellm.exceptions import (
    AuthenticationError as LiteLLMAuthenticationError,
    BadRequestError,
    ContextWindowExceededError as LiteLLMContextWindowExceededError,
)
from tenacity import (
    retry,
    retry_if_exception_type,
    retry_if_not_exception_type,
    stop_after_attempt,
    wait_exponential,
)


class BlockError(Exception):
    """Raised when infrastructure API call blocks for too long."""

    pass


BLOCK_TIMEOUT_SEC = 600  # 10 minutes
_MARKER_PREFIX = "__CMDEND__"  # Marker prefix for command completion detection



@dataclass
class ToolCallResponse:
    """Extended response that includes tool calls."""

    content: str | None
    tool_calls: list[dict[str, Any]]
    reasoning_content: str | None = None
    usage: UsageInfo | None = None


@dataclass
class ImageReadRequest:
    """Request to read and analyze an image file."""

    file_path: str
    image_read_instruction: str


# Tool description strings
_EXECUTE_COMMANDS_DESC = (
    "Call this to execute commands in the terminal with your analysis and plan."
)

_ANALYSIS_DESC = (
    "Analyze the current state based on the terminal output provided. "
    "What do you see? What has been accomplished? What still needs to be done?"
)

_PLAN_DESC = (
    "Describe your plan for the next steps. "
    "What commands will you run and why? "
    "Be specific about what you expect each command to accomplish."
)

_COMMANDS_DESC = (
    "The commands array can be empty if you want to wait without taking action."
)

_KEYSTROKES_DESC = (
    "String containing the exact keystrokes to send to the terminal. "
    "The text will be used completely verbatim as keystrokes. "
    "Write commands exactly as you want them sent to the terminal. "
    "Most bash commands should end with a newline (\\n) to cause them to execute. "
    "For special key sequences, use tmux-style escape sequences: C-c for Ctrl+C, C-d for Ctrl+D. "
    "Each command's keystrokes are sent exactly as written to the terminal. "
    "Do not include extra whitespace before or after the keystrokes unless it's part of the intended command."
)

_DURATION_DESC = (
    "Number of seconds to wait for the command to complete (default: 1.0) "
    "before the next command will be executed. "
    "On immediate tasks (e.g., cd, ls, echo, cat) set a duration of 0.1 seconds. "
    "On commands (e.g., gcc, find, rustc) set a duration of 1.0 seconds. "
    "On slow commands (e.g., make, python3 [long running script], wget [file]) set an appropriate duration as you determine necessary. "
    "It is better to set a smaller duration than a longer duration. "
    "It is always possible to wait again if the prior output has not finished, "
    "by running empty keystrokes with a duration on subsequent requests to wait longer. "
    "Never wait longer than 60 seconds; prefer to poll to see intermediate result status."
)

_TASK_COMPLETE_DESC = "Call this when the task is complete."

_RESET_TERMINAL_DESC = (
    "Emergency recovery: kills ALL running processes and resets the terminal. "
    "Use this ONLY when the terminal is completely stuck and unresponsive — "
    "e.g., a process ignores Ctrl+C, a command hangs indefinitely, or you "
    "cannot type new commands. After calling this, you will get a fresh bash "
    "shell in the same working directory. Any background processes will be killed."
)

_IMAGE_READ_DESC = (
    "Read and analyze an image file. "
    "Use this ONLY for image files that you need to visually analyze. "
    "Do NOT use this for text files — use shell commands (cat, head, etc.) instead. "
    "The image will be sent to the model for visual analysis "
    "and you will receive a text description in the next turn."
)

_FILE_PATH_DESC = (
    "Absolute path to the image file. Supported formats: PNG, JPG, JPEG, GIF, WEBP."
)

_IMAGE_READ_INSTRUCTION_DESC = (
    "A text instruction describing what you want to learn from the image. "
    "Be specific about what information to extract."
)

# Tool definitions for native tool use
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "execute_commands",
            "description": _EXECUTE_COMMANDS_DESC,
            "parameters": {
                "type": "object",
                "properties": {
                    "analysis": {
                        "type": "string",
                        "description": _ANALYSIS_DESC,
                    },
                    "plan": {
                        "type": "string",
                        "description": _PLAN_DESC,
                    },
                    "commands": {
                        "type": "array",
                        "description": _COMMANDS_DESC,
                        "items": {
                            "type": "object",
                            "properties": {
                                "keystrokes": {
                                    "type": "string",
                                    "description": _KEYSTROKES_DESC,
                                },
                                "duration": {
                                    "type": "number",
                                    "description": _DURATION_DESC,
                                },
                            },
                            "required": ["keystrokes"],
                        },
                    },
                    "parallel": {
                        "type": "boolean",
                        "description": (
                            "Set to true to run commands in parallel across separate terminal windows. "
                            "Use this when commands are independent and don't depend on each other's output "
                            "(e.g., installing packages while writing code, multiple independent file reads, "
                            "running tests while editing other files). "
                            "Default is false (sequential execution)."
                        ),
                    },
                },
                "required": ["analysis", "plan", "commands"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "task_complete",
            "description": _TASK_COMPLETE_DESC,
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "reset_terminal",
            "description": _RESET_TERMINAL_DESC,
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "image_read",
            "description": _IMAGE_READ_DESC,
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": _FILE_PATH_DESC,
                    },
                    "image_read_instruction": {
                        "type": "string",
                        "description": _IMAGE_READ_INSTRUCTION_DESC,
                    },
                },
                "required": ["file_path", "image_read_instruction"],
            },
        },
    },
]


class TmuxWindowPool:
    """Pre-allocated pool of tmux windows for parallel execution and stall recovery."""

    def __init__(self, environment, session_name: str, size: int = 4):
        self._env = environment
        self._sess = session_name
        self._ready: list[str] = []
        self._seq = 0
        self._size = size

    async def start(self):
        """Pre-create the pool windows."""
        for _ in range(self._size):
            name = await self._create()
            self._ready.append(name)

    async def _create(self) -> str:
        self._seq += 1
        name = f"pool{self._seq}"
        await self._env.exec(
            command=f"tmux new-window -t {self._sess} -n {name} -d"
        )
        # Set PAGER=cat in the new window
        await self._env.exec(
            command=f"tmux send-keys -t {self._sess}:{name} 'export PAGER=cat GIT_PAGER=cat MANPAGER=cat' Enter"
        )
        return name

    async def acquire(self) -> str:
        """Get a ready window, creating one if pool is empty."""
        if self._ready:
            return self._ready.pop(0)
        return await self._create()

    async def release(self, name: str):
        """Return a window to the pool after resetting it."""
        target = f"{self._sess}:{name}"
        await self._env.exec(command=f"tmux send-keys -t {target} C-c")
        await self._env.exec(command=f"tmux send-keys -t {target} ' reset' Enter")
        self._ready.append(name)


class TmuxSessionPool:
    """Pool of TmuxSession objects for parallel command execution.

    Each pool session is a separate tmux session with its own pane.
    asyncio.gather enables true parallel send/capture across sessions.
    """

    def __init__(self, environment, size: int = 4):
        self._env = environment
        self._size = size
        self._sessions: list[TmuxSession] = []
        self._ready: list[TmuxSession] = []
        self._seq = 0

    async def start(self):
        """Pre-create pool sessions."""
        # Create sessions concurrently for faster startup
        coros = [self._create() for _ in range(self._size)]
        sessions = await asyncio.gather(*coros, return_exceptions=True)
        for s in sessions:
            if isinstance(s, TmuxSession):
                self._sessions.append(s)
                self._ready.append(s)

    async def _create(self) -> TmuxSession:
        self._seq += 1
        name = f"pool{self._seq}"
        s = TmuxSession(
            session_name=name,
            environment=self._env,
            logging_path=PurePosixPath(f"/tmp/{name}.pane"),
            local_asciinema_recording_path=None,
            remote_asciinema_recording_path=None,
            pane_width=160,
            pane_height=40,
        )
        await s.start()
        # Set PAGER=cat and cd /app
        await s.send_keys(
            "export PAGER=cat GIT_PAGER=cat MANPAGER=cat && cd /app\n",
            block=False, min_timeout_sec=0.3,
        )
        await asyncio.sleep(0.3)
        await s.get_incremental_output()  # drain
        return s

    async def acquire(self) -> TmuxSession | None:
        if self._ready:
            return self._ready.pop(0)
        # Pool exhausted — create on demand
        try:
            s = await self._create()
            self._sessions.append(s)
            return s
        except Exception:
            return None

    async def release(self, session: TmuxSession):
        """Return session to pool. Drain output for clean state."""
        try:
            await session.get_incremental_output()
        except Exception:
            pass
        self._ready.append(session)


class AgentHarness(Terminus2):
    """
    TerminusKira extends harbor's Terminus2 with native tool calling.

    Instead of prompting the model to output JSON/XML and parsing it, TerminusKira uses the `tools` parameter in LLM API calls for structured outputs.
    """

    _PLANNING_EPISODES = 0        # only episode 0 uses high reasoning
    _PLANNING_EFFORT = "high"     # deep thinking for understanding + planning (NOT "max" — too slow)
    _EXECUTION_EFFORT = None       # use API default (don't override)
    _VERIFICATION_EFFORT = "high"  # careful check before completing

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._marker_seq = 0
        self._total_time_saved = 0.0
        self._window_pool: TmuxWindowPool | None = None
        self._session_pool: TmuxSessionPool | None = None
        self._current_episode = 0
        self._consecutive_stalls = 0
        self._last_pane_snapshot: str | None = None  # For smart stall detection

    async def _reset_terminal(self, session: TmuxSession) -> str:
        """Kill all processes and respawn a fresh bash shell.

        Uses environment.exec() which bypasses the stuck tmux pane entirely,
        so this works even when the terminal is completely unresponsive.
        Each step has a 15s timeout to avoid blocking on infra failures.
        """
        env = session.environment
        session_name = session._session_name

        # Step 1: Kill all user processes via environment.exec (bypasses tmux)
        try:
            await asyncio.wait_for(
                env.exec(command="pkill -9 -u $(whoami) || true", user="root"),
                timeout=15,
            )
        except Exception:
            pass
        await asyncio.sleep(0.5)

        # Step 2: Kill the old tmux session
        try:
            await asyncio.wait_for(
                env.exec(
                    command=f"tmux kill-session -t {session_name} 2>/dev/null || true",
                    user=session._user,
                ),
                timeout=15,
            )
        except Exception:
            pass
        await asyncio.sleep(0.5)

        # Step 3: Start a fresh tmux session with the same name
        try:
            start_cmd = (
                f"export TERM=xterm-256color && export SHELL=/bin/bash && "
                f'script -qc "'
                f"tmux new-session -x {session._pane_width} -y {session._pane_height} "
                f"-d -s {session_name} 'bash --login'"
                f'" /dev/null'
            )
            result = await asyncio.wait_for(
                env.exec(command=start_cmd, user=session._user),
                timeout=15,
            )
            if result.return_code != 0:
                self.logger.error(f"Failed to restart tmux session: {result.stderr}")
        except Exception as e:
            self.logger.error(f"Exception restarting tmux: {e}")
        await asyncio.sleep(0.5)

        # Step 4: Re-disable pagers in the fresh shell
        try:
            await session.send_keys(
                "export PAGER=cat GIT_PAGER=cat MANPAGER=cat LESS='-F -X'\n",
                block=False,
                min_timeout_sec=0.3,
            )
        except Exception:
            pass

        # Step 5: cd back to /app
        try:
            await session.send_keys("cd /app\n", block=False, min_timeout_sec=0.3)
        except Exception:
            pass

        await asyncio.sleep(1.0)

        self._consecutive_stalls = 0
        self._last_pane_snapshot = None
        session._previous_buffer = None

        try:
            output = await session.get_incremental_output()
        except Exception:
            output = "[Terminal reset complete. Fresh bash shell ready.]"

        self.logger.info("Terminal reset completed successfully")

        return f"[TERMINAL RESET] All processes killed. Fresh bash shell ready in /app.\n\n{output}"

    @staticmethod
    def _sanitize_command(keystrokes: str) -> str:
        """Rewrite known-dangerous command patterns at infrastructure level.

        This prevents terminal stalls without relying on the model following
        prompt instructions (which V4 and V9 proved is unreliable).
        """
        import re
        stripped = keystrokes.strip()
        # tail -f → tail -100 (tail -f blocks terminal permanently)
        if re.match(r'^tail\s+(-[nN]\s*\d+\s+)?-f\b', stripped):
            keystrokes = re.sub(r'-f\b', '-100', keystrokes, count=1)
        elif re.match(r'^tail\s+--follow\b', stripped):
            keystrokes = keystrokes.replace('--follow', '-100', 1)
        # curl without timeout → inject --connect-timeout 30 --max-time 300
        elif re.match(r'^curl\s', stripped) and '--connect-timeout' not in stripped and '--max-time' not in stripped:
            keystrokes = keystrokes.replace('curl ', 'curl --connect-timeout 30 --max-time 300 ', 1)
        # wget without timeout → inject --timeout=30
        elif re.match(r'^wget\s', stripped) and '--timeout' not in stripped:
            keystrokes = keystrokes.replace('wget ', 'wget --timeout=30 ', 1)
        return keystrokes

    async def _with_block_timeout(self, coro, timeout_sec: int = BLOCK_TIMEOUT_SEC):
        """Wrap coroutine with block detection timeout."""
        try:
            return await asyncio.wait_for(coro, timeout=timeout_sec)
        except asyncio.TimeoutError:
            raise BlockError(f"Infrastructure API blocked for {timeout_sec}s")

    async def _execute_commands(
        self,
        commands: list[Command],
        session: TmuxSession,
    ) -> tuple[bool, str]:
        """Execute commands — auto-parallelizes batches of 2+ commands across
        separate tmux windows when a pool is available.
        """
        if not commands:
            output = await session.get_incremental_output()
            return False, self._limit_output_length(output)

        # Sanitize commands at infrastructure level
        for cmd in commands:
            cmd.keystrokes = self._sanitize_command(cmd.keystrokes)

        max_dur = max(c.duration_sec for c in commands)

        # Auto-parallel: multiple commands with at least one slow command
        if len(commands) >= 2 and self._session_pool is not None:
            return await self._execute_commands_parallel(commands, session)

        total_duration = sum(c.duration_sec for c in commands)

        # ---- Fast path: all commands are quick, skip marker overhead ----
        if max_dur <= 0.5:
            for command in commands:
                await session.send_keys(
                    command.keystrokes, block=False, min_timeout_sec=0.0,
                )
            await asyncio.sleep(max(total_duration, 0.5))
            output = await session.get_incremental_output()
            return False, self._limit_output_length(output)

        # ---- Slow path: pipelined markers ----
        # Phase 1: fire all keystrokes + markers without waiting
        batch_markers = []
        for command in commands:
            self._marker_seq += 1
            marker = f"{_MARKER_PREFIX}{self._marker_seq}__"
            batch_markers.append(marker)

            await session.send_keys(
                command.keystrokes, block=False, min_timeout_sec=0.0,
            )
            await session.send_keys(
                f"echo '{marker}'\n", block=False, min_timeout_sec=0.0,
            )

        last_marker = batch_markers[-1]
        hard_timeout = min(max(total_duration, 10.0), 120.0)
        start = time.monotonic()

        # Capture pane before polling to detect if content changes (alive vs stuck)
        pre_pane = await session.capture_pane(capture_entire=True)

        # Phase 2: poll for the last marker
        await asyncio.sleep(min(0.3, total_duration))
        found_last = False
        while time.monotonic() - start < hard_timeout:
            pane_content = await session.capture_pane(capture_entire=True)
            if last_marker in pane_content:
                found_last = True
                break
            await asyncio.sleep(0.5)

        elapsed = time.monotonic() - start
        saved = total_duration - elapsed
        if saved > 0.1:
            self._total_time_saved += saved
            self.logger.debug(
                f"[hybrid] saved {saved:.1f}s "
                f"(total_duration={total_duration:.1f}s, "
                f"actual={elapsed:.1f}s, "
                f"cmds={len(commands)})"
            )

        if not found_last:
            pane_content = await session.capture_pane(capture_entire=True)
            completed = sum(1 for m in batch_markers if m in pane_content)
            self.logger.warning(
                f"[stall] hard timeout {hard_timeout:.0f}s hit, "
                f"{completed}/{len(commands)} commands completed"
            )

        # Phase 3: filter markers from output
        output = await session.get_incremental_output()
        all_markers = {
            f"{_MARKER_PREFIX}{seq}__"
            for seq in range(1, self._marker_seq + 1)
        }
        lines = output.split("\n")
        lines = [line for line in lines if not any(m in line for m in all_markers)]
        output = "\n".join(lines)

        # Smart stall detection: distinguish "truly stuck" from "long-running process"
        if not found_last:
            stall_cmds = [c.keystrokes.strip()[:80] for c in commands[completed:]]
            # Check if pane content changed — if so, the process is alive, just slow
            post_pane = pane_content if pane_content else await session.capture_pane(capture_entire=True)
            pane_changed = (pre_pane != post_pane)
            content_changed = bool(output.strip())

            if pane_changed or content_changed:
                # Process is alive but slow — DON'T escalate stall counter
                # This prevents killing legitimate long-running tasks (training, VM, builds)
                self._consecutive_stalls = max(0, self._consecutive_stalls - 1)
                output += (
                    f"\n\n[INFO: {len(commands) - completed} command(s) still running "
                    f"after {hard_timeout:.0f}s (process is producing output). "
                    f"Commands: {'; '.join(stall_cmds)}. "
                    f"The process appears active — check progress with ps or tail.]"
                )
            else:
                # Pane is truly frozen — escalate
                self._consecutive_stalls += 1
                if self._consecutive_stalls >= 5:
                    output += (
                        f"\n\n[CRITICAL: Terminal has been frozen for {self._consecutive_stalls} "
                        f"consecutive commands with NO output. Stalled on: {'; '.join(stall_cmds)}. "
                        f"Call reset_terminal to kill all processes and get a fresh shell.]"
                    )
                elif self._consecutive_stalls >= 3:
                    output += (
                        f"\n\n[WARNING: Terminal may be stuck ({self._consecutive_stalls} "
                        f"consecutive timeouts with no output change). Stalled on: {'; '.join(stall_cmds)}. "
                        f"Try: kill the process with kill -9, use Ctrl+C, "
                        f"or call reset_terminal if nothing works.]"
                    )
                else:
                    output += (
                        f"\n\n[WARNING: {len(commands) - completed} command(s) may not have "
                        f"completed within {hard_timeout:.0f}s. Possibly stalled commands: "
                        f"{'; '.join(stall_cmds)}. "
                        f"If a process is stuck, try: kill the process or use Ctrl+C.]"
                    )
        else:
            self._consecutive_stalls = 0

        return False, self._limit_output_length(output)

    async def _execute_commands_parallel(
        self,
        commands: list[Command],
        session: TmuxSession,
    ) -> tuple[bool, str]:
        """Execute commands in parallel across separate TmuxSession objects.

        Uses asyncio.gather for true concurrent send/capture.
        Benchmark shows parallel ops take same time as single op (~320ms).
        """
        pool = self._session_pool

        # Sanitize commands at infrastructure level
        for cmd in commands:
            cmd.keystrokes = self._sanitize_command(cmd.keystrokes)

        # Filter empty commands
        real_cmds = [(i, cmd) for i, cmd in enumerate(commands) if cmd.keystrokes.strip()]
        if not real_cmds:
            output = await session.get_incremental_output()
            return False, self._limit_output_length(output)

        # Acquire sessions for each command
        assignments = []  # (index, cmd, pool_session, marker)
        for i, cmd in real_cmds:
            ps = await pool.acquire()
            if ps is None:
                # Pool exhausted, fall back to main session for remaining
                break
            self._marker_seq += 1
            marker = f"{_MARKER_PREFIX}{self._marker_seq}__"
            assignments.append((i, cmd, ps, marker))

        # Phase 1: Send all commands + markers in parallel via asyncio.gather
        async def send_one(idx, cmd, ps, marker):
            try:
                await ps.send_keys(cmd.keystrokes, block=False, min_timeout_sec=0.0)
                await ps.send_keys(f"echo '{marker}'\n", block=False, min_timeout_sec=0.0)
                return True
            except Exception as e:
                self.logger.warning(f"Pool send failed: {e}")
                return False

        send_results = await asyncio.gather(*[send_one(*a) for a in assignments])
        # Filter out failed sends — release their sessions
        ok_assignments = []
        for (idx, cmd, ps, marker), ok in zip(assignments, send_results):
            if ok:
                ok_assignments.append((idx, cmd, ps, marker))
            else:
                await pool.release(ps)
        assignments = ok_assignments

        if not assignments:
            # All sends failed — fall back to sequential
            return await self._execute_commands_sequential(commands, session)

        # Phase 2: Poll all sessions in parallel
        async def poll_one(idx, cmd, ps, marker):
            try:
                timeout = min(cmd.duration_sec + 5.0, 65.0)
                start = time.monotonic()
                await asyncio.sleep(min(0.3, cmd.duration_sec))

                while time.monotonic() - start < timeout:
                    pane = await ps.capture_pane(capture_entire=True)
                    if marker in pane:
                        elapsed = time.monotonic() - start
                        saved = cmd.duration_sec - elapsed
                        if saved > 0.1:
                            self._total_time_saved += saved
                        output = await ps.get_incremental_output()
                        await pool.release(ps)
                        return idx, output, True
                    await asyncio.sleep(0.5)

                # Timed out
                output = await ps.get_incremental_output()
                await pool.release(ps)
                return idx, output + f"\n[WARNING: command may not have completed within {timeout:.0f}s]", False
            except Exception as e:
                self.logger.warning(f"Pool poll failed: {e}")
                try:
                    await pool.release(ps)
                except Exception:
                    pass
                return idx, f"[ERROR: command execution failed: {e}]", False

        results = await asyncio.gather(*[poll_one(*a) for a in assignments])

        # Phase 3: Collect outputs in order, strip markers
        all_markers = {f"{_MARKER_PREFIX}{seq}__" for seq in range(1, self._marker_seq + 1)}
        output_parts = []
        any_stall = False

        for idx, output, completed in sorted(results, key=lambda x: x[0]):
            lines = output.split("\n")
            clean = "\n".join(l for l in lines if not any(m in l for m in all_markers))
            output_parts.append(clean)
            if not completed:
                any_stall = True

        if any_stall:
            self._consecutive_stalls += 1
        else:
            self._consecutive_stalls = 0

        # Handle empty-keystroke commands (just sleep briefly)
        empty_cmds = [cmd for i, cmd in enumerate(commands) if not cmd.keystrokes.strip()]
        for cmd in empty_cmds:
            await asyncio.sleep(min(cmd.duration_sec, 2.0))

        # Also get main session output
        try:
            main_out = await session.get_incremental_output()
            if main_out.strip():
                output_parts.append(main_out)
        except Exception:
            pass

        combined = "\n".join(output_parts)
        return False, self._limit_output_length(combined)

    @staticmethod
    def name() -> str:
        return "terminus-kira-env-bootstrap"

    def version(self) -> str | None:
        return "1.1.0"

    async def run(
        self, instruction: str, environment: BaseEnvironment, context: AgentContext
    ) -> None:
        """Run the agent, storing the original instruction for later use."""
        self._original_instruction = instruction
        await super().run(instruction, environment, context)

    def _get_parser(self):
        """Return None since we use native tool calling instead of parsing."""
        return None

    def _get_prompt_template_path(self) -> Path:
        """Return the path to the prompt template for native tool use."""
        return Path(__file__).parent / "prompt-templates" / "terminus-kira.txt"

    def _get_error_response_type(self) -> str:
        """Return error response type for native tool use."""
        return "response with valid tool calls"

    def _get_completion_confirmation_message(self, terminal_output: str) -> str:
        """Return task completion confirmation message for native tool use."""
        instruction = getattr(self, "_original_instruction", "N/A")
        return (
            f"Original task:\n{instruction}\n\n"
            f"Current terminal state:\n{terminal_output}\n\n"
            "Are you sure you want to mark the task as complete?\n\n"
            "[!] Checklist\n"
            "- Does your solution meet the requirements in the original task above? [TODO/DONE]\n"
            "- Does your solution account for potential changes in numeric values, array sizes, file contents, or configuration parameters? [TODO/DONE]\n"
            "- Have you verified your solution from the all perspectives of a test engineer, a QA engineer, and the user who requested this task?\n"
            "  - test engineer [TODO/DONE]\n"
            "  - QA engineer [TODO/DONE]\n"
            "  - user who requested this task [TODO/DONE]\n\n"
            "After this point, solution grading will begin and no further edits will be possible. If everything looks good, call task_complete tool again."
        )

    def _limit_output_length(self, output: str, max_bytes: int = 30000) -> str:
        return super()._limit_output_length(output, max_bytes)

    def _extract_tool_calls(self, response) -> list[dict[str, Any]]:
        """Extract tool calls from litellm response."""
        tool_calls = []
        try:
            message = response.choices[0].message
            if hasattr(message, "tool_calls") and message.tool_calls:
                for tc in message.tool_calls:
                    tool_calls.append(
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                    )
        except (AttributeError, IndexError):
            pass
        return tool_calls

    def _extract_usage_info(self, response) -> UsageInfo | None:
        """Extract usage info from litellm response."""
        try:
            usage = response.usage
            if usage:
                cost = 0.0
                try:
                    cost = litellm.completion_cost(completion_response=response) or 0.0
                except Exception:
                    pass
                return UsageInfo(
                    prompt_tokens=usage.prompt_tokens or 0,
                    completion_tokens=usage.completion_tokens or 0,
                    cache_tokens=getattr(usage, "cache_read_input_tokens", 0) or 0,
                    cost_usd=cost,
                )
        except (AttributeError, TypeError):
            pass
        return None

    def _parse_tool_calls(
        self, tool_calls: list[dict[str, Any]]
    ) -> tuple[list[Command], bool, str, str, str, ImageReadRequest | None, bool, bool]:
        """Parse tool calls into commands.

        Returns:
            Tuple of (commands, is_task_complete, feedback, analysis, plan, image_read, parallel, reset_terminal)
        """
        commands = []
        is_task_complete = False
        feedback = ""
        analysis = ""
        plan = ""
        image_read = None
        parallel = False
        reset_terminal = False

        if not tool_calls:
            feedback = (
                "WARNINGS: Your response contained no tool calls. "
                "Please use execute_commands to run commands."
            )
            return commands, is_task_complete, feedback, analysis, plan, image_read, parallel, reset_terminal

        for tool_call in tool_calls:
            function_name = tool_call.get("function", {}).get("name", "")
            arguments_str = tool_call.get("function", {}).get("arguments", "{}")

            try:
                if isinstance(arguments_str, str):
                    arguments = json.loads(arguments_str)
                else:
                    arguments = arguments_str
            except json.JSONDecodeError:
                self.logger.warning(f"Failed to parse tool arguments: {arguments_str}")
                continue

            if function_name == "execute_commands":
                # Extract analysis and plan
                analysis = arguments.get("analysis", "")
                plan = arguments.get("plan", "")
                parallel = arguments.get("parallel", False)

                # Extract commands array (Haiku sometimes double-encodes as a JSON string)
                cmds = arguments.get("commands", [])
                if isinstance(cmds, str):
                    try:
                        cmds = json.loads(cmds)
                    except json.JSONDecodeError:
                        cmds = []
                for cmd in cmds:
                    keystrokes = cmd.get("keystrokes", "")
                    duration = cmd.get("duration", 1.0)
                    commands.append(
                        Command(
                            keystrokes=keystrokes,
                            duration_sec=min(duration, 60),
                        )
                    )
            elif function_name == "task_complete":
                # Mark task as complete
                is_task_complete = True
            elif function_name == "reset_terminal":
                reset_terminal = True
            elif function_name == "image_read":
                # Extract image read request
                file_path = arguments.get("file_path", "")
                instruction = arguments.get("image_read_instruction", "")
                if file_path and instruction:
                    image_read = ImageReadRequest(
                        file_path=file_path,
                        image_read_instruction=instruction,
                    )
                else:
                    feedback = (
                        "WARNINGS: image_read requires both file_path and "
                        "image_read_instruction arguments."
                    )
            else:
                # Unknown function name - provide feedback
                feedback = (
                    f"WARNINGS: Unknown function '{function_name}'. "
                    "Please use execute_commands, task_complete, reset_terminal, or image_read."
                )
                self.logger.warning(f"Unknown function called: {function_name}")

        return commands, is_task_complete, feedback, analysis, plan, image_read, parallel, reset_terminal

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=4),
        retry=(
            retry_if_exception_type(Exception)
            & retry_if_not_exception_type(
                (
                    BadRequestError,
                    LiteLLMAuthenticationError,
                    ContextLengthExceededError,
                    OutputLengthExceededError,
                    asyncio.CancelledError,
                )
            )
        ),
        reraise=True,
    )
    async def _call_llm_for_image(
        self,
        messages: list[dict],
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> object:
        """Call litellm.acompletion with retry for transient errors.

        Retries on rate limit, network, and server errors.
        Does NOT retry on BadRequestError (e.g. image too large).
        """
        kwargs = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "timeout": 900,  # 15 minutes timeout, retry on timeout
            "drop_params": True,
        }
        # Image analysis doesn't need high reasoning effort
        # Skip reasoning_effort to use default (faster response)
        return await litellm.acompletion(**kwargs)

    async def _execute_image_read(
        self,
        image_read: ImageReadRequest,
        chat: Chat,
        original_instruction: str = "",
    ) -> str:
        """Execute a file read command to analyze an image file.

        Reads the file from the container via base64, sends it as a multimodal
        message to the LLM, and returns the analysis result.
        """
        if self._session is None:
            raise RuntimeError("Session is not set")

        file_path = image_read.file_path

        # Read image from container as base64 via harbor environment exec
        result = await self._with_block_timeout(
            self._session.environment.exec(command=f"base64 {file_path}")
        )
        if result.return_code != 0:
            error_output = result.stderr or ""
            return f"ERROR: Failed to read file '{file_path}': {error_output}"

        b64 = (result.stdout or "").replace("\n", "")

        # Determine MIME type from file extension
        ext = Path(file_path).suffix.lower()
        mime_map = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".gif": "image/gif",
            ".webp": "image/webp",
        }
        mime = mime_map.get(ext)
        if mime is None:
            return (
                f"ERROR: Unsupported image format '{ext}'. "
                f"Convert to PNG first (e.g. convert image{ext} to image.png), "
                f"then use `image_read` on the PNG file."
            )

        # Construct multimodal user message
        multimodal_messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": image_read.image_read_instruction},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime};base64,{b64}"},
                    },
                ],
            },
        ]

        messages = add_anthropic_caching(multimodal_messages, self._model_name)

        # Call LLM with retry logic
        try:
            response = await self._call_llm_for_image(
                messages=messages,
                model=self._model_name,
                temperature=self._temperature,
                max_tokens=self._llm.get_model_output_limit(),
            )
        except Exception as e:
            return f"ERROR: {e}"

        response_text = response["choices"][0]["message"]["content"]

        # Manually update token counts from litellm response
        usage = response.get("usage", {})
        if usage:
            chat._cumulative_input_tokens += usage.get("prompt_tokens", 0)
            chat._cumulative_output_tokens += usage.get("completion_tokens", 0)
            prompt_details = usage.get("prompt_tokens_details")
            cached = (
                getattr(prompt_details, "cached_tokens", 0) if prompt_details else 0
            )
            chat._cumulative_cache_tokens += cached or 0

        return f"File Read Result for '{file_path}':\n{response_text}"

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=4),
        retry=(
            retry_if_exception_type(Exception)
            & retry_if_not_exception_type(
                (
                    BadRequestError,
                    LiteLLMAuthenticationError,
                    ContextLengthExceededError,
                    OutputLengthExceededError,
                    asyncio.CancelledError,
                )
            )
        ),
        reraise=True,
    )
    async def _call_llm_with_tools(
        self,
        messages: list[dict],
    ) -> ToolCallResponse:
        """Call LLM directly with tools parameter.

        This bypasses harbor's Chat class to get access to tool_calls.
        """
        # Apply Anthropic caching
        messages = add_anthropic_caching(messages, self._model_name)

        # Build completion kwargs
        completion_kwargs = {
            "model": self._model_name,
            "messages": messages,
            "temperature": self._temperature,
            "tools": TOOLS,
            "timeout": 900,  # 15 minutes timeout, retry on timeout
            "drop_params": True,
        }

        # Add api_base if available
        if hasattr(self._llm, "_api_base") and self._llm._api_base:
            completion_kwargs["api_base"] = self._llm._api_base

        # Adaptive thinking: high for planning, default for execution, high for verification
        if self._current_episode <= self._PLANNING_EPISODES:
            effort = self._PLANNING_EFFORT
        elif self._pending_completion:
            effort = self._VERIFICATION_EFFORT
        else:
            effort = self._EXECUTION_EFFORT

        if effort is not None:
            completion_kwargs["reasoning_effort"] = effort
            completion_kwargs["temperature"] = 1

        try:
            response = await litellm.acompletion(**completion_kwargs)
        except LiteLLMContextWindowExceededError:
            raise ContextLengthExceededError()

        # Extract response data
        message = response.choices[0].message
        content = message.content or ""
        tool_calls = self._extract_tool_calls(response)
        usage_info = self._extract_usage_info(response)

        # Check for truncation
        finish_reason = response.choices[0].finish_reason
        if finish_reason == "length":
            raise OutputLengthExceededError(
                "Response was truncated due to max tokens limit",
                truncated_response=content,
            )

        # Extract reasoning content (for models that support it)
        reasoning_content = None
        if hasattr(message, "reasoning_content"):
            reasoning_content = message.reasoning_content

        return ToolCallResponse(
            content=content,
            tool_calls=tool_calls,
            reasoning_content=reasoning_content,
            usage=usage_info,
        )

    async def _handle_llm_interaction(
        self,
        chat: Chat,
        prompt: str,
        logging_paths: tuple[Path | None, Path | None, Path | None],
        original_instruction: str = "",
        session: TmuxSession | None = None,
    ) -> tuple[
        list[Command], bool, str, str, str, LLMResponse, ImageReadRequest | None, bool, bool
    ]:
        """Handle LLM interaction using native tool calling.

        This overrides the parent's _handle_llm_interaction to use native tools
        instead of JSON/XML parsing.
        """
        _, prompt_path, response_path = logging_paths

        if prompt_path is not None:
            prompt_path.write_text(prompt)

        # Build messages from chat history + new prompt
        messages = chat.messages.copy()
        messages.append({"role": "user", "content": prompt})

        try:
            start_time = time.time()
            tool_response = await self._call_llm_with_tools(messages)
            end_time = time.time()
            request_time_ms = (end_time - start_time) * 1000
            self._api_request_times.append(request_time_ms)

            # Update chat history
            assistant_message = {"role": "assistant", "content": tool_response.content}
            if tool_response.tool_calls:
                assistant_message["tool_calls"] = tool_response.tool_calls

            chat._messages.append({"role": "user", "content": prompt})
            chat._messages.append(assistant_message)

            # Add tool result messages for each tool call (required by OpenAI API)
            if tool_response.tool_calls:
                for tc in tool_response.tool_calls:
                    tool_call_id = tc.get("id", "")
                    chat._messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call_id,
                            "content": "executed",
                        }
                    )
                chat.reset_response_chain()

            # Update cumulative metrics
            if tool_response.usage:
                chat._cumulative_input_tokens += tool_response.usage.prompt_tokens
                chat._cumulative_output_tokens += tool_response.usage.completion_tokens
                chat._cumulative_cache_tokens += tool_response.usage.cache_tokens
                chat._cumulative_cost += tool_response.usage.cost_usd

        except ContextLengthExceededError:
            if not self._enable_summarize:
                self.logger.debug("Context length exceeded and summarization is OFF.")
                raise

            self.logger.debug("Context length exceeded. Using fallback summarization.")

            if session is None:
                raise RuntimeError("Cannot handle context length error without session")

            self._unwind_messages_to_free_tokens(chat, target_free_tokens=4000)

            summary_prompt = None
            try:
                summary_prompt, subagent_refs = await self._with_block_timeout(
                    self._summarize(chat, original_instruction, session)
                )
                self._pending_subagent_refs = subagent_refs
                self._pending_handoff_prompt = summary_prompt
            except Exception as e:
                self.logger.debug(f"SUMMARIZATION failed: {e}")

            if summary_prompt is None:
                current_screen = await self._with_block_timeout(
                    session.capture_pane(capture_entire=False)
                )
                limited_screen = current_screen[-1000:] if current_screen else ""
                summary_prompt = (
                    f"{original_instruction}\n\nCurrent state: {limited_screen}"
                )

            # Retry with summarized context
            messages = chat.messages.copy()
            messages.append({"role": "user", "content": summary_prompt})

            start_time = time.time()
            tool_response = await self._call_llm_with_tools(messages)
            end_time = time.time()
            request_time_ms = (end_time - start_time) * 1000
            self._api_request_times.append(request_time_ms)

            # Update chat history
            assistant_message = {"role": "assistant", "content": tool_response.content}
            if tool_response.tool_calls:
                assistant_message["tool_calls"] = tool_response.tool_calls

            chat._messages.append({"role": "user", "content": summary_prompt})
            chat._messages.append(assistant_message)

            # Add tool result messages for each tool call
            if tool_response.tool_calls:
                for tc in tool_response.tool_calls:
                    tool_call_id = tc.get("id", "")
                    chat._messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call_id,
                            "content": "executed",
                        }
                    )
                chat.reset_response_chain()

            # Update cumulative metrics
            if tool_response.usage:
                chat._cumulative_input_tokens += tool_response.usage.prompt_tokens
                chat._cumulative_output_tokens += tool_response.usage.completion_tokens
                chat._cumulative_cache_tokens += tool_response.usage.cache_tokens
                chat._cumulative_cost += tool_response.usage.cost_usd

        except OutputLengthExceededError as e:
            self.logger.debug(f"Output length exceeded: {e}")

            error_msg = (
                "ERROR!! Your response was truncated. "
                "Please provide a shorter response with fewer commands."
            )

            chat._messages.extend(
                [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": "[truncated]"},
                    {"role": "user", "content": error_msg},
                ]
            )
            chat.reset_response_chain()

            # Retry
            messages = chat.messages.copy()
            start_time = time.time()
            tool_response = await self._call_llm_with_tools(messages)
            end_time = time.time()
            self._api_request_times.append((end_time - start_time) * 1000)

            assistant_message = {"role": "assistant", "content": tool_response.content}
            if tool_response.tool_calls:
                assistant_message["tool_calls"] = tool_response.tool_calls
            chat._messages.append(assistant_message)

            # Add tool result messages for each tool call
            if tool_response.tool_calls:
                for tc in tool_response.tool_calls:
                    tool_call_id = tc.get("id", "")
                    chat._messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call_id,
                            "content": "executed",
                        }
                    )
                chat.reset_response_chain()

            # Update cumulative metrics
            if tool_response.usage:
                chat._cumulative_input_tokens += tool_response.usage.prompt_tokens
                chat._cumulative_output_tokens += tool_response.usage.completion_tokens
                chat._cumulative_cache_tokens += tool_response.usage.cache_tokens
                chat._cumulative_cost += tool_response.usage.cost_usd

        # Log response
        if response_path is not None:
            response_text = (
                f"Content: {tool_response.content or ''}\n\n"
                f"Tool Calls: {json.dumps(tool_response.tool_calls, indent=2)}"
            )
            response_path.write_text(response_text)

        # Parse tool calls into commands
        commands, is_task_complete, feedback, analysis, plan, image_read, parallel, reset_terminal = (
            self._parse_tool_calls(tool_response.tool_calls)
        )

        # Create LLMResponse for compatibility with parent class
        llm_response = LLMResponse(
            content=tool_response.content or "",
            reasoning_content=tool_response.reasoning_content,
            usage=tool_response.usage,
        )

        return (
            commands,
            is_task_complete,
            feedback,
            analysis,
            plan,
            llm_response,
            image_read,
            parallel,
            reset_terminal,
        )

    async def _gather_env_snapshot(self) -> str:
        """Gather a compact environment snapshot to eliminate early exploration turns.

        Returns a short text block with working directory, file listing,
        available languages, and package managers. On any failure, returns
        empty string so the agent falls back to normal exploration.
        """
        if self._session is None:
            return ""

        env = self._session.environment

        # Single compound command for efficiency — each section guarded by || true
        bootstrap_cmd = (
            "echo '@@PWD@@' && pwd && "
            "echo '@@LS@@' && ls -la /app/ 2>/dev/null && "
            "echo '@@LANG@@' && "
            "(python3 --version 2>&1 || echo 'python3: not found') && "
            "(gcc --version 2>&1 | head -1 || echo 'gcc: not found') && "
            "(g++ --version 2>&1 | head -1 || echo 'g++: not found') && "
            "(node --version 2>&1 || echo 'node: not found') && "
            "(java -version 2>&1 | head -1 || echo 'java: not found') && "
            "(rustc --version 2>&1 || echo 'rustc: not found') && "
            "(go version 2>&1 || echo 'go: not found') && "
            "echo '@@PKG@@' && "
            "(pip3 --version 2>&1 || echo 'pip3: not found') && "
            "(pip --version 2>&1 || echo 'pip: not found') && "
            "(apt-get --version 2>&1 | head -1 || echo 'apt-get: not found') && "
            "echo '@@MEM@@' && free -h 2>/dev/null | head -2 && "
            # Read key task files (README, instructions) to give agent context
            "echo '@@DOCS@@' && "
            "for f in /app/README* /app/readme* /app/TASK* /app/task* /app/INSTRUCTIONS* /app/instructions* /app/*.md /app/*.txt; do "
            "  if [ -f \"$f\" ] && [ $(wc -c < \"$f\") -lt 5000 ]; then "
            "    echo \"--- $f ---\"; cat \"$f\"; echo; "
            "  fi; "
            "done 2>/dev/null || true"
        )

        try:
            result = await asyncio.wait_for(
                env.exec(command=bootstrap_cmd, timeout_sec=15),
                timeout=20,
            )
        except Exception:
            return ""

        stdout = (result.stdout or "").strip()
        if not stdout:
            return ""

        # Parse sections
        sections = {}
        current_key = None
        current_lines: list[str] = []
        for line in stdout.split("\n"):
            if line.startswith("@@") and line.endswith("@@"):
                if current_key:
                    sections[current_key] = "\n".join(current_lines)
                current_key = line.strip("@")
                current_lines = []
            else:
                current_lines.append(line)
        if current_key:
            sections[current_key] = "\n".join(current_lines)

        # Build compact snapshot
        parts = []
        if "PWD" in sections:
            parts.append(f"Working directory: {sections['PWD'].strip()}")
        if "LS" in sections:
            ls_lines = sections["LS"].strip().split("\n")
            if len(ls_lines) <= 1 or (len(ls_lines) == 2 and "total 0" in ls_lines[0]):
                parts.append("/app contents: (empty directory)")
            elif len(ls_lines) > 25:
                parts.append(
                    f"/app contents ({len(ls_lines)} entries):\n"
                    + "\n".join(ls_lines[:20])
                    + f"\n... ({len(ls_lines) - 20} more files)"
                )
            else:
                parts.append(f"/app contents:\n{sections['LS'].strip()}")
        if "LANG" in sections:
            lang_lines = [
                l.strip()
                for l in sections["LANG"].strip().split("\n")
                if l.strip()
            ]
            parts.append("Available languages/tools: " + "; ".join(lang_lines))
        if "PKG" in sections:
            pkg_lines = [
                l.strip()
                for l in sections["PKG"].strip().split("\n")
                if l.strip()
            ]
            parts.append("Package managers: " + "; ".join(pkg_lines))
        if "MEM" in sections:
            mem = sections["MEM"].strip()
            if mem:
                parts.append(f"Memory: {mem}")
        if "DOCS" in sections:
            docs = sections["DOCS"].strip()
            if docs and len(docs) > 10:
                # Limit to 4000 chars to avoid overwhelming the prompt
                if len(docs) > 4000:
                    docs = docs[:4000] + "\n... (truncated)"
                parts.append(f"Task documentation found:\n{docs}")

        if not parts:
            return ""

        return "[Environment Snapshot]\n" + "\n".join(parts)

    async def _run_agent_loop(
        self,
        initial_prompt: str,
        chat: Chat,
        logging_dir: Path | None = None,
        original_instruction: str = "",
    ) -> int:
        """Run the agent loop with environment bootstrapping and image_read support."""
        if self._context is None:
            raise RuntimeError("Agent context is not set. This should never happen.")

        if self._session is None:
            raise RuntimeError("Session is not set. This should never happen.")

        # Disable pagers globally to prevent the #1 agent failure mode
        # (git log, man, etc. opening less which blocks the terminal)
        try:
            await self._session.send_keys(
                "export PAGER=cat GIT_PAGER=cat MANPAGER=cat LESS='-F -X'\n",
                block=False,
                min_timeout_sec=0.3,
            )
        except Exception:
            pass

        # Inject environment snapshot into the first prompt
        try:
            snapshot = await self._gather_env_snapshot()
            if snapshot:
                initial_prompt = f"{initial_prompt}\n\n{snapshot}"
        except Exception:
            pass  # Silent failure — don't break the agent

        # Initialize TmuxSession pool for parallel execution
        try:
            self._session_pool = TmuxSessionPool(
                self._session.environment,
                size=4,
            )
            await self._session_pool.start()
        except Exception as e:
            self.logger.warning(f"Session pool init failed: {e}")
            self._session_pool = None

        prompt = initial_prompt

        self._context.n_input_tokens = 0
        self._context.n_output_tokens = 0
        self._context.n_cache_tokens = 0
        self._context.cost_usd = None

        for episode in range(self._max_episodes):
            self._current_episode = episode
            self._n_episodes = episode + 1
            if not await self._with_block_timeout(self._session.is_session_alive()):
                self.logger.debug("Session has ended, breaking out of agent loop")
                return episode + 1

            if original_instruction and self._enable_summarize:
                proactive_summary_result = await self._with_block_timeout(
                    self._check_proactive_summarization(
                        chat,
                        original_instruction,
                        self._session,
                    )
                )
                if proactive_summary_result:
                    prompt, subagent_refs = proactive_summary_result
                    self._pending_subagent_refs = subagent_refs
                    self._pending_handoff_prompt = prompt

            logging_paths = self._setup_episode_logging(logging_dir, episode)

            # Track token counts and cost before this step
            tokens_before_input = chat.total_input_tokens
            tokens_before_output = chat.total_output_tokens
            tokens_before_cache = chat.total_cache_tokens
            cost_before = chat.total_cost

            (
                commands,
                is_task_complete,
                feedback,
                analysis,
                plan,
                llm_response,
                image_read,
                parallel,
                reset_terminal,
            ) = await self._handle_llm_interaction(
                chat, prompt, logging_paths, original_instruction, self._session
            )

            # If we have pending subagent refs, add a system step
            if self._pending_subagent_refs:
                self._trajectory_steps.append(
                    Step(
                        step_id=len(self._trajectory_steps) + 1,
                        timestamp=datetime.now(timezone.utc).isoformat(),
                        source="system",
                        message="Performed context summarization and handoff to continue task.",
                        observation=Observation(
                            results=[
                                ObservationResult(
                                    subagent_trajectory_ref=self._pending_subagent_refs
                                )
                            ]
                        ),
                    )
                )
                self._pending_subagent_refs = None

            if self._pending_handoff_prompt:
                if self._linear_history:
                    self._split_trajectory_on_summarization(
                        self._pending_handoff_prompt
                    )
                else:
                    self._trajectory_steps.append(
                        Step(
                            step_id=len(self._trajectory_steps) + 1,
                            timestamp=datetime.now(timezone.utc).isoformat(),
                            source="user",
                            message=self._pending_handoff_prompt,
                        )
                    )
                self._pending_handoff_prompt = None

            # Create message content
            if self._save_raw_content_in_trajectory:
                message_content = llm_response.content
            else:
                message_parts = []
                if analysis:
                    message_parts.append(f"Analysis: {analysis}")
                if plan:
                    message_parts.append(f"Plan: {plan}")
                message_content = "\n".join(message_parts) if message_parts else ""

            self._context.n_input_tokens = chat.total_input_tokens
            self._context.n_output_tokens = chat.total_output_tokens
            self._context.n_cache_tokens = chat.total_cache_tokens
            self._context.cost_usd = chat.total_cost if chat.total_cost > 0 else None

            self._record_asciinema_marker(
                f"Episode {episode}: {len(commands)} commands"
                + (" (image_read)" if image_read else ""),
            )

            if feedback and "ERROR:" in feedback:
                prompt = (
                    f"Previous response had parsing errors:\n{feedback}\n\n"
                    f"Please fix these issues and provide a proper "
                    f"{self._get_error_response_type()}."
                )
                cache_tokens_used = chat.total_cache_tokens - tokens_before_cache
                step_cost = chat.total_cost - cost_before
                self._trajectory_steps.append(
                    Step(
                        step_id=len(self._trajectory_steps) + 1,
                        timestamp=datetime.now(timezone.utc).isoformat(),
                        source="agent",
                        model_name=self._model_name,
                        message=llm_response.content,
                        reasoning_content=llm_response.reasoning_content,
                        observation=Observation(
                            results=[ObservationResult(content=prompt)]
                        ),
                        metrics=Metrics(
                            prompt_tokens=chat.total_input_tokens - tokens_before_input,
                            completion_tokens=chat.total_output_tokens
                            - tokens_before_output,
                            cached_tokens=cache_tokens_used
                            if cache_tokens_used > 0
                            else None,
                            cost_usd=step_cost if step_cost > 0 else None,
                            prompt_token_ids=llm_response.prompt_token_ids,
                            completion_token_ids=llm_response.completion_token_ids,
                            logprobs=llm_response.logprobs,
                        ),
                    )
                )
                continue

            if reset_terminal:
                self.logger.info("Agent requested terminal reset")
                try:
                    reset_output = await asyncio.wait_for(
                        self._reset_terminal(self._session),
                        timeout=60,  # 60s should be plenty for a reset
                    )
                    observation = reset_output
                except (asyncio.TimeoutError, Exception) as e:
                    self.logger.error(f"Terminal reset failed: {e}")
                    self._consecutive_stalls = 0
                    observation = (
                        f"[TERMINAL RESET FAILED: {e}. "
                        f"The terminal may still be stuck. Try running commands normally — "
                        f"the session may have partially recovered.]"
                    )
                # Record trajectory step
                cache_tokens_used = chat.total_cache_tokens - tokens_before_cache
                step_cost = chat.total_cost - cost_before
                tool_calls_list = [ToolCall(
                    tool_call_id=f"call_{episode}_reset",
                    function_name="reset_terminal",
                    arguments={},
                )]
                self._trajectory_steps.append(Step(
                    step_id=len(self._trajectory_steps) + 1,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    source="agent",
                    model_name=self._model_name,
                    message=message_content,
                    reasoning_content=llm_response.reasoning_content,
                    tool_calls=tool_calls_list,
                    observation=Observation(results=[ObservationResult(content=observation)]),
                    metrics=Metrics(
                        prompt_tokens=chat.total_input_tokens - tokens_before_input,
                        completion_tokens=chat.total_output_tokens - tokens_before_output,
                        cached_tokens=cache_tokens_used if cache_tokens_used > 0 else None,
                        cost_usd=step_cost if step_cost > 0 else None,
                    ),
                ))
                self._dump_trajectory()
                prompt = observation
            elif image_read is not None:
                # File read path
                image_read_result = await self._execute_image_read(
                    image_read, chat, original_instruction
                )

                # Capture pending state before modifying
                was_pending_completion = self._pending_completion

                # Handle task completion with double confirmation
                if is_task_complete:
                    if self._pending_completion:
                        observation = image_read_result
                    else:
                        self._pending_completion = True
                        observation = self._get_completion_confirmation_message(
                            image_read_result
                        )
                else:
                    self._pending_completion = False
                    if feedback and "WARNINGS:" in feedback:
                        observation = (
                            f"Previous response had warnings:\n{feedback}\n\n"
                            f"{image_read_result}"
                        )
                    else:
                        observation = image_read_result

                # Build tool_calls for image_read
                tool_calls_list: list[ToolCall] = []
                observation_results: list[ObservationResult] = []

                if not self._save_raw_content_in_trajectory:
                    tool_calls_list.append(
                        ToolCall(
                            tool_call_id=f"call_{episode}_image_read",
                            function_name="image_read",
                            arguments={
                                "file_path": image_read.file_path,
                                "image_read_instruction": image_read.image_read_instruction,
                            },
                        )
                    )
                    observation_results.append(ObservationResult(content=observation))
                    if is_task_complete:
                        tool_calls_list.append(
                            ToolCall(
                                tool_call_id=f"call_{episode}_task_complete",
                                function_name="mark_task_complete",
                                arguments={},
                            )
                        )
                else:
                    observation_results.append(ObservationResult(content=observation))

                cache_tokens_used = chat.total_cache_tokens - tokens_before_cache
                step_cost = chat.total_cost - cost_before
                self._trajectory_steps.append(
                    Step(
                        step_id=len(self._trajectory_steps) + 1,
                        timestamp=datetime.now(timezone.utc).isoformat(),
                        source="agent",
                        model_name=self._model_name,
                        message=message_content,
                        reasoning_content=llm_response.reasoning_content,
                        tool_calls=tool_calls_list or None,
                        observation=Observation(results=observation_results),
                        metrics=Metrics(
                            prompt_tokens=chat.total_input_tokens - tokens_before_input,
                            completion_tokens=chat.total_output_tokens
                            - tokens_before_output,
                            cached_tokens=cache_tokens_used
                            if cache_tokens_used > 0
                            else None,
                            cost_usd=step_cost if step_cost > 0 else None,
                            prompt_token_ids=llm_response.prompt_token_ids,
                            completion_token_ids=llm_response.completion_token_ids,
                            logprobs=llm_response.logprobs,
                        ),
                    )
                )
                self._dump_trajectory()

                if is_task_complete and was_pending_completion:
                    return episode + 1

                prompt = observation
            else:
                # Commands path (existing behavior)
                if parallel and len(commands) > 1:
                    timeout_occurred, terminal_output = await self._with_block_timeout(
                        self._execute_commands_parallel(
                            commands,
                            self._session,
                        )
                    )
                else:
                    timeout_occurred, terminal_output = await self._with_block_timeout(
                        self._execute_commands(
                            commands,
                            self._session,
                        )
                    )

                was_pending_completion = self._pending_completion

                if is_task_complete:
                    if self._pending_completion:
                        observation = terminal_output
                    else:
                        self._pending_completion = True
                        observation = self._get_completion_confirmation_message(
                            terminal_output
                        )
                else:
                    self._pending_completion = False
                    if feedback and "WARNINGS:" in feedback:
                        observation = (
                            f"Previous response had warnings:\n{feedback}\n\n"
                            f"{self._limit_output_length(terminal_output)}"
                        )
                    else:
                        observation = self._limit_output_length(terminal_output)

                # Record trajectory step
                cache_tokens_used = chat.total_cache_tokens - tokens_before_cache
                step_cost = chat.total_cost - cost_before

                tool_calls: list[ToolCall] | None = None
                observation_results: list[ObservationResult] = []

                if not self._save_raw_content_in_trajectory:
                    tool_calls_list: list[ToolCall] = []
                    if commands:
                        for i, cmd in enumerate(commands):
                            tool_calls_list.append(
                                ToolCall(
                                    tool_call_id=f"call_{episode}_{i + 1}",
                                    function_name="bash_command",
                                    arguments={
                                        "keystrokes": cmd.keystrokes,
                                        "duration": cmd.duration_sec,
                                    },
                                )
                            )
                        observation_results.append(
                            ObservationResult(content=observation)
                        )
                    if is_task_complete:
                        tool_calls_list.append(
                            ToolCall(
                                tool_call_id=f"call_{episode}_task_complete",
                                function_name="mark_task_complete",
                                arguments={},
                            )
                        )
                        if not commands:
                            observation_results.append(
                                ObservationResult(content=observation)
                            )
                    elif not commands:
                        observation_results.append(
                            ObservationResult(content=observation)
                        )
                    tool_calls = tool_calls_list or None
                else:
                    observation_results.append(ObservationResult(content=observation))

                self._trajectory_steps.append(
                    Step(
                        step_id=len(self._trajectory_steps) + 1,
                        timestamp=datetime.now(timezone.utc).isoformat(),
                        source="agent",
                        model_name=self._model_name,
                        message=message_content,
                        reasoning_content=llm_response.reasoning_content,
                        tool_calls=tool_calls,
                        observation=Observation(results=observation_results),
                        metrics=Metrics(
                            prompt_tokens=chat.total_input_tokens - tokens_before_input,
                            completion_tokens=chat.total_output_tokens
                            - tokens_before_output,
                            cached_tokens=cache_tokens_used
                            if cache_tokens_used > 0
                            else None,
                            cost_usd=step_cost if step_cost > 0 else None,
                            prompt_token_ids=llm_response.prompt_token_ids,
                            completion_token_ids=llm_response.completion_token_ids,
                            logprobs=llm_response.logprobs,
                        ),
                    )
                )
                self._dump_trajectory()

                if is_task_complete and was_pending_completion:
                    return episode + 1

                prompt = observation

        return self._n_episodes
