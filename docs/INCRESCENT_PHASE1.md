# Phase 1

**Goal:** make the existing Clawd-Code engine a reliable single-agent foundation for Increscent Code Factory.

- **Branch:** `increscent-factory-v1`. Nothing committed.
- **Platform verified:** Windows 11, Python 3.12.3, Git for Windows. Everything else runs in CI-style subprocesses.

Out of scope for Phase 1, and not built: multi-agent orchestration, teams, real git worktrees, planner, model router, deployment, browser automation, IDE extension.

---

## Problems Found

### Confirmed against the original code (`HEAD` 5a5e4dd)

Each bug was reproduced by running a scripted provider against a clean `git archive HEAD` copy of `src/`. Every reproduction FAILS there and PASSES on the Phase 1 working tree.

| # | Problem | Impact |
|---|---|---|
| A | OpenAI/GLM: the system prompt was sent only on the first request of each agent loop. | From the second tool turn on, the model lost its instructions and project context. |
| B | OpenAI/GLM: tool calls and tool results lived in a throw-away list inside `run_agent_loop`. Only assistant text reached the `Conversation`. | The next user turn, `/save` and `/load` lost all tool history. |
| C | Anthropic/MiniMax: dict tool outputs were sent as raw dicts in `tool_result.content`. | Invalid for the Messages API. Most tool calls (Read, Glob, Bash, …) return dicts. |
| D | `Conversation.max_history` dropped the oldest message with `pop(0)`. | This could leave a leading `tool_result` with no `tool_use`, or a leading assistant message, and the API rejects both. |

### Found while establishing the real test baseline

| Problem | Impact |
|---|---|
| The dependencies weren't installed during the audit. With `pip install -e .[dev]` the true baseline was **19 failed**, not 1. | Mocks and fallbacks had been hiding failures. |
| `ClawdREPL.__init__` created a prompt_toolkit `PromptSession` straight away. On Windows that needs a real console. | 18 REPL tests failed under pytest. The REPL couldn't be constructed headlessly at all. |
| **Bash ran inside WSL on Windows.** `subprocess` looks up a bare `bash` in `C:\Windows\System32` *before* `PATH`, so it found the WSL launcher. | Every Bash command the agent ran went to a Linux VM (`/mnt/c/...`, a different Python and git). WSL kept the directory handle open, which is why `test_bash_echo` failed (`WinError 32`). |
| 4 `async def` tests in `test_command_system.py` sat on plain `unittest.TestCase`. | They counted as passes without running. |
| Test runs wrote to the developer's real `~/.clawd`. | Tests weren't isolated. |

### Safety bypasses fixed

| Problem | Impact |
|---|---|
| The model-callable **Config tool** returned any config value, including `providers.*.api_key` (decoded), and could set any key or `base_url`. | Key exfiltration into the transcript, session files and logs. Prompt injection could redirect the API key to another server. |
| The Bash tool treated `cd sub && anything` as a plain `cd`. | The rest of the command was silently dropped. |
| Bash's only check was a short regex deny list. Everything else ran silently, including `git push`, `git reset --hard`, `rm -r` and `curl … \| sh`. | Dangerous operations went through with no approval. |
| Nothing stopped writing `.env` or key files, or committing them. | Secrets could leak into the repository. |
| Bash rejected Windows absolute `cwd` values (it required a leading `/`). The PowerShell tool was a stub that always refused. | No native command execution on Windows. |

### Other defects fixed along the way

- The OpenAI schema converter silently dropped the **Skill** tool, whose schema is a top-level `anyOf`. OpenAI and GLM models could never call skills.
- `max_result_size_chars` was declared on every tool but never enforced.
- `/load` left `/compact` and `/context` working on the *old* conversation.
- `/load` crashed on its preview for block-content messages.
- Session files were written without an encoding, non-atomically, with IDs at one-second resolution and no path validation.
- The `session.auto_save` config key existed but was never used.
- Git context subprocesses had no timeout and no closed stdin.

---

## Changes Made

1. **Provider-neutral conversation.** `Conversation` is the single source of truth. A new adapter module, `src/agent/message_format.py`, converts it to the Anthropic or OpenAI wire format. Each provider declares `message_format`. The agent loop no longer branches on `isinstance` and no longer keeps a second message list.
2. **Tool pairing repair.**
   - One normalisation pass merges same-role messages, puts `tool_result`s first, drops orphaned results, and synthesises a result for interrupted tool calls.
   - History trimming only cuts at the start of a user turn. If no safe cut point exists, it lets the history exceed the soft limit rather than corrupt it.
3. **Agent loop.**
   - The system prompt goes on every request.
   - Every assistant turn (text and tool calls) is persisted.
   - All tool results of a turn are stored as one user message.
   - Tool output is serialised and truncated to `max_result_size_chars`.
   - `AgentLoopResult` gains `stop_reason`, `tool_calls`, `tool_errors` and `permission_denials`, all backwards compatible.
4. **Sessions.**
   - Schema v2: messages with timestamps, metadata (project root, git root, cwd, app version, platform) and restorable tool state (cwd, todos, tasks).
   - Collision-safe IDs, ID validation, atomic UTF-8 writes.
   - `list_sessions` and `latest_session_id`.
   - v1 files still load.
   - The REPL autosaves after every turn (honours `session.auto_save`). `/resume` was added.
5. **Cross-platform shell** (`src/tool_system/shell.py`).
   - Windows prefers Git Bash, found by absolute path and never WSL. It falls back to PowerShell (`pwsh`, then Windows PowerShell), then cmd. POSIX keeps `bash -lc`.
   - stdin is closed, output is UTF-8, and the timeout kills the whole process tree. A timeout returns exit code 124; it doesn't raise.
   - The PowerShell tool now works, with the same safety checks.
6. **Configuration layers:** CLI > env > project > user > default, with source tracking. Project settings may only set `provider` and `model`. API keys come only from env or user config, and project files can't set a base URL.
7. **Non-interactive mode.**
   - `clawd run` / `increscent run` with exit codes 0/1/2/3, `--output json`, `--stream`, `--yes`, `--resume`, `--cwd`, `--max-turns` and `--no-save`.
   - Input can also be piped on stdin.
   - `clawd sessions` was added.
8. **Project-root detection** (`src/project.py`): git first, then markers, then cwd. Home and filesystem roots are never picked up from below, and are flagged when they are the start directory. `run` refuses them unless `--allow-unsafe-root` is passed. The system prompt now states the project root, git root, OS and the shell the Bash tool will use.
9. **Safety baseline** (`src/tool_system/safety.py`):
   - Every tool is labelled READ/WRITE/EXECUTE.
   - Shell commands are classified as `read`, `mutating`, `dangerous` (needs approval) or `blocked` (always refused).
   - Secret files (`.env*`, keys) and credential-looking content need approval before writing.
   - `git add`/`commit` is blocked while an un-ignored `.env` would be staged.
   - The Config tool redacts reads and refuses to set credentials.
10. **Observability** (`src/observability.py`): structured JSON-lines events with recursive secret redaction. It is off by default and turned on with `--log-file`, `CLAWD_LOG_FILE` or `--verbose`.
11. **Test quality:**
    - The async tests really run now (`IsolatedAsyncioTestCase`).
    - A hygiene test fails if the never-awaited pattern returns.
    - `tests/conftest.py` sandboxes `Path.home()` and clears credential env vars for every test.
12. **Packaging:** `increscent` console script alias (same `src.cli:main`). `tiktoken` synced into `requirements.txt` to match `pyproject.toml`.

---

## Architecture Changes

```
CLI (src/cli.py)  ── REPL (src/repl/core.py)          ── interactive
                  └─ run  (src/headless.py)            ── scripts / CI
        │
        ├─ config.resolve_provider_config()   CLI > env > project > user > default
        ├─ project.detect_project()           project root / git root / unsafe-root guard
        ├─ agent.session.Session              v2 files: conversation + metadata + tool state
        └─ tool_system.agent_loop.run_agent_loop()
               │  conversation (neutral blocks) ──► agent.message_format ──► provider wire format
               │                                     (anthropic | openai, declared per provider)
               ├─ ToolRegistry.dispatch() ── check_permissions() ── safety.classify_command()
               │                                                 └─ shell.run_shell_command()
               └─ observability.log_event()   redacted structured events
```

**New modules:** `src/agent/message_format.py`, `src/project.py`, `src/config.py` (layered resolution added), `src/branding.py`, `src/observability.py`, `src/headless.py`, `src/tool_system/shell.py`, `src/tool_system/safety.py`.

**Extension points for the Increscent transition:**
- `src/branding.py` is the single place for the app name, CLI names, directory names and env prefixes. `INCRESCENT_*` env vars already take precedence over `CLAWD_*`.
- `providers.register_provider(name, module, class, info)` and `PROVIDER_INFO` fields `api_key_env`, `base_url_env` and `wire_format`. An OpenAI-compatible backend (Ollama, Qwen/DashScope and similar) needs one registry entry plus a thin `OpenAICompatibleProvider` subclass. Gemini needs a new wire format in `message_format.py`.
- `run_headless(options, provider_factory=…)` is the programmatic entry point for a future `increscent build`.
- `ToolResult.error_kind` and `AgentLoopResult.permission_denials` are the hooks for the Phase 2 permission-policy engine.

**Removed:** nothing user-facing. `src/repl.py`, the mirror layer and all existing commands are untouched.

---

## Tests Added

There are 152 new tests plus 92 subtests. Shared helpers: `tests/fakes.py` (scripted providers that record exact requests) and `tests/conftest.py` (isolation).

| File | Covers |
|---|---|
| `test_message_format.py` | Anthropic/OpenAI conversion, tool_result serialisation, tool pair repair, system placement, OpenAI schema conversion of every default tool |
| `test_agent_loop_conversation.py` | System prompt on every request, tool call/result persistence, multi-turn and multiple tool calls, the Anthropic structure, truncation during a loop, size limit, max turns, permission denials |
| `test_conversation_history.py` | Safe trimming, the soft limit, valid API messages after trimming |
| `test_session_persistence.py` | create → message → tool → save → reload → continue (OpenAI and Anthropic), file contents, tool state, ID safety, v1 compatibility, list and latest |
| `test_shell_execution.py` | Shell detection (never WSL), stdout/stderr/exit code, cwd, Unicode, timeout and tree kill, closed stdin, cmd, PowerShell, the Bash `cd` fix |
| `test_safety_baseline.py` | READ/WRITE/EXECUTE labels, a 45-case command classification table, `.env` commit guard, secret files and content, ask/deny/approve flows, Config tool redaction |
| `test_config_layers.py` | Every precedence layer, `INCRESCENT_` over `CLAWD_`, project restrictions, secrets never printed, `CLAWD_HOME` |
| `test_project_detection.py` | Git root from a subdirectory, worktree `.git` file, nested repos, markers, home/root guard, this repo |
| `test_headless_run.py` | Exit codes 0/1/2/3, `--yes`, JSON output, resume, no-save, stream, CLI in-process and as a real subprocess |
| `test_observability.py` | Redaction, JSON-lines sink, lifecycle events, provider errors, no secret in logs |
| `test_repl_phase1.py` | Lazy prompt session, resolved config, autosave, `/load` rebinding, `/resume`, direct-stream guard, **real interactive startup driven through prompt_toolkit pipe input** |
| `test_test_hygiene.py` | Fails if an async test is put on a plain `TestCase` |

## Tests Fixed

| Test(s) | Fix |
|---|---|
| `test_command_system.py` (4 async tests) | The test was incorrect: `TestCommandEngine` and `TestInitCommand` now inherit `unittest.IsolatedAsyncioTestCase`. The tests now run and pass. |
| `test_repl.py` (18 tests) | **Code fix, tests unchanged.** The `PromptSession` is created lazily on first interactive use. |
| `test_tool_system_tools.py::TestBashTool::test_bash_echo` | **Code fix, test unchanged.** Commands now run in Git Bash instead of WSL, and the process-tree cleanup releases the directory. The test isn't marked platform-specific and passes on Windows. |
| `test_safety_baseline.py` (new) | One expectation in my own table was wrong: `rm -rf ./build` is correctly *dangerous*, not merely mutating. |

---

## Test Results

The suite runs with `python -m pytest tests/` inside `.venv`, created with `pip install -e ".[dev]"` from `pyproject.toml`.

| Run | Total | Passed | Failed | Skipped | Warnings |
|---|---|---|---|---|---|
| Audit (no deps installed) | 252 | 250 | 1 | 1 | 8 |
| **True baseline** (deps installed, before changes) | 252 | 232 | **19** | 1 | 8 (4 async tests never ran) |
| **After Phase 1** | **404** | **403** | **0** | **1** | **0** |

The one skipped test is the pre-existing `test_config.py:211`: "POSIX file permission semantics differ on Windows". It is platform-specific by design. It is **not** skipped for missing dependencies, and no test is.

These new tests are conditional and **all ran on this Windows machine**:
- On Linux/macOS: `test_windows_bash_is_never_wsl` and `test_cmd_shell_on_windows` skip there by design.
- Where PowerShell isn't installed: the PowerShell tests skip.

**Linting and type checking:** none are configured in this repository (no ruff, mypy or flake8 config), and I didn't add or install any. Instead I compiled every `src/` and `tests/` file, and scanned the changed files for unused imports. No new ones were introduced; the pre-existing ones were left alone.

**End-to-end CLI verification** used no real API keys (none are configured on this machine). The real `increscent.exe` and the real OpenAI and Anthropic SDKs were pointed via `--base-url` at a local mock server that records every HTTP request:

| Check | Result |
|---|---|
| `increscent run` from a `src/` subdirectory (OpenAI wire) | exit 0. Project and git root detected at the repo root. Bash tool wrote a file via Git Bash. |
| On-the-wire OpenAI requests | `system` first on **every** request. `assistant.tool_calls` followed by one `tool` message per call. All 43 tools sent (Skill included). |
| Anthropic wire, `--output json -q` | exit 0. `system` on every request. Both `tool_result`s in one user turn, as strings. |
| `--resume <id>` | The continued request carried the full earlier tool history. |
| Dangerous `rm -r build` without / with `--yes` | exit **3** and `build/` untouched / exit 0 and removed. |
| Run from the home directory | exit **2**, refused. |
| Missing API key | exit **2** with the env var name. |
| `--log-file` | All lifecycle events present. The key string appears **0** times. |
| `config` with a key in env | Key printed **0** times. Sources shown. |
| Interactive REPL | Started through the real `main()`, header shows the project, `/tools`, `/resume`, `/exit` → 0. This is also an automated test. |

---

## CLI Usage

```bash
# setup (once)
python -m venv .venv
.venv\Scripts\python -m pip install -e ".[dev]"      # Windows  (POSIX: .venv/bin/python)

# interactive (unchanged)
clawd                    # or: increscent, python -m src.cli
clawd --stream --provider openai --model gpt-5.4

# non-interactive
increscent run "Find and fix the failing tests"
increscent run --yes --max-turns 40 "Refactor utils.py and run the tests"
increscent run --output json -q "Summarise the repo" > result.json
echo "Explain src/cli.py" | increscent run -
increscent run --resume latest "Now add tests for it"
increscent run --cwd C:\Projects\shishu-mandir "Check the build"

# misc
increscent sessions              # this project's sessions (--all for every project)
increscent config                # effective config + sources (keys masked)
increscent --log-file run.jsonl run "..."     # structured logs
```

REPL additions: `/resume [id]` (latest session for this project) and autosave after every turn.

**Exit codes (`run`):**

| Code | Meaning |
|---|---|
| 0 | Success |
| 1 | Task failure (provider/runtime error, turn limit reached, interrupted) |
| 2 | Configuration error (unknown provider, missing key, auth failure, unsafe project root, bad arguments) |
| 3 | Permission/security rejection (a tool call was refused) |

## Configuration

| Setting | CLI | Environment | Project `.clawd/settings.json` | User `~/.clawd/config.json` |
|---|---|---|---|---|
| provider | `--provider` | `INCRESCENT_PROVIDER`, `CLAWD_PROVIDER` | `"provider"` | `default_provider` |
| model | `--model` | `INCRESCENT_MODEL`, `CLAWD_MODEL` | `"model"` | `providers.<p>.default_model` |
| base URL | `--base-url` | `ANTHROPIC_BASE_URL`, `OPENAI_BASE_URL`, `GLM_BASE_URL`, `MINIMAX_BASE_URL` | *not allowed* | `providers.<p>.base_url` |
| API key | *not allowed* | `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `ZHIPUAI_API_KEY`/`GLM_API_KEY`/`ZAI_API_KEY`, `MINIMAX_API_KEY` | *not allowed* | `providers.<p>.api_key` (`clawd login`) |

Other settings:
- `INCRESCENT_HOME` / `CLAWD_HOME`: relocate `~/.clawd` (config, sessions, history).
- `CLAWD_LOG_FILE`: JSON-lines log path.
- `CLAWD_BASH`: explicit bash path.

`.env` files are **not** loaded automatically, because a repository could otherwise set your base URL. Existing `~/.clawd/config.json` files keep working unchanged.

---

## Known Limitations

- **No live-model test.** No provider keys were available. Wire formats were verified against a local mock server and the official SDKs, not the real Anthropic or OpenAI endpoints.
- **Classification is pattern-based.** The command classifier doesn't parse the shell, so obfuscated commands (variables, `eval`, base64) can evade it. It is a Phase 1 baseline, not a sandbox. Workspace path sandboxing applies to file tools, but not to arbitrary paths inside shell commands.
- **`--yes` approves everything classed as needing approval**, including `git push` and `git commit`. There are no per-rule allow lists yet (Phase 2).
- **Tool outputs aren't redacted for the model.** Reading a `.env` file is still allowed, and its contents go to the provider. Redaction applies to logs, the Config tool and progress output.
- **Windows without Git Bash:** Bash-tool commands run in PowerShell or cmd. Bash-only syntax fails there, and the result's `shell` field tells the model which shell ran.
- **Context management:** the history soft limit can grow during one very long tool loop. There is no auto-compaction yet, and `/compact`'s summary request still sends the last 20 messages without the new pair repair.
- **REPL keyword heuristic:** the REPL's "direct stream" heuristic (keyword-based, no tools) is kept for compatibility. It is now skipped whenever the history holds tool blocks.
- **`AgentTool`, teams, cron, MCP clients and worktrees are unchanged stubs** (see the audit).
- **No linter or type checker is configured** in the repo.
- **Package name.** The package is still named `src`, and README/FEATURE_LIST haven't been updated for Phase 1.

## Phase 2 Recommendations

1. **Permission-policy engine.** Permission modes (default, accept-edits, plan, bypass) and allow/deny rules such as `Bash(git status:*)` and `Write(src/**)`. It would build on `safety.classify_command`, `ToolResult.error_kind` and `AgentLoopResult.permission_denials`. It would replace the blanket `--yes` and merge the two interactive permission prompts.
2. **Real sub-agents.** An `AgentProfile` (role prompt, model, tool subset, permission profile) plus an `AgentRunner` extracted from `run_agent_loop`, replacing the batch-call `AgentTool`. Enforce skills' `allowed-tools`.
3. **Git/worktree service.** `git worktree add -b`, diff/commit/merge helpers, per-agent path sandbox at the worktree root.
4. **Durable task store** under the project (`.clawd/` → `.increscent/` via `branding.py`), shared by the future coordinator.
5. **Auto-compaction** using `compact_service` at a context-window threshold. Route `/compact` through `message_format` normalisation.
6. **Hooks runtime** (pre/post tool, on stop) wired to the observability events.
7. **Provider additions** via `register_provider`: Ollama and Qwen (OpenAI-compatible), and Gemini as a new wire format. Also retries/backoff and rate-limit handling in one provider wrapper.
8. **Tooling.** Add ruff and mypy config, and a CI matrix on Windows and Linux that runs `pytest` plus an opt-in live smoke test (`-m live`, skipped without keys).
