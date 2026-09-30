# Increscent Code Factory — Repository Audit

- **Repository:** Clawd-Code (Python rewrite of Claude Code), fork at `origin = github.com/devopsachistar/Clawd-Code`
- **Branch audited:** `increscent-factory-v1` @ `5a5e4dd` (identical to `main` and `origin/main`; no commits ahead)
- **Audit date:** 2026-09-30
- **Mode:** read-only. No source files modified, no packages installed, no commits.

> **Working definition used in this audit.** "Increscent Code Factory" is assumed to mean an
> agentic software-production system: a coordinator breaks a goal into tasks, dispatches them to
> specialised sub-agents (planner, implementer, tester, reviewer) that work in isolated git
> worktrees, gates their changes through tests and review, and merges the results. This definition
> is an assumption; revise it before Phase 1 if the intended product is different.

---

# Current Architecture

## 1. Two separate CLIs share one `src/` package

| Entry point | What it is | Status |
|---|---|---|
| `python -m src.cli` (console script `clawd = src.cli:main`) | **The real agent CLI.** argparse → REPL. | Working |
| `python -m src.main <subcommand>` | "Porting workspace" inspector: lists mirrored TS commands/tools from JSON snapshots, fake `turn-loop`, simulated `remote-mode`/`ssh-mode`/`teleport-mode`. | Scaffolding, no LLM |

`src/` has about 148 Python files. Roughly a third are real runtime code. The rest are either the **mirror layer** (`runtime.py`, `query_engine.py`, `session_store.py`, `tools.py`, `commands.py`, `port_manifest.py`, `parity_audit.py`, `remote_runtime.py`, `direct_modes.py`, `bootstrap_graph.py`, `command_graph.py`, `tool_pool.py`, and others) or **27 placeholder packages**: `hooks/`, `plugins/`, `coordinator/`, `memdir/`, `bridge/`, `services/`, `state/`, `server/`, `remote/`, `vim/`, `voice/` and more. Each placeholder is a 16-line `__init__.py` that only loads `reference_data/subsystems/<name>.json`.

## 2. How the CLI starts

```
python -m src.cli [--stream] [login|config|--version|--config]
  └─ src/cli.py:main()
       ├─ --version       → prints src.__version__ (0.1.0)
       ├─ login           → handle_login(): Rich prompts → config.set_api_key / set_default_provider
       ├─ config/--config → show_config(): masked keys from ~/.clawd/config.json
       └─ (default)       → start_repl(stream)
              └─ config.get_default_provider()
              └─ src.repl.ClawdREPL(provider_name, stream)   # src/repl/core.py
                     .run()  → prompt_toolkit loop
                        ├─ "/..." → handle_command()  (built-ins, command_system, skills)
                        └─ text  → chat() → run_agent_loop()
```

Note: `src/repl/` (package) shadows `src/repl.py` (legacy `ClawdCodexCLI` built on the mirror layer). The legacy module can't be imported as `src.repl`, so it is dead code. `tests/test_legacy_repl_module.py` covers this case.

## 3. Providers and model selection

- Registry: `src/providers/__init__.py`. `PROVIDER_INFO` is a dict of label, default base URL, default model and model list for **anthropic, openai, glm, minimax**. `get_provider_class()` is an `if` chain with lazy imports.
- Base: `src/providers/base.py`. `BaseProvider` defines `chat()`, `chat_stream()` (text only), and the optional `chat_stream_response()` (structured streaming with tool calls). `ChatResponse` holds content, usage, finish_reason and `tool_uses`, normalised to Anthropic shape (`{id, name, input}`).
- `AnthropicProvider` and `MinimaxProvider` use the Anthropic SDK. MiniMax works through its Anthropic-compatible endpoint.
- `OpenAICompatibleProvider` is the base for `OpenAIProvider` (openai SDK) and `GLMProvider` (zhipuai SDK). It converts Anthropic tool schemas to OpenAI function format.
- **Selection:** always `config["default_provider"]` from `~/.clawd/config.json`. Models come from `providers.<name>.default_model` in the same file.
  - The CLI has no `--provider` or `--model` flag.
  - There are no environment variables for API keys (`python-dotenv` is a dependency but is never imported).
  - The model can't be switched per request or per agent.
- **Config file:** `src/config.py`. The first load creates `~/.clawd/config.json`. API keys are base64-encoded, which hides them from a casual glance but isn't encryption. The file is chmod 600 on POSIX; on Windows the permissions aren't restricted. `session.auto_save` and `session.max_history` are written but never read.
- Model catalogs are duplicated in `PROVIDER_INFO` and in each provider's `get_available_models()`, and they're already out of date.

## 4. Tool registration

- Protocol (`src/tool_system/registry.py`): each tool has `spec() -> ToolSpec` (name, description, JSON schema, aliases, `is_read_only`, `is_destructive`, `max_result_size_chars`), `run(input, ctx) -> ToolResult`, and an optional `check_permissions()`.
- `build_default_registry()` (`src/tool_system/defaults.py`) hard-codes 41 tool instances. It then adds `AgentTool(registry)` and `ToolSearchTool(registry)`, and loads user tools from `~/.clawd/tools/*.py` (`loader.py`: either a `TOOL` object or a `tool_spec` dict with `run()`).
- Every registered tool's schema goes to the model on every turn. There is no filtering by skill, agent, mode or deferral, even though `ToolSearch` exists.

## 5. Tool execution

`ToolRegistry.dispatch()` runs these steps in order:
1. Look up the tool by name or alias.
2. `ensure_tool_allowed` (deny list).
3. `validate_json_schema`.
4. `check_permissions` → allow, deny, or ask.
5. On ask, call `context.permission_handler(tool, msg, suggestion)`.
6. `tool.run()`.

Tools run **synchronously and one at a time** in the agent loop. A tool raises `ToolInputError` or `ToolPermissionError`, and the agent loop catches these and returns them as a tool error. `max_result_size_chars` is declared but **never enforced**.

## 6. The agent loop (`src/tool_system/agent_loop.py`)

`run_agent_loop(conversation, provider, registry, ctx, max_turns=20, stream, on_event, on_text_chunk)`:

1. Builds the tool schemas and the system prompt: output-style prompt plus `build_context_prompt()`.
2. Branches on `isinstance(provider, (AnthropicProvider, MinimaxProvider))`:
   - **Anthropic path:** uses `Conversation` content blocks directly. `system` is passed as a kwarg.
   - **OpenAI path:** keeps a separate local `openai_messages` list. It is seeded from string-only messages in `Conversation`. The system message is prepended on turn 0 only.
3. Calls `provider.chat_stream_response()` when streaming. If that fails, it falls back to `chat()`.
4. Adds up usage. If there are no `tool_uses`, it returns `AgentLoopResult`. Otherwise it dispatches each tool, emits `ToolEvent`s, appends the results, and loops.
5. After `max_turns` it returns `"[Max tool turns reached]"`.

Before the loop, `ClawdREPL._should_try_direct_stream()` checks the prompt against a keyword list. If no keyword matches, it streams a reply with no tools at all.

## 7. Context assembly (`src/context_system/`)

`build_context_prompt()` runs again on **every** `run_agent_loop` call. It includes:
- a workspace snapshot: date, root, cwd, Python and test file counts, key files, top-level entries
- git context: branch, latest commit, `git status` snapshot, collected by `subprocess` git calls
- instruction files: `~/.clawd|.claude/CLAUDE.md`, then `CLAUDE.md`, `.clawd/CLAUDE.md` and `.claude/CLAUDE.md` walking up from cwd to the workspace root. Limits are 6 files, 4k characters per file and 12k characters in total.

Supporting modules:
- `context_analyzer.py`: token breakdown for `/context`
- `microcompact.py`: clears old tool results
- `compact_service/`: LLM summarisation for `/compact`
- `token_estimation.py`: tiktoken with a character-count fallback

None of these run automatically. There is no auto-compaction and no context-window guard.

## 8. Sessions and persistence

| Store | Location | Used by |
|---|---|---|
| `src/agent/session.py` `Session` | `~/.clawd/sessions/<YYYYMMDD_HHMMSS>.json` | REPL `/save`, `/load <id>` (manual only) |
| `src/session_store.py` `StoredSession` | `./.port_sessions/` | mirror CLI only |
| prompt history | `~/.clawd/history` | prompt_toolkit |
| team | `<workspace>/.clawd/team.json` | TeamCreate |
| plan | `<workspace>/.clawd/plan.md` | ExitPlanMode |

Gaps:
- There's no auto-save and no `/resume`.
- Session IDs have one-second resolution.
- Files are opened without `encoding=` (a Windows code-page risk).
- `ToolContext` state isn't persisted: tasks, todos, cwd, crons, team, read fingerprints.
- `Conversation.max_history=100` drops the oldest message (`pop(0)`), which can leave a `tool_result` without its `tool_use`.

## 9. Permissions

- `ToolPermissionContext` (`tool_system/permissions.py`): tool-name deny list and prefixes, path sandbox (workspace root plus extra directories), and the `allow_docs` flag.
- `PermissionResult` allow/deny/ask (`permission_handler.py`).
- **Only `Write` and `Edit` implement `check_permissions`**, and they only ask about writing `.md` files.
- `Bash` has a regex deny list (`sudo`, `rm -rf /`, fork bomb, and a few others). Otherwise it runs any command without asking.
- There are no permission modes (default, acceptEdits, plan, bypass), no persistent allow or deny rules, and no per-agent policy.
- There are two duplicate prompt implementations: `InteractivePermissionHandler` and `ClawdREPL._handle_permission_request`.

## 10. Agents and teams

- `AgentTool` (alias `Task`) **is not a sub-agent.** It runs a list of `{name, input}` tool calls in order through the same registry. No LLM is involved, and it has no separate context or tool subset.
- `TeamCreate`/`TeamDelete` write or delete `.clawd/team.json` and set `ctx.team`. Nothing reads it.
- `SendMessage` appends to `ctx.outbox`, and nothing reads that either.
- `coordinator/` is a placeholder. There is no multi-agent runtime.

## 11. Git and worktrees

- Git is **read-only** today: `context_system/git_context.py` runs status, branch and last commit.
- `EnterWorktree` does `mkdir .clawd/worktrees/<name>` and changes `ctx.cwd`. **It does not run `git worktree add`**, create a branch, or check anything out, so `worktreeBranch` is always `None`. `ExitWorktree` only resets cwd.
- No commit, diff, merge or PR tooling exists apart from what the model runs through Bash.

## 12. Skills

- `src/skills/` loads `SKILL.md` files with YAML-ish frontmatter from these places:
  - `$CLAWD_SKILLS_DIR`, `$CLAUDE_SKILLS_DIR`
  - `~/.clawd/skills`, `~/.claude/skills`
  - the project's `.clawd/skills` and `.claude/skills`
  - `$CLAWD_MANAGED_SKILLS_DIR`
- Supported: argument substitution (`$ARGUMENTS`, `$0`, `$name`) and `${CLAUDE_SKILL_DIR}`.
- A `/skill-name args` slash command → `Skill` tool → prompt text → `self.chat(prompt)`.
- **Parsed but ignored:** `allowed-tools`, which is only displayed, never enforced. `model`, `context: fork`, `agent`, `effort` and `paths` are also ignored.
- The legacy Python skills path runs `~/.clawd/skills/<name>.py` `run()`.
- Repo examples: `.clawd/skills/echo-arg`, `.clawd/skills/explain-code`.

## 13. Slash commands (`src/command_system/`)

- The registry, engine, `LocalCommand` and `PromptCommand` types are real.
- Built-ins: help, clear, exit, skills, cost, context, compact, init.
- The REPL also hard-codes save, load, multiline, stream, render-last, tools and tool.
- Routing is split between `ClawdREPL.handle_command` special cases and the command system, and that split is fragile.

## 14. MCP, LSP, tasks and cron

- `MCPTool` and the MCP resource tools read `ctx.mcp_clients`, but **nothing ever fills it in**. There is no MCP protocol client (stdio or HTTP), no config file and no discovery. The tools always return "not connected". LSP has the same problem.
- `TaskCreate`/`Get`/`List`/`Update`/`Output` edit an in-memory `ctx.tasks` dict, which is lost on exit.
- `TaskManager` (threads with stop events) exists and `TaskStop` uses it, but no tool ever starts a managed task.
- `CronCreate` stores entries in memory and they never run.
- `TodoWrite` is in-memory.
- `PowerShell`, `NotebookEdit`, `REPL` and `RemoteTrigger` are stubs.
- `WebSearch` scrapes the DuckDuckGo HTML page with a regex, which breaks easily.

---

# Existing Capabilities

These are verified by reading the code and running the tests:
- An installable CLI with `login`/`config`/`--version`. `--help` and `--version` work without the provider SDKs installed.
- A rich prompt_toolkit REPL: history, slash completion, a `/` palette, multiline input, `/stream` and `/render-last`.
- Four providers behind one interface, all normalised to Anthropic-style tool calls. Anthropic and MiniMax (Anthropic-compatible) stream text with tool calls, and so do the OpenAI-compatible providers.
- A working single-agent tool loop with streaming, tool events and usage totals.
- Real file tools (Read, Write, Edit, Glob, Grep) with a path sandbox, read-before-write fingerprints and diff utilities. Also real Bash, WebFetch, WebSearch, AskUserQuestion, plan mode and a tool search.
- User-extensible tools (`~/.clawd/tools/*.py`) and markdown skills with arguments.
- Context injection: workspace, git, and a CLAUDE.md hierarchy.
- Manual `/compact` (LLM summary plus microcompact), `/context` token analysis and `/cost`.
- Manual session save and load.
- Output styles (`default`, `explanatory`, plus custom from a directory).
- A test suite: 17 files, 252 tests, all offline and mocked.

---

# Existing Gaps

**Correctness bugs found in this audit (to confirm with tests before fixing):**
1. **OpenAI/GLM path drops the system prompt after turn 0.** In the loop, `api_messages = openai_messages` without the system message on turns 1 and later.
2. **OpenAI/GLM tool history isn't persisted.** Only assistant text is written to `Conversation`, so tool calls and results disappear between user turns and from saved sessions. Each new user turn re-seeds from text only.
3. **Anthropic tool results can be dicts.** `add_tool_result_message(tool_id, result_output)` passes dict outputs straight through as `tool_result.content`. The Anthropic API expects a string or a list of content blocks, so this is likely rejected in real use. Tests mock the provider, so they don't catch it.
4. **The history cap can corrupt the transcript.** `max_history=100` drops the oldest message and can leave a `tool_result` without its `tool_use`.
5. **The Bash tool isn't portable to Windows.**
   - The `cwd` argument must start with `/`.
   - It runs `bash -lc`. On this machine, `bash` may resolve to WSL. The one failing test fails while cleaning up a temp directory: `WinError 32`, file in use.
6. **The keyword heuristic in `_should_try_direct_stream`** can send a real coding request down a path with no tools. For example, "update the docs" matches no keyword.
7. **4 async tests in `tests/test_command_system.py` never run.** They are unittest methods declared `async` without an async test case, so they "pass" without executing. You can see this in the `RuntimeWarning: coroutine ... was never awaited` warnings.
8. `CostTracker.record` adds up token counts with a label. No cost is computed.

**Missing capabilities needed for a code factory:**
- Real sub-agents: isolated context, their own system prompt, a tool subset, and a model per agent.
- A coordinator or orchestrator. Task graph with dependencies. Durable task state.
- Real `git worktree` and branch lifecycle, commit, diff and merge. Merge-conflict handling.
- A permission policy engine: modes, allow and deny rules, Bash command classification, per-agent sandboxes.
- A headless or non-interactive mode (`-p "prompt"`, JSON output). Today everything is REPL-only, which blocks automation and CI.
- An MCP client runtime and a hooks runtime (pre and post tool use).
- Auto-compaction and a context-budget guard. `/resume` and auto-save.
- Provider and model selection from the CLI or env. Retries and backoff. Rate-limit handling. Cancelling a running tool.
- Parallel tool execution and background tasks.
- Observability: a structured event log or trace per run.

---

# Important Files

| Area | File(s) |
|---|---|
| CLI entry | `src/cli.py`, `pyproject.toml` (`[project.scripts] clawd = "src.cli:main"`) |
| REPL | `src/repl/core.py` (1348 lines, a god class), `src/repl/__init__.py` |
| Agent loop | `src/tool_system/agent_loop.py` |
| Tool framework | `src/tool_system/{registry,protocol,context,defaults,loader,schema_validation,errors,diff_utils,task_manager}.py` |
| Tools | `src/tool_system/tools/*.py` (28 files) |
| Permissions | `src/tool_system/permissions.py`, `src/tool_system/permission_handler.py` |
| Providers | `src/providers/{__init__,base,anthropic_provider,openai_compatible,openai_provider,glm_provider,minimax_provider}.py` |
| Config | `src/config.py` (→ `~/.clawd/config.json`) |
| Conversation / session | `src/agent/conversation.py`, `src/agent/session.py` |
| Context | `src/context_system/*.py`, `src/compact_service/*.py`, `src/token_estimation.py` |
| Commands | `src/command_system/*.py` |
| Skills | `src/skills/*.py`, `src/tool_system/tools/skill.py`, `.clawd/skills/` |
| Output styles | `src/outputStyles/*.py` |
| Mirror layer (not runtime) | `src/main.py`, `src/runtime.py`, `src/query_engine.py`, `src/session_store.py`, `src/tools.py`, `src/commands.py`, `src/reference_data/**`, 27 placeholder packages |
| Tests | `tests/*.py` (17 files) |
| Docs | `README.md`, `FEATURE_LIST.md` (Chinese), `CONTRIBUTING.md`, `CHANGELOG.md`, `docs/guide/*`, `docs/i18n/*` |

---

# Reusable Components

Keep these, possibly with light changes:
- **`ToolSpec` / `ToolRegistry` / `ToolResult` / `ToolContext`.** A clean, small protocol. Add filtering (`registry.subset(names)`) for per-agent tool sets.
- **File tools** (Read, Write, Edit, Glob, Grep), `diff_utils`, read fingerprints and the path sandbox. They are solid and well tested.
- **`schema_validation.py`**, the dependency-free JSON-schema checker.
- **The provider abstraction and `ChatResponse` normalisation**, including the OpenAI tool-schema conversion and streaming tool-call assembly.
- **`context_system`**: git context, CLAUDE.md loader, workspace snapshot and context analyzer.
- **`compact_service` + `microcompact`.** They work and only need to be called automatically.
- **The skills loader and frontmatter parser**, and argument substitution.
- **`command_system`** registry, engine and types.
- **`TaskManager`**, a thread plus stop-event primitive. It can back background agents.
- **`outputStyles`**, `token_estimation`, and the user-tool `loader`.
- **Test fixtures and patterns.** The suite runs fully offline and fast (about 16 s).

---

# Components That Need Refactoring

| Component | Problem | Direction |
|---|---|---|
| `agent_loop.py` | isinstance provider branching; two message stores; bugs 1–3; sequential; not reusable by sub-agents | One provider-neutral message model (`Conversation` as the source of truth). Providers translate on the way out. Loop becomes an `AgentRunner` class with policy hooks, a budget and cancellation. |
| `repl/core.py` | 1.3k-line god class mixing UI, command routing, provider init, permissions and relogin | Split into `ui/`, `app/session_controller`, and permission prompter adapters. REPL becomes one thin frontend next to a headless frontend. |
| `providers/__init__.py` + `config.py` | hard-coded if-chain; base64 "obfuscation"; no env/CLI override; duplicated model lists | A provider registry with entry points. Config layers (defaults → user → project → env → CLI flags). Keys from env or the OS keyring. |
| Permissions | only `.md` asks; Bash unguarded; duplicate handlers | A `PermissionPolicy` engine: modes and rules (`Bash(git status:*)`, `Write(src/**)`), a Bash command classifier, and per-agent profiles. |
| `AgentTool`, Team, SendMessage | fake sub-agents, write-only state | Replace with real sub-agent spawning on top of `AgentRunner`. Mailbox and team state persisted in the factory store. |
| Worktree tools | mkdir only | `git worktree add -b <branch>`, cleanup, and diff/merge helpers in a `git/` service. |
| Tasks, Todo, Cron | in-memory dicts | A durable task store (SQLite or JSON-lines under `.increscent/`) shared by the coordinator and agents. |
| Session persistence | manual, second-resolution IDs, no encoding | Append-only JSONL transcripts with UUIDs, auto-save and `/resume`. |
| Mirror layer and placeholder packages | about 60% of `src/` is non-functional scaffolding; confuses navigation | Move to `archive/` or remove in a dedicated PR (tests in `test_porting_workspace.py` and `test_claude_code_tool_parity.py` depend on it). |
| Packaging | top-level import package named `src`; `requirements.txt` lacks `tiktoken` and mixes dev deps; URLs point to upstream; no ruff, mypy or pytest config | Rename the package (for example `increscent/`) in one mechanical PR. Keep `pyproject.toml` as the single source of dependencies. Add tooling config. |
| `.gitignore` | ignores `README.md`, `pyproject.toml`, `FEATURE_LIST.md`, `CHANGELOG.md`, `CONTRIBUTING.md`, `CLAUDE_example.md`. They're tracked, so it has no effect now, but new copies would be silently ignored. | Remove those lines. |

---

# Proposed Increscent Code Factory Architecture

```
                 ┌──────────────────────────── Frontends ────────────────────────────┐
                 │  REPL (existing, slimmed)   Headless CLI (-p / --json)   (later) API │
                 └───────────────┬───────────────────────────────┬───────────────────┘
                                 ▼                               ▼
                    ┌──────────────────── Factory Core ─────────────────────┐
                    │  Coordinator: goal → TaskGraph → dispatch → gate → merge│
                    │  TaskStore (durable, .increscent/state.db or JSONL)     │
                    │  EventBus / RunLog (structured trace per run)           │
                    └───────┬───────────────┬───────────────┬────────────────┘
                            ▼               ▼               ▼
                    ┌──────────── AgentRunner (one per agent) ─────────────┐
                    │ AgentProfile: role prompt · model · tool subset ·      │
                    │   permission profile · budget (turns/tokens/time)      │
                    │ Conversation (provider-neutral) · auto-compaction      │
                    │ Loop: model → policy check → tools (parallel if R/O) → │
                    │   hooks → results → repeat · cancellation              │
                    └───────┬───────────────┬───────────────┬──────────────┘
                            ▼               ▼               ▼
      ┌────────────┐ ┌────────────┐ ┌──────────────┐ ┌────────────┐ ┌──────────────┐
      │ Providers  │ │ ToolRegistry│ │ PermissionPolicy│ │ Git/Worktree│ │ Context svc  │
      │ registry + │ │ (existing) │ │ modes · rules · │ │ service    │ │ (existing +  │
      │ retry/rate │ │ + subset() │ │ bash classifier │ │ add/diff/  │ │ auto-compact)│
      │ limit      │ │ + MCP tools│ │ + hooks         │ │ merge/clean│ │              │
      └────────────┘ └────────────┘ └──────────────┘ └────────────┘ └──────────────┘
```

**Standard pipeline (default roles):**
1. **Planner:** read-only tools. Writes the TaskGraph (tasks, dependencies, acceptance criteria).
2. **Implementer(s):** one git worktree and branch per task (`increscent/<run>/<task>`), with write tools inside that worktree only.
3. **Tester:** runs the project's test command in the worktree and reports a structured pass or fail.
4. **Reviewer:** read-only. Diffs against the base and approves or requests changes, with a bounded retry loop.
5. **Integrator:** merges the approved branches into an integration branch in dependency order. It stops for a human on a conflict or a failed gate. It never pushes or opens a PR without explicit configuration.

**Design principles:**
- One message model. Providers are adapters.
- Everything an agent does is a logged event. Runs can be resumed from the TaskStore.
- Safety by default: agents can't write outside their worktree, and Bash is classified and ask-by-default in headless mode unless a rule allows it.
- Keep the existing REPL working at every phase.

---

# Phase 1

**Goal: fix the single-agent foundation and make it automatable. No factory features yet.**

1. **Baseline tests.** Fix the 4 never-awaited async tests. Make the Bash test pass on Windows. Add regression tests that reproduce bugs 1–4 (these fail first).
2. **Unify the message model.**
   - `Conversation` stores provider-neutral blocks, and each provider serialises to its own wire format.
   - Remove `_is_anthropic_provider` branching and the parallel `openai_messages` list.
   - Fixes bugs 1–3.
3. **Tool-result hygiene.** Serialise dict outputs to JSON text. Enforce `max_result_size_chars`. Make history truncation keep tool_use and tool_result pairs together (fixes bug 4).
4. **Extract `AgentRunner`** from `run_agent_loop` (keep a compatibility wrapper). Add budget, cancellation and event callbacks.
5. **Config layering.** Env vars (`ANTHROPIC_API_KEY`, and so on), `--provider`/`--model` flags, and a project config at `.increscent/config.toml`. Keep `~/.clawd/config.json` readable for backward compatibility.
6. **Headless mode.** `clawd -p "<prompt>" [--output json] [--max-turns N]`, with no REPL dependency.
7. **Portable shell tool.** Windows-aware Bash and a working PowerShell tool. Accept native absolute `cwd` values.

**Exit criteria:** all tests green on Windows and POSIX. The Anthropic and OpenAI paths each pass a multi-turn tool conversation test that survives save and load. A headless run works end to end.

# Phase 2

**Goal: real sub-agents, permissions, worktrees and durable state.**

1. `PermissionPolicy`: modes (`default`, `accept-edits`, `plan`, `bypass`), allow and deny rule syntax, a Bash command classifier (read-only vs mutating vs dangerous), and one shared prompter. Delete the duplicate handler.
2. `AgentProfile`s, loaded from `.increscent/agents/*.md` using frontmatter the same way skills do. Each has a role prompt, model, tool subset and permission profile. Enforce skill `allowed-tools`, `model` and `context: fork` with the same machinery.
3. Replace `AgentTool` with a real sub-agent spawn that returns a summary. Keep the `Task` alias.
4. `git/` service: `worktree add -b`, list, remove, diff, commit (inside the agent's worktree only), and a merge check. Rewrite `EnterWorktree`/`ExitWorktree` on top of it.
5. Durable `TaskStore` and session transcripts (JSONL plus UUIDs). Auto-save and `/resume`.
6. Auto-compaction at a configurable share of the context window, reusing `compact_service`.
7. Hooks runtime (pre and post tool, on stop), configured in project settings.

# Phase 3

**Goal: the factory.**

1. `Coordinator` with a TaskGraph model: plan → fan-out to implementers in parallel worktrees → test gate → review gate → integrate.
2. `increscent run "<goal>"` (headless) and a `/factory` REPL command. Add `increscent status`, `increscent resume <run>` and `increscent abort <run>`.
3. Parallel agent execution (thread pool or asyncio) with per-run concurrency and token and cost budgets. Add real cost accounting per model.
4. MCP client runtime (stdio first) that registers MCP tools into the registry per agent.
5. A run log and report: a markdown or JSON summary of the tasks, diffs, test results and review verdicts.
6. Optional, disabled by default: push branches or open a PR through `gh`, only with explicit per-run approval.
7. Rebrand and packaging: rename the package, rename the console script to `increscent` (keeping `clawd` as an alias), and archive the mirror layer.

---

# Risks

| Risk | Impact | Mitigation |
|---|---|---|
| Message-model refactor breaks both provider paths | High: core loop | Write the failing regression tests first. Keep `run_agent_loop` as a compatibility wrapper. Land provider by provider. |
| Tests mock every provider, so real API regressions go unseen | High | Add opt-in live smoke tests (`-m live`, skipped without keys). Add recorded-fixture tests for the wire formats. |
| Parallel agents editing one repo | High: data loss or conflicts | Strict one-worktree-per-agent. The path sandbox is rooted at the worktree. The integrator is the only thing that merges. |
| Unguarded Bash in autonomous mode | High: security | The Phase 2 policy engine must land **before** Phase 3 autonomy. Headless defaults to deny or ask for mutating commands. |
| Runaway cost or loops across many agents | Medium–High | Per-agent and per-run budgets for turns, tokens and wall-clock time. Hard stop plus a summary. |
| Renaming the `src` package and archiving the mirror layer | Medium: large diffs, broken imports | Do it in isolated mechanical PRs after behaviour is stable (Phase 3). Don't mix it with logic changes. |
| Windows as the primary dev platform (this machine) | Medium | CI matrix with Windows and Linux. No assumptions about bash or `/`. |
| Provider SDKs not installed locally | Low–Medium | Tests already pass without them. Document `pip install -e .[dev]` in a venv. |
| Upstream divergence (fork of GPT-AGI/Clawd-Code) | Low–Medium | Decide early whether to track upstream. A rename makes later merges costly. |
| Scope creep from the 30+ half-implemented mirrored tools | Medium | Remove stub tools (`REPL`, `NotebookEdit`, `RemoteTrigger`, `TestingPermission`) from the default registry rather than finishing them. |

---

# Recommended Implementation Order

1. **Test-baseline PR.** Fix the never-awaited async tests and the Windows Bash test. Add failing regression tests for bugs 1–4.
2. **Message-model unification.** Covers bugs 1–3 and tool-result serialisation, truncation pairing and result-size limits.
3. **`AgentRunner` extraction**, keeping a `run_agent_loop` wrapper.
4. **Config layering**, the provider and model flags, and env-var keys.
5. **Headless `-p` mode** with JSON output.
6. **Portable shell and PowerShell tools.**
7. **`PermissionPolicy` engine** and the Bash classifier. Remove the duplicate handler.
8. **Durable JSONL sessions**, auto-save and `/resume`, then auto-compaction.
9. **Git/worktree service**, then rewrite the worktree tools.
10. **`AgentProfile` + real sub-agents** (replace `AgentTool`). Enforce skill tool limits.
11. **Durable TaskStore.**
12. **Hooks runtime.**
13. **Coordinator and factory pipeline** (plan → implement → test → review → integrate), then parallelism and budgets.
14. **MCP stdio client.**
15. **Run reports, cost accounting, and optional PR integration.**
16. **Rebrand and packaging.** Package rename, `increscent` console script, archive the mirror layer, clean up `.gitignore`, update the docs.

Each step is its own branch and PR from `increscent-factory-v1`, and each keeps the test suite green.

---

## Appendix A — Baseline test status (at audit time)

Command: `python -m pytest tests/ -q -p no:cacheprovider` (Python 3.12.3, Windows 11). Only pytest, rich and python-dotenv are installed. anthropic, openai, zhipuai, prompt_toolkit and tiktoken are **not** installed, and the tests pass without them thanks to mocks and fallbacks.

```
1 failed, 250 passed, 1 skipped, 8 warnings in 16.36s
FAILED tests/test_tool_system_tools.py::TestBashTool::test_bash_echo
       PermissionError: [WinError 32] … temp dir in use (bash subprocess holds cwd on Windows)
SKIPPED tests/test_config.py:211 — POSIX file permission semantics differ on Windows
WARNINGS: 4 coroutine tests in tests/test_command_system.py "never awaited" (false passes)
```

## Appendix B — How to run the current application

```bash
# from repo root, inside a virtualenv
pip install -e .[dev]            # or: pip install -r requirements.txt  (then also: pip install tiktoken)
python -m src.cli login          # choose provider, enter API key → ~/.clawd/config.json
python -m src.cli                # start REPL   (or: clawd, after pip install -e .)
python -m src.cli --stream       # REPL with live streaming
python -m src.cli config         # show config (keys masked)
python -m src.main summary       # porting-workspace inspector (no LLM)
python -m pytest tests/ -v       # tests
```
