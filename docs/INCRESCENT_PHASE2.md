# Phase 2: Safe, persistent, multi-agent engineering

Phase 2 turns the Phase 1 single-agent foundation into a multi-agent system. Every agent has its own git worktree, branch, tool set, permission policy and budget. A durable task graph drives the work, an orchestrator coordinates it, and a human approves the merge.

**Scope boundary:** nothing is ever pushed, deployed or merged into the main branch. Browser automation, deployment and the full product factory are Phase 3+ and were not built.

- Branch: `increscent-factory-v1`. Not committed.
- Tests: **503 passed, 1 skipped (Windows-only POSIX test, pre-existing), 0 failed, 0 warnings.**

---

## Architecture

```
CLI  (src/cli.py + src/cli_team.py)
 ├─ run / REPL (Phase 1)                         ── single agent, session policy
 └─ team / tasks / agents / worktrees / review   ── multi-agent
        │
        ▼
Orchestrator (src/orchestration/orchestrator.py)
 ├─ Planner (planner.py)             goal ──► TaskSpecs (standard chain or JSON plan)
 ├─ TaskStore (src/taskgraph/)       durable graph: status, deps, attempts, history, results
 ├─ WorktreeManager (src/gitops/)    real git worktrees + branches + locks
 ├─ AgentRunner (src/agents/runner)  one isolated LLM session per task attempt
 │    ├─ ProfileRegistry (profiles/*.md)   role, tools, capabilities, budgets
 │    ├─ ToolRegistry.subset(profile tools) + ReportResult + MessageOrchestrator
 │    └─ run_agent_loop(budget, should_cancel)
 ├─ MessageBus (agents/messages.py)  auditable structured messages
 ├─ Test runner (testing.py)         measured test counts
 └─ EventRecorder (state.py)         events.jsonl (persisted, redacted)

Every tool call ─► ToolRegistry.dispatch ─► PermissionEngine (src/security/permissions.py)
Every tool result / provider request / log / session / report ─► secret boundary (src/security/secrets.py)
```

**Durable state** lives in `<git-common-dir>/increscent/`, inside `.git`. It is shared by all worktrees, survives restarts, and can't be committed by accident:

| File | Contents |
|---|---|
| `tasks.json` | Graphs and tasks |
| `agents.json` | Agent records (status, result, usage, session id) |
| `worktrees.json` | Worktree registry (task, role, base commit) |
| `locks/*.lock` | Worktree locks (owner, pid) |
| `messages.jsonl` | Structured agent/orchestrator messages |
| `events.jsonl` | Structured event log |
| `sessions/*.json` | Each agent's full, redacted conversation |
| `reports/<graph>.md` and `.json` | Delivery reports |

Projects without git fall back to `<project>/.clawd/state/`. Multi-agent runs require git.

**New modules:**
- `src/security/permissions.py`, `src/security/secrets.py`
- `src/agents/`: `models`, `profiles` (+ `profiles/*.md`), `runner`, `store`, `messages`, `tools`
- `src/gitops/worktrees.py`
- `src/taskgraph/`: `models`, `store`
- `src/orchestration/`: `orchestrator`, `planner`, `state`, `testing`, `reports`, `simulation`
- `src/cli_team.py`

**Changed modules:**
- `ToolRegistry.dispatch` (engine on every call; new `subset()`)
- `ToolContext` (policy, approval handler, agent attribution, secret registration)
- `agent_loop` (budgets, cancellation, outgoing and tool-result redaction)
- Read, Grep, Bash, PowerShell, Config, EnterWorktree and ExitWorktree tools
- Context builder, observability, sessions, headless, REPL, config, CLI, branding, packaging

---

## Agent model

An **Agent** (`src/agents/models.py`) is one isolated LLM session working on one task attempt. It is *not* a list of tool calls: the old batch `Agent` tool is unchanged and still subject to the permission engine.

| Field | Meaning |
|---|---|
| `id`, `name`, `role`, `profile` | Identity (`A-xxxxxxxx`) |
| `system_prompt` | Profile instructions + execution contract + worktree context |
| `provider`, `model` | Per-agent provider/model |
| `allowed_tools` | The profile's tool subset (+ ReportResult, MessageOrchestrator) |
| `permissions` | The profile's `PermissionPolicy` (capability → ALLOW/ASK/DENY) |
| `working_directory` | Its git worktree, which is also its file-tool sandbox |
| `task_id`, `attempt` | What it works on |
| `budget` | `max_iterations`, `max_tool_calls`, `max_seconds`, `max_tokens` |
| `timeout_s`, `status`, timestamps, `result`, `error`, `usage`, `session_id` | Lifecycle and outcome |

**Lifecycle:** `CREATED → RUNNING ⇄ WAITING (approval pending) → COMPLETED | FAILED | CANCELLED`. Invalid transitions raise.

**Structured result** (`AgentResult`), which every run produces:
- `task_id`, `agent_id`, `status` (completed / failed / blocked), `summary`
- `files_changed`, `tests_run`, `tests_passed`, `tests_failed`
- `risks`, `warnings`, `commit`, `next_action`
- Provenance: `reported_by_agent`, `stop_reason`, `review_verdict`, `error`

How results are established:
- Agents report through the `ReportResult` tool. **An agent that stops without calling it has failed**, and its prose is never used as a result.
- The orchestrator then **replaces claims with facts**:
  - `files_changed` comes from `git status` in the worktree.
  - `commit` comes from the orchestrator's own commit.
  - Test counts come from running the tests itself (QA tasks and final verification).
- Mismatches become warnings. A QA agent claiming "all green" when tests fail makes the task fail.

**Budgets** (`LoopBudget` in `agent_loop.py`) are checked before every model call and every tool call:
- The **iteration** limit is `max_turns`.
- **Tool-call**, **token** (when the provider reports usage) and **wall-clock** limits are enforced as well.
- Cancellation is supported.
- A stopped batch still records a result for every tool call, so tool pairing stays valid.
- Hitting any budget fails the attempt with `stop_reason` = `max_turns`, `budget_tool_calls`, `budget_tokens`, `timeout` or `cancelled`.

## Agent profiles

Profiles are **data**, Markdown with frontmatter; the orchestrator hard-codes none of them.

- **Built-in:** `src/agents/profiles/*.md`
- **User overrides:** `~/.clawd/agents/*.md`
- **Project overrides:** `<project>/.clawd/agents/*.md`

| Profile | Allowed tools | ALLOW | ASK | Budget (iter / tools / s) | Commits |
|---|---|---|---|---|---|
| Architect | Read Glob Grep Write Edit TodoWrite | READ WRITE | NETWORK | 25 / 60 / 1200 | yes (docs) |
| Developer | + Bash PowerShell | READ WRITE EXECUTE GIT_BRANCH | NETWORK | 40 / 150 / 2400 | yes |
| FrontendDeveloper | same as Developer | same | NETWORK | 40 / 150 / 2400 | yes |
| BackendDeveloper | same | same | NETWORK DATABASE_READ | 40 / 150 / 2400 | yes |
| DatabaseEngineer | same | same | NETWORK DATABASE_READ DATABASE_WRITE SECRET_USE | 30 / 100 / 1800 | yes |
| QAEngineer | same | same | NETWORK | 20 / 80 / 1800 | yes (tests) |
| SecurityReviewer | Read Glob Grep Bash | READ EXECUTE | none | 20 / 60 / 1200 | no (read-only) |
| CodeReviewer | Read Glob Grep Bash | READ EXECUTE | none | 20 / 60 / 1200 | no (read-only) |
| DevOpsEngineer | Developer tools | READ WRITE EXECUTE GIT_BRANCH | NETWORK DEPLOY_STAGING | 30 / 100 / 1800 | yes |

- Unlisted capabilities are **DENY**. Every profile denies `GIT_COMMIT` and `GIT_PUSH` to the agent itself; the orchestrator commits on its behalf when `commits: true`.
- **Safety clamp:** no profile, including built-in, user and project profiles, can pre-approve `GIT_PUSH`, `DATABASE_WRITE`, `DEPLOY_STAGING`, `DEPLOY_PRODUCTION` or `SECRET_USE`. These are forced to ASK, with a warning.
- **Project-level profiles** (shipped inside a repository) are treated as untrusted: unknown tools are dropped.

---

## Permission model

`src/security/permissions.py`. `ToolRegistry.dispatch` evaluates **every tool invocation**, including calls nested inside the batch `Agent` tool. It never relies on prompt instructions.

1. **Assess** (`assess_tool_call`): the call's capabilities, risk, and whether it is hard-blocked.
   - Shell commands are analysed per pattern. For example, `git push` → GIT_PUSH, `vercel --prod` → DEPLOY_PRODUCTION, `psql … DROP TABLE` → DATABASE_WRITE/CRITICAL, and `cat .env` → SECRET_USE.
   - `secret_env` → SECRET_USE.
   - `Write`/`Edit` of secret files or credential content → SECRET_USE.
   - `Config` changes to keys or endpoints → SECRET_USE.
2. **Decide** (`PermissionEngine.evaluate`) → `ALLOW | DENY | ASK`, applying these checks in order:
   - Hard-blocked commands → DENY (CRITICAL).
   - The tool's own `check_permissions` deny → DENY.
   - A matching **deny rule** → DENY.
   - Any capability the policy denies → DENY (unless an explicit allow rule matches).
   - Ask rules, asked capabilities, Phase 1 destructive patterns and tool asks → ASK.
   - An explicit **allow rule** may waive ordinary asks, but never CRITICAL risk or SECRET_USE.
3. **Approve.** ASK produces a structured `ApprovalRequest` (id, tool summary, capabilities, risk, reasons, policy, `auto_approvable`, agent/task). Only the caller's approval handler can grant it; no handler means denied.

```
Tool: Bash: git push origin main
Capability: EXECUTE, GIT_PUSH
Risk: HIGH
Decision: ASK
Why: GIT_PUSH requires approval; ...
This request requires explicit approval (cannot be auto-approved).
```

**Capabilities:** READ, WRITE, EXECUTE, NETWORK, GIT_BRANCH, GIT_COMMIT, GIT_PUSH, DATABASE_READ, DATABASE_WRITE, DEPLOY_STAGING, DEPLOY_PRODUCTION, SECRET_USE.

**Risk mapping** (the highest applicable level wins):

| Risk | Capabilities / operations |
|---|---|
| LOW | READ, GIT_BRANCH (create/switch branch) |
| MEDIUM | WRITE, EXECUTE, NETWORK, GIT_COMMIT, DATABASE_READ, recursive delete inside the workspace |
| HIGH | GIT_PUSH, DATABASE_WRITE, DEPLOY_STAGING, SECRET_USE, `git reset --hard`, `git rebase`, `git branch -D`, pipe-to-shell |
| CRITICAL | DEPLOY_PRODUCTION (cloud deploys, `kubectl apply`, `terraform apply`, publishing packages, `npm run deploy`), destructive SQL / migration resets, repo deletion, everything hard-blocked (sudo, disk formatting, `rm -rf /`, …) |

**Auto-approval** (`--yes`) covers only requests with **risk ≤ MEDIUM** and none of GIT_PUSH, DEPLOY_STAGING, DEPLOY_PRODUCTION, DATABASE_WRITE or SECRET_USE. Production deployment, database destruction, credential operations, pushes and anything HIGH or CRITICAL always need an explicit human decision. **This changes Phase 1 behaviour:** `--yes` no longer approves `git push`.

**Policies:**

| Principal | Policy |
|---|---|
| Interactive REPL / `run` | `session_default`: READ, WRITE, EXECUTE, NETWORK and GIT_BRANCH are ALLOW; everything else is ASK. This preserves Phase 1 behaviour. |
| Agents | Their profile's policy |

**Rules** (`Tool` or `Tool(argument-glob)`):

```json
// ~/.clawd/config.json: allow / ask / deny
"permissions": {"allow": ["Bash(npm test*)"], "ask": ["Write(*.lock)"], "deny": ["Bash(git push*)"]}
// <project>/.clawd/settings.json: ask / deny only (a repository cannot pre-approve anything)
```

**Approval UIs:**

| Interface | Behaviour |
|---|---|
| REPL | Shows the full request and prompts. |
| `run` | `--yes` covers auto-approvable requests only; anything else is denied with the reason, and the run exits with code 3. |
| `team` | Prompts on a TTY (serialised across parallel agents); `--yes` covers auto-approvable requests. |
| Merge | Needs an interactive "y" or an explicit `--approve` flag. `--yes` does not count. |

Approval lifecycle events are logged and also persisted to `events.jsonl` for agent runs: `approval.requested`, then `approval.approved` or `approval.denied`, with the agent entering `agent.waiting` meanwhile.

## Secret model

`src/security/secrets.py`. Its goal is that secret values do not enter model context, logs, session files or reports.

**Detection:**
1. **Known values.** At workspace start, values from project `.env*` files (up to depth 3) with sensitive names or credential-like values are registered in a process-wide registry, never persisted. So are sensitive environment variables. Every occurrence of a registered value is masked.
2. **Patterns.** Anthropic/OpenAI/Stripe/GitHub/GitLab/AWS/Google/Slack/npm/SendGrid tokens, JWTs, Bearer tokens, PEM private keys, Azure `AccountKey=`, and passwords in URLs.
3. **Sensitive assignments.** `STRIPE_SECRET_KEY=…` becomes `STRIPE_SECRET_KEY=<SECRET>`. Env-style lines are always masked. In code, quoted literals are masked only when they look like credentials, so ordinary code stays editable.

**Model-visible form:**

```
DATABASE_URL=<SECRET>
STRIPE_SECRET_KEY=<SECRET>
```

**Where it applies:**

| Surface | Mechanism |
|---|---|
| Read tool | Secret files (`.env*`, `*.pem`, `*.key`, credentials, …) are returned with every value masked; `secretsRedacted: true` |
| Grep tool | Matching lines from secret files are masked |
| Bash / PowerShell | Output is masked; commands that read secrets (`cat .env`, `printenv`, `echo $API_KEY`, …) need SECRET_USE approval |
| Every tool result | Redacted before it enters the conversation (`bound_tool_result_content`) |
| Every provider request | The outgoing messages and system prompt are redacted, covering model-written tool inputs and user-pasted secrets |
| Context builder | CLAUDE.md contents are redacted |
| Config tool | Reads redacted; setting credentials is refused; changing endpoints needs approval |
| Logs / events | `observability.redact` → secret boundary |
| Sessions | `Session.to_dict` redacts conversation, metadata and state |
| Messages / reports | The message bus and delivery reports redact everything |

**Controlled use without disclosure:** `Bash(command, secret_env=["DATABASE_URL"])`.
- The value comes from project `.env` files or the environment and is injected into the child process environment only.
- It requires explicit SECRET_USE approval (never auto-approved), and any echo of it is masked.
- Tests prove the process receives the real value while the model sees `<SECRET>`.

---

## Task model

`src/taskgraph/`. JSON store with locked read-modify-write and atomic replace.

- **Task fields:** `id`, `title`, `description`, `status`, `priority`, `owner`, `agent`, `dependencies`, `worktree`, `branch`, `files_changed`, `tests`, `attempts`, `created_at`, `updated_at`, `result`, `error`. Plus `graph_id`, `profile`, `kind` (project/task), `max_attempts`, `blocked_reason` and `history` (one entry per attempt).
- **Statuses:** PENDING, READY, RUNNING, BLOCKED, REVIEW, COMPLETED, FAILED, CANCELLED.

**Dependencies** (`refresh` / `ready_tasks`):
- A task becomes READY only when **all** of its dependencies are COMPLETED.
- A FAILED or CANCELLED dependency makes dependents BLOCKED, and they recover when it is retried.
- Cycles and unknown dependencies are rejected before anything is written.
- READY tasks are ordered by priority.
- The **project task** is the root; it depends on every task and moves to REVIEW when the graph is done.

**Standard plan:** Architecture → Data model → Backend → Frontend → Integration tests → Code review. Custom plans can be given as JSON with `--plan`.

**Control:** `pause`, `resume`, `retry` (FAILED only; it keeps the history) and `cancel`, through the API or `increscent task …`.

## Worktree model

`src/gitops/worktrees.py`

```
C:\Projects\myapp                                             main repository (never modified)
C:\Projects\.increscent-worktrees\myapp-T-1a2b3c4d-frontend   branch increscent/T-1a2b3c4d/frontend
C:\Projects\.increscent-worktrees\myapp-G-9f8e7d6c-integration branch increscent/G-9f8e7d6c/integration
```

- `create`, `list`, `get`, `status`, `changed_files`, `commit`, `diff`, `merge_branch` and `remove`, all using real `git worktree` and branches.
- A task's worktree starts from its dependency's branch. With several dependencies, the others are merged in; a conflict marks the task BLOCKED.
- **Locking:** one `O_EXCL` lock file per worktree records owner and pid. Two agents can't use the same worktree. Locks from dead processes are reclaimed; the pid check uses `OpenProcess` on Windows, never `os.kill(pid, 0)`.
- `create` is idempotent, so retries reuse the preserved worktree and its partial work.
- **Commit safety:**
  - Commits use an identifiable agent author, `agent+<role>@increscent.local`.
  - Un-ignored secret files block the commit.
- **Merge safety:**
  - `merge_branch` refuses main/master.
  - A merge conflict aborts cleanly.
- **Removal safety:**
  - `remove` refuses worktrees it didn't create, locked worktrees, and uncommitted changes (unless `--force`).
  - Branches are kept unless `--delete-branch` is given.
- `EnterWorktree`/`ExitWorktree` (model tools) now create real worktrees in git repositories. In a directory with no git (or no commits) they keep the old scratch-directory behaviour, marked `isGitWorktree: false`.

## Orchestration model

`src/orchestration/orchestrator.py`

```
create_graph ─► run: loop { ready tasks ─► (≤ max_parallel threads) execute ─► collect }
execute(task): worktree(from deps) ─► lock ─► create agent ─► TASK_ASSIGNMENT
             ─► AgentRunner.run ─► verify (git files, tests) ─► commit (if profile commits)
             ─► TASK_RESULT / REVIEW_RESULT ─► COMPLETED | retry | FAILED | BLOCKED ─► unlock
finalize: all COMPLETED ─► verification tests in the final worktree ─► REVIEW_REQUEST ─► graph REVIEW
review / prepare_merge(approved=True) ─► increscent/<graph>/integration   (never main, never pushed)
```

- **Communication** (`MessageBus`): the message types are TASK_ASSIGNMENT, TASK_RESULT, BLOCKED, QUESTION, ERROR, REVIEW_REQUEST and REVIEW_RESULT.
  - Messages are validated, redacted and appended to `messages.jsonl`.
  - Agents can only send result, blocked, question, error and review-result messages, and **only to the orchestrator**.
  - Agents have no channel to each other and can't reach another agent's shell or worktree.
- **Events** (persisted to `events.jsonl`, redacted):
  - Agents: `agent.created`, `agent.started`, `agent.tool_started`, `agent.tool_completed`, `agent.waiting`, `agent.completed`, `agent.failed`, `agent.cancelled`
  - Tasks: `task.created`, `task.ready`, `task.started`, `task.blocked`, `task.retry`, `task.completed`, `task.failed`, `task.cancelled`, `task.resumed`
  - Approvals: `approval.requested`, `approval.approved`, `approval.denied`
  - Git and tests: `worktree.created`, `worktree.removed`, `commit.created`, `merge.completed`, `merge.prepared`, `merge.conflict`, `tests.run`
  - Graph and messages: `graph.created`, `graph.planned`, `graph.started`, `graph.finished`, `review.requested`, `message.sent`
- **Human merge flow:** agent completes → review (CodeReviewer agent plus `increscent review`) → human approval → merge into the integration branch.

## Failure recovery

When an agent fails (no structured result, provider error, budget exhausted, tests failing, or a git error):

1. The error is captured and the task state saved: `error`, a `history` entry (attempt, agent, status, summary, error, stop_reason, warnings), `task.retry` or `task.failed`, and an ERROR message.
2. **The worktree and branch are preserved**, and the lock is released.
3. While `attempts < max_attempts` (`--max-attempts`, default 3), the task returns to READY.
4. **The next attempt is not a blind repeat.** Its assignment includes every previous attempt's error, summary, stop reason and warnings, lists the uncommitted work still in the worktree, and instructs the agent to diagnose before taking a different approach.
5. When attempts are exhausted, the task is FAILED and its dependents BLOCKED. `increscent task retry <id>` grants another attempt.
6. If an orchestrator run dies mid-task, the next `team --resume <graph>` records those RUNNING tasks as interrupted, cancels their agents and retries them.

---

## CLI examples

```bash
# safe offline demonstration (scripted agents, real git/tools/tests; nothing deployed or merged)
increscent team --simulate "Create a simple task management application"

# real model (needs a provider key); 2 agents in parallel; auto-approve LOW/MEDIUM only
increscent team --provider anthropic --parallel 2 --yes "Add CSV export to the reports page"
increscent team --plan plan.json "Custom plan"
increscent team --resume G-1a2b3c4d

increscent tasks [--graph G-…]          increscent task show T-…
increscent task pause|resume|retry|cancel T-…
increscent agents [--graph G-…]         increscent agent show A-…
increscent worktrees [--all]            increscent worktree remove <name> [--force] [--delete-branch]
increscent review G-…                    # per-task branches, commits, diffstats, verdicts, risks
increscent review G-… --merge            # asks; or --merge --approve (explicit) → increscent/G-…/integration
```

**Exit codes:**

| Command | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| `team` | Graph reached REVIEW | Failed / blocked | Config: not git, no key, unknown profile | — |
| `review --merge` | Merged into the integration branch | Merge conflict | — | Not approved |

Verified demo result (Windows, simulation): six tasks were completed by six agents in six worktrees, with six branches and five commits (the reviewer is read-only). The QA tests ran with 3/3 passing, verification ran 3/3 passing, and the reviewer returned **approve**. The graph ended in REVIEW with `main` unchanged; `review --merge --approve` then produced `increscent/<graph>/integration`.

---

## Tests

| File | Covers |
|---|---|
| `test_permission_engine.py` | Capability and risk mapping (22 commands), tool capabilities, policies, rules, auto-approval limits, structured requests, the dispatch approval flow, configured user/project rules, headless `--yes` limits |
| `test_secret_boundary.py` | Redaction units, Read/Grep/Bash/Config/context surfaces, controlled `secret_env`, and **no secrets in model messages (both wire formats), session files, logs or delivery reports** |
| `test_agents.py` | The 9 profiles, clamps, overrides, lifecycle, result validation, runner success/failure, profile permissions, reviewer read-only, WAITING + persisted approvals, the four budgets, cancellation, provider errors, communication rules |
| `test_worktrees.py` | create → modify → commit → inspect → remove, isolation, persistence, idempotent retries, base refs, locking and stale locks, dirty/unmanaged removal, secret-commit guard, merges/conflicts, EnterWorktree |
| `test_taskgraph.py` | Persistence of all fields, dependency chains, all-deps rule, failure propagation and recovery, priority, cycles, pause/resume/cancel, concurrent mutation |
| `test_orchestrator.py` | Full graph to REVIEW (main untouched, structured results, commits, measured tests, events), restart, retries with failure context, exhausted retries, manual retry, interrupted recovery, pause/resume, approvals, parallelism, approval-gated merge, measured-vs-claimed tests |
| `test_cli_phase2.py` | `team --simulate`, tasks, task show, agents, agent show, worktrees, review, merge approval, error exit codes |

Results: 404 tests in Phase 1, 504 now (503 passed, 1 skipped), plus 161 subtests; 0 failures, 0 warnings. The single skip is `test_config.py:211` ("POSIX file permission semantics differ on Windows"), which is platform-specific by design.

## Security limitations

- **Pattern-based command analysis.** Shell classification uses regexes, not a shell parser. Obfuscated commands (variables, `eval`, encoded payloads, aliases, scripts written and then executed) can hide capabilities. It is a guard rail, not a sandbox.
- **No OS-level sandbox for shell commands.** File *tools* are confined to the agent's worktree, but a `Bash` command runs with the user's permissions and could touch paths outside it. Agents run on the local machine as the user.
- **Heuristic secret detection.** A secret is caught only if it is registered (project `.env` files, sensitive environment variables) or matches a pattern or sensitive-name heuristic. Secrets in unusual files with generic names can be missed. Masked literals also can't be edited by the model (an `Edit` whose `old_string` contains `<SECRET>` won't match).
- **Registry scope.** The secret registry is per process and in memory. Secrets introduced mid-run in files other than `.env` aren't registered until matched by a pattern.
- **Interrupting in-flight work.** Budgets and cancellation are checked between steps. An in-flight provider call, or a shell command running within its own timeout, isn't interrupted.
- **`--yes` has no per-rule granularity.** It approves every LOW/MEDIUM ask; use deny/ask rules to tighten it.
- **Git commit hooks and signing.** Agent commits use a fixed identity and run the repository's hooks. A GPG-signing setup that needs a passphrase will make commits fail rather than prompt.

## Known limitations

- **No live-model validation.** No provider keys were available. Multi-agent behaviour was verified with the offline simulation provider and scripted fakes; real models may need prompt tuning.
- **Deterministic planning.** Planning uses the template (or a JSON plan). LLM-generated plans are Phase 3.
- **Agents can't ask the human mid-run.** QUESTION and BLOCKED messages are recorded, and blocked tasks surface for a human to resume.
- **Integration merges.** They abort on conflict with no automatic resolution. The integration branch is never merged into main or pushed by Increscent.
- **Single orchestrator per graph.** Parallelism is in-process threads. Worktree locks prevent sharing a worktree, but two concurrent orchestrators on the *same graph* are not coordinated.
- **Model-facing task tools.** `TaskCreate`/`TodoWrite` are still session-scoped; the durable store is the orchestrator's task graph.
- **Windows performance.** Each git call costs about 100–200 ms, so the full test suite takes about 100 s.
- **Scratch-directory fallback.** `EnterWorktree` in a non-git directory uses a scratch directory, not a git worktree; it keeps a Phase 1 test and non-git projects working.
