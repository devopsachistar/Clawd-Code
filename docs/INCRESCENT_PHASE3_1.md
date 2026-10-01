# Phase 3.1: AI planning engine

The first AI capability: a natural-language requirement becomes a validated plan, and the plan becomes the existing Phase 2 task graph. This milestone adds no coding agents, deployment, model routing or security changes.

```
Requirement
    ↓  Planner.build_request()   provider-neutral (build_provider_request, same path as the agent loop)
LLM (any configured provider)
    ↓  JSON
Planner: extract_json() + parse_plan()   → ImplementationPlan (typed, fully validated)
    ↓  ImplementationPlan.to_task_specs()  → existing TaskSpec
Orchestrator.create_graph()  → existing TaskStore (its own dependency/cycle checks stay authoritative)
```

## Components

The planner lives in `src/orchestration/ai_planner.py`. It contains no provider-specific code.

**Plan schema:**
- `project {name, summary}`
- `requirements[]`
- `assumptions[]`
- `architecture {frontend, backend, database, infrastructure}`
- `modules[] {name, description}`
- `tasks[] {id, title, description, dependencies[], role, acceptance_criteria[]}`

**Validation:** `parse_plan` rejects the model output and lists every problem it finds when:
- a required field is missing or has the wrong type
- a task title is empty
- a task ID is not a lowercase identifier, or is duplicated
- a dependency is unknown or points at its own task
- the dependencies form a cycle
- a task's role isn't an agent role from the configurable `ProfileRegistry`

  Valid roles are architect, database, backend, frontend, developer, qa, reviewer, security and devops, or any profile name.

**Failure handling:** provider errors and timeouts, empty responses, invalid JSON, invalid plans and truncated output (`finish_reason=length`) all raise `PlanningError` with a clear message. The planner never fabricates output and never retries. An invalid plan never reaches the task graph.

**Task graph integration:** `plan_to_task_graph(plan, orchestrator, requirement)` maps roles to agent profiles and converts tasks to `TaskSpec`. It then calls the existing `Orchestrator.create_graph()`: the requirement becomes the root project task, dependencies are wired, and READY/PENDING states are resolved by `TaskStore`. The full plan is also stored in `graph.settings["plan"]`.

**Offline mode:** `SimulatedPlannerProvider` (in `src/orchestration/simulation.py`) is a deterministic planning model that returns `SIMULATED_PLAN`. No API keys are needed.

## CLI

```bash
increscent plan --simulate "Build a simple task management application with users, projects, tasks, priorities and a dashboard."
increscent plan "Build a CRM for a small agency"            # uses the configured provider (needs an API key)
increscent plan --no-save "..."                             # print only, no task graph
increscent plan --json --simulate "..."                     # validated plan + graph id as JSON
increscent tasks --graph G-xxxxxxxx                         # inspect the created graph
increscent team --resume G-xxxxxxxx                         # (Phase 2) run the planned graph with agents
```

Saving a graph needs a git repository with at least one commit.

**Exit codes:**

| Code | Meaning |
|---|---|
| 0 | Plan created |
| 1 | Planning failed (provider error or invalid model output) |
| 2 | Configuration error (no requirement, no API key, not a git repository) |

## Limitations

- The plan shape is fixed, with one prompt and a single model call. There is no plan refinement or repair loop.
- Plan quality depends on the configured model. Only the offline simulated planner has been exercised, because no real API credentials were available.
- Planning creates the graph only; running it is still `increscent team --resume`. Graph runs can't yet be driven directly from the planner's role choices beyond profile mapping.
