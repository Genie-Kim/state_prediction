# Agent model guidelines

When spawning ad-hoc subagents (via the Agent tool or Workflow `agent()` calls)
for routine or mechanical work — file search, simple edits, running tests,
summarizing output — default to a lighter model (`sonnet` or `haiku`) instead
of inheriting the orchestrator's model.

Reserve the orchestrator's own model for planning, synthesis, and tasks that
need deep reasoning.
