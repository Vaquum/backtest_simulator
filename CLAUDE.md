# AGENTS.md

## The laws

Ten laws. Nine are workflow gates on every PR; the tenth is branch protection on `main`. Any failure blocks merge. No bypass.

1. **Every PR closes exactly one OPEN slice-labelled issue.** PR title byte-equals the issue title. Diff stays within the issue's `## Surfaces` globs. Diff touches no path in `## Out of Scope`. Issue body preserves every `> **Significance.**` blockquote from the slice template verbatim. *(pr_checks_slice)*

2. **PR title, every non-merge commit, and the linked issue title match Conventional Commits v1.0.0.** Allowed types: `feat, fix, docs, style, refactor, perf, test, build, ci, chore, revert`. *(pr_checks_cc)*

3. **Typing discipline never weakens.** No new `Any`, `cast(..., Any)`, `# type: ignore`, `# pyright: ignore`, or `# noqa`. Pyright error count cannot rise. `.github/typing_budget.json` cannot be raised by the PR it gates. *(pr_checks_typing)*

4. **Silent-fallback patterns never grow.** No new bare `except:`, empty handler (`pass`, `...`, `return`, `return None`, `continue`, `break`), `contextlib.suppress` (or any alias chain thereof), or `errors='ignore'`. `.github/fail_loud_budget.json` cannot be raised by the PR it gates. *(pr_checks_fail_loud)*

5. **Every PR bumps the version and leaves a CHANGELOG trail.** `[project].version` advances strictly forward by `MAJOR.MINOR.PATCH`. `CHANGELOG.md`'s first `# v<X.Y.Z>` header equals the new version and carries at least one content line. Bump level meets the Conventional Commits type minimum: `type!` → major, `feat` → minor, else patch. *(pr_checks_version)*

6. **Ruff 0.15.11 is clean on `tools/` and `tests/tools/`.** *(pr_checks_lint)*

7. **`pytest tests/origo_source_native` passes.** *(pr_checks_tests)*

8. **CodeQL reports no new Python security anti-patterns.** *(PR Checks CodeQL)*

9. **Live branch protection on `main` matches `.github/rulesets/main.json`.** Changing branch protection out-of-band (in the GitHub UI) blocks the next PR until the snapshot is updated in a PR of its own. *(pr_checks_ruleset)*

10. **No direct push to `main`. No force-push. No branch deletion.** Branch must be up-to-date with `main` before merge. One Copilot review required; all review threads resolved. *(branch protection, server-side)*

## Workflow

Branch off `main`. Push to a remote branch of the same name. **Open the PR the moment the change is ready for CI to run on it — not when it feels finished.** The gates run on GitHub while you keep working locally. Don't wait for CI in the foreground.

**`zero-bang` is the approving authority.** Request their review the moment the PR is open. Once every requested change is addressed, re-request `zero-bang`'s review.

Each push re-runs every gate. Prefer new commits to amends — amends don't give you anything and they muddle the PR history.

Merge unlocks when every required gate is green **and** the branch is up-to-date with `main`. Up-to-date is enforced server-side; rebase when main advances.

When a gate fails, the gate's own output names the reason. Read the output, fix the code or the slice issue, push again. If the failure is the gate being wrong rather than the PR being wrong, fix the gate in its own PR — the ruleset drift gate (law 9) will force the matching ruleset-snapshot update so no gate relaxation side-enters.

## Review work

When reviewing a PR, post each comment inline as a review comment. Do not confirm with the operator first.

When reviewing an issue, post comments as a new comment in the thread. Do not confirm with the operator first.

When reviewing comments left on an issue you are working on, always address the issues by editing the original body or by adding a new comment that clearly explains why something is not addressed. Do not confirm with the operator first

The opinion is the deliverable. Confirming before posting adds a round-trip and slows collaboration.

## Beyond the laws

The gates check shape, scope, format, ratchets, and named test suites. They do not check whether the slice's capability actually works. The operator judges that at review time, against the following stance:

**Radical simplicity.** The simplest code that meets the requirement wins. Complexity earns its place by naming the specific concern it addresses — not "robustness" or "future-proofing" in general.

**No defensive fog.** Agents are primed to produce defensible-looking code: `try/except` that swallows everything, fallbacks for cases that don't happen, docstrings that restate the signature, comments that narrate the line, parameters that might be useful someday. None of it belongs here. The fail-loud gate (law 4) catches the AST-detectable forms mechanically; the rest is operator-caught at review.

**No sitting in the dark.** Do not block callable outputs or script outputs in anyway to save context. Or do not hide sub-agent logs to save your context. Always stay fully aware of what the running thing is doing. 

**No long-running commands.** Do not repeatedly run long-running commands. If something needs to be run repeatedly, immediately profile it, and report back to operator the profile results. The operator is expert in performance hacking, you are not. 

**Minimal scope.** Touch only the files the task demands. Drive-by cleanups go in a separate slice.

**No synthetic data.** Ever. Inventing data is not a shortcut — it corrupts everything downstream. If the real data isn't there, stop and ask.

**Validate against the stated expectation.** The question is never "did it run" — it's "did it return what the slice promised."

**Deliver meaning not mechanics.** It's better to deliver the right meaning poorly, than deliver meaningless scaffolding and mechanics in an impressive way. 

**The smallest possible honest way always.** Slice spec, code, communication, everything, let it be the smallest possible unit size that honestly delivers what is required.

## When in doubt, stop

This is collaboration. If the requirement is unclear, if the scope is ambiguous, if a gate's meaning is unobvious, if the fix would require touching something that wasn't asked for — stop and ask the operator. Proceeding through doubt is where harm accumulates.

## Task completion

Always work with codex on command line until codex approves your work. 

Only use this command: `codex exec --skip-git-repo-check -c model_reasoning_effort=xhigh {content} 2>&1`

Never use tail or other things that block codex output. Let it all be fully exposed to you. Do not save the context.

If you made a commit, post the `hash`.
If you made a PR, post the link.

## The Five Principles

- no duck-taping or saga ornamentation
- "bts or it didn't happen" -> everything is bts cli first
- not building a parallel universe around praxis/nexus ->  extending them
- honest to how market actuall works -> works just like nexus/praxis paper-trade
- strategy tested, is strategy deployed -> limen leads to nexus leads to praxis