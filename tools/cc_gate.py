#!/usr/bin/env python3
"""Conventional Commits gate -- hard-fail on any deviation.

This gate enforces the Conventional Commits v1.0.0 specification on:

  1. The PR title.
  2. The title of the issue the PR closes (resolved by the same
     Closes/Fixes/Resolves #N rule the slice gate uses; if no such
     reference exists or the issue does not exist, this check is not
     run because the slice gate already fails the PR).
  3. Every non-merge commit message in the PR's commit range
     (``$BASE..$HEAD``).

Specification reference: https://www.conventionalcommits.org/en/v1.0.0/

Accepted format:

    <type>[optional scope]!?: <description>

where ``<type>`` is one of::

    feat | fix | docs | style | refactor | perf | test | build |
    ci | chore | revert

all lowercase. Optional ``scope`` is parenthesized alphanumeric
(hyphens, underscores, slashes, dots allowed). Optional ``!`` before
the colon marks a breaking change. ``<description>`` must be
non-empty after the ``:`` plus exactly one space.

Exit codes:
  0 -- every checked subject matches CC.
  1 -- at least one deviation.
  2 -- gate setup failure (bad args, git/gh failure, etc.).

Usage:

  python tools/cc_gate.py \\
    --pr-title "<pr title>" \\
    --pr-body-file <path> \\
    --base-ref <rev> \\
    --head-ref <rev> \\
    --repo <owner>/<name>
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path
from typing import Final

# Allowed types per the widely-followed set (Angular convention + CC v1.0.0).
CC_TYPES: Final[frozenset[str]] = frozenset({
    'feat', 'fix', 'docs', 'style', 'refactor', 'perf',
    'test', 'build', 'ci', 'chore', 'revert',
})

# The CC regex. Captures type, scope (without parens), breaking marker.
CC_RE: Final[re.Pattern[str]] = re.compile(
    r'^(?P<type>[a-z]+)'
    r'(?:\((?P<scope>[a-z0-9._/\-]+)\))?'
    r'(?P<breaking>!)?'
    r': (?P<description>.+)$'
)

# Same closing-keyword regex as slice_gate.py; kept local so cc_gate
# does not depend on slice_gate (they gate independent concerns and
# should be independently invocable).
CLOSING_KEYWORD_RE: Final[re.Pattern[str]] = re.compile(
    r'\b(?:close[sd]?|fix(?:e[sd])?|resolve[sd]?)\s+#(\d+)\b',
    re.IGNORECASE,
)


def check_cc(subject: str) -> str | None:
    """Return ``None`` if ``subject`` matches CC; otherwise return a
    human-readable reason for the failure."""
    first_line = subject.split('\n', 1)[0]
    if not first_line:
        return 'empty subject'
    match = CC_RE.match(first_line)
    if match is None:
        return (
            'does not match Conventional Commits v1.0.0 format '
            "'<type>[(scope)][!]: <description>'"
        )
    cc_type = match.group('type')
    if cc_type not in CC_TYPES:
        return (
            f"type {cc_type!r} is not one of the allowed CC types "
            f"(allowed: {sorted(CC_TYPES)!r})"
        )
    description = match.group('description')
    if not description or not description.strip():
        return 'description after the colon is empty or whitespace-only'
    return None


def run_git(args: list[str]) -> str:
    try:
        result = subprocess.run(
            ['git', *args],
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        print(f'cc_gate: git not found: {exc}', file=sys.stderr)
        raise SystemExit(2) from exc
    if result.returncode != 0:
        print(
            f'cc_gate: git {" ".join(args)} failed (exit {result.returncode}): '
            f'{result.stderr.strip()}',
            file=sys.stderr,
        )
        raise SystemExit(2)
    return result.stdout


def list_commits(base_ref: str, head_ref: str) -> list[dict[str, object]]:
    """List every commit in ``base_ref..head_ref`` with metadata for
    deciding whether to check it. ``is_merge`` is derived from the
    parent count (more than one = merge commit)."""
    out = run_git([
        'log', f'{base_ref}..{head_ref}',
        '--format=%H%x09%P%x09%s',
    ])
    commits: list[dict[str, object]] = []
    for line in out.splitlines():
        if not line.strip():
            continue
        parts = line.split('\t', 2)
        if len(parts) != 3:
            continue
        sha, parents, subject = parts
        is_merge = len(parents.split()) > 1
        commits.append({
            'sha': sha,
            'subject': subject,
            'is_merge': is_merge,
        })
    return commits


def fetch_issue_title(repo: str, number: int) -> str:
    """Return the title of the issue.

    Any gh failure here is a gate setup failure because linked-issue
    title validation is part of cc_gate's own contract.
    """
    try:
        result = subprocess.run(
            [
                'gh', 'api', f'repos/{repo}/issues/{number}',
                '--jq', '.title',
            ],
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        print(f'cc_gate: gh not found: {exc}', file=sys.stderr)
        raise SystemExit(2) from exc
    if result.returncode != 0:
        error_output = (
            result.stderr.strip()
            or result.stdout.strip()
            or 'no error output'
        )
        print(
            f'cc_gate: gh api repos/{repo}/issues/{number} failed '
            f'(exit {result.returncode}): {error_output}',
            file=sys.stderr,
        )
        raise SystemExit(2)
    title = result.stdout.strip()
    if not title:
        print(
            f'cc_gate: linked issue #{number} has an empty title payload',
            file=sys.stderr,
        )
        raise SystemExit(2)
    return title


def find_closing_references(body: str) -> list[int]:
    return [int(m.group(1)) for m in CLOSING_KEYWORD_RE.finditer(body)]


def gate(
    pr_title: str,
    pr_body: str,
    base_ref: str,
    head_ref: str,
    repo: str,
) -> list[str]:
    failures: list[str] = []

    # Rule 1: PR title.
    err = check_cc(pr_title)
    if err is not None:
        failures.append(f'PR title {pr_title!r} {err}.')

    # Rule 2: every non-merge commit in range.
    commits = list_commits(base_ref, head_ref)
    for c in commits:
        if c['is_merge']:
            continue
        sha = str(c['sha'])
        subject = str(c['subject'])
        err = check_cc(subject)
        if err is not None:
            failures.append(f'commit {sha[:8]} subject {subject!r} {err}.')

    # Rule 3: linked issue title. Multiple refs or no refs are a
    # slice_gate concern; cc_gate does not duplicate that failure.
    refs = find_closing_references(pr_body)
    if len(refs) == 1:
        issue_title = fetch_issue_title(repo, refs[0])
        err = check_cc(issue_title)
        if err is not None:
            failures.append(
                f'linked issue #{refs[0]} title {issue_title!r} {err}.'
            )

    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description='Conventional Commits gate')
    parser.add_argument('--pr-title', required=True)
    parser.add_argument('--pr-body-file', required=True)
    parser.add_argument('--base-ref', required=True)
    parser.add_argument('--head-ref', required=True)
    parser.add_argument('--repo', required=True)
    args = parser.parse_args()

    try:
        pr_body = Path(args.pr_body_file).read_text(encoding='utf-8')
    except OSError as exc:
        print(
            f'cc_gate: cannot read --pr-body-file {args.pr_body_file}: {exc}',
            file=sys.stderr,
        )
        return 2

    failures = gate(
        args.pr_title,
        pr_body,
        args.base_ref,
        args.head_ref,
        args.repo,
    )

    if failures:
        print('CC GATE -- FAIL')
        print()
        for msg in failures:
            print(f'  - {msg}')
        print()
        print(f'{len(failures)} failure(s). Merge blocked.')
        return 1

    print('CC GATE -- PASS')
    return 0


if __name__ == '__main__':
    sys.exit(main())
