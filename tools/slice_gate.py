#!/usr/bin/env python3
"""Slice gate -- mechanical enforcement of the PR <-> slice-issue contract.

This gate blocks a PR that:

  1.  Does not cite exactly one underlying issue via Closes/Fixes/Resolves #N.
  2.  Cites a number that does not resolve in the repo.
  2a. Cites a number that resolves to a pull request rather than an issue
      (GitHub's issues endpoint also returns PRs).
  3.  Cites an issue that is not in the OPEN state.
  4.  Cites an issue that lacks the ``slice`` label.
  5.  Has a title that does not byte-equal the cited issue's title.
  6.  Cites an issue whose body is missing any full multi-line
      Significance blockquote from the slice template (byte-equal
      substring check).
  7.  Has a diff that touches any file not matched by a glob in the
      cited issue's ``Surfaces`` section.
  8.  Has a diff that touches any file matched by a glob in the cited
      issue's ``Out of Scope`` section.

Rules 6 and 7 together bind the template and the issue: the template's
Significance notes cannot drift away from what the validator demands
(rule 6 reads the template at runtime and asserts every blockquote is
in the issue verbatim), and the scope the issue declares cannot be
exceeded by the PR (rule 7 checks the PR diff against the issue's own
Surfaces globs).

Usage:

  python tools/slice_gate.py \
    --pr-title "<pr title>" \
    --pr-body-file <path> \
    --pr-files-file <path>         # one path per line
    --template <path>              # .github/ISSUE_TEMPLATE/slice.yml
    --repo <owner>/<name>

Exit codes:

  0 -- all gates pass
  1 -- one or more gates failed
  2 -- gate itself could not run (bad args, gh failure, etc.)
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Final

try:
    import yaml  # type: ignore[import-untyped]
except ImportError as _exc:
    print(
        'slice_gate: PyYAML is required (pip install pyyaml)',
        file=sys.stderr,
    )
    raise SystemExit(2) from _exc

# GitHub's own regex for closing keywords.
# https://docs.github.com/en/issues/tracking-your-work-with-issues/linking-a-pull-request-to-an-issue
CLOSING_KEYWORD_RE: Final[re.Pattern[str]] = re.compile(
    r'\b(?:close[sd]?|fix(?:e[sd])?|resolve[sd]?)\s+#(\d+)\b',
    re.IGNORECASE,
)

SURFACES_SECTION_RE: Final[re.Pattern[str]] = re.compile(
    r'##\s+Surfaces\s*\n(.*?)(?=\n##\s|\Z)',
    re.DOTALL,
)

OUT_OF_SCOPE_SECTION_RE: Final[re.Pattern[str]] = re.compile(
    r'##\s+Out of Scope\s*\n(.*?)(?=\n##\s|\Z)',
    re.DOTALL,
)


def extract_significance_blockquotes(template_path: Path) -> list[str]:
    """Extract every full multi-line Significance blockquote from the
    slice template. Each blockquote is returned as a newline-joined
    string with the ``> `` (or ``>``) prefix preserved, exactly as it
    will appear in an issue body filed via the template.

    This runs at gate time so the validator cannot drift from the
    template: if the template's Significance paragraphs change, the
    gate immediately expects the new text in every slice issue body.
    """
    try:
        with template_path.open(encoding='utf-8') as fh:
            template = yaml.safe_load(fh)
    except (OSError, yaml.YAMLError) as exc:
        raise SystemExit(
            f'slice_gate: cannot parse template {template_path}: {exc}'
        ) from exc

    if not isinstance(template, dict):
        raise SystemExit(
            f'slice_gate: template {template_path} is not a YAML mapping'
        )

    blocks: list[str] = []
    body_items = template.get('body', [])
    if not isinstance(body_items, list):
        raise SystemExit(
            f'slice_gate: template {template_path} has no body list'
        )

    for item in body_items:
        if not isinstance(item, dict):
            continue
        if item.get('type') != 'textarea':
            continue
        attrs = item.get('attributes', {})
        if not isinstance(attrs, dict):
            continue
        value = attrs.get('value')
        if not isinstance(value, str):
            continue
        block = _extract_blockquote_from_value(value)
        if block:
            blocks.append(block)

    if not blocks:
        raise SystemExit(
            f'slice_gate: template {template_path} contains no '
            f'Significance blockquotes; cannot run rule 6'
        )
    return blocks


def _extract_blockquote_from_value(value: str) -> str | None:
    """Inside a single textarea's ``value:`` block, pull the contiguous
    blockquote that opens with ``> **Significance.**``. Returns lines
    joined with newlines, verbatim. Returns None if the textarea has
    no such blockquote."""
    collected: list[str] = []
    started = False
    for line in value.split('\n'):
        if not started:
            if line.startswith('> **Significance.**'):
                collected.append(line)
                started = True
        else:
            if line.startswith('>'):
                collected.append(line)
            else:
                break
    return '\n'.join(collected) if collected else None


def _extract_globs_from_section(
    issue_body: str,
    section_pattern: re.Pattern[str],
) -> list[str]:
    """Shared helper: pull every bullet entry from one markdown
    section of the issue body. ``(none)`` is ignored; surrounding
    backticks are stripped."""
    match = section_pattern.search(issue_body)
    if match is None:
        return []
    section = match.group(1)
    globs: list[str] = []
    for raw in section.split('\n'):
        stripped = raw.strip()
        if not stripped.startswith('- '):
            continue
        entry = stripped[2:].strip().strip('`').strip()
        if not entry or entry in {'(none)', 'none'}:
            continue
        globs.append(entry)
    return globs


def extract_surfaces_globs(issue_body: str) -> list[str]:
    """Pull every bullet entry from the ``## Surfaces`` section of the
    issue body. The template's three sub-lists (Modified / Added /
    Removed) are flattened into a single allowed-glob list."""
    return _extract_globs_from_section(issue_body, SURFACES_SECTION_RE)


def extract_out_of_scope_globs(issue_body: str) -> list[str]:
    """Pull every bullet entry from the ``## Out of Scope`` section of
    the issue body. These are the deny-list: any PR file matching one
    of these globs fails the gate, even if the file ALSO matches a
    Surfaces allow-list entry."""
    return _extract_globs_from_section(issue_body, OUT_OF_SCOPE_SECTION_RE)


def read_lines(path: Path) -> list[str]:
    try:
        text = path.read_text(encoding='utf-8')
    except OSError as exc:
        print(
            f'slice_gate: cannot read {path}: {exc}',
            file=sys.stderr,
        )
        raise SystemExit(2) from exc
    return [line for line in text.splitlines() if line.strip()]


def find_closing_references(body: str) -> list[int]:
    """Return the list of issue numbers referenced by closing keywords
    in the PR body. Empty list if none; multi-element if several."""
    return [int(m.group(1)) for m in CLOSING_KEYWORD_RE.finditer(body)]


def fetch_issue(repo: str, number: int) -> dict[str, object] | None:
    """Fetch an issue via gh. Returns None if the issue does not exist.
    Any other gh failure (auth, network, permissions) raises SystemExit(2).

    Implemented with a single subprocess call so a 404 does not emit
    misleading stderr about an unrelated failure path.
    """
    # NOTE: /repos/{repo}/issues/{number} returns pull requests too.
    # Include ``is_pull_request`` in the projection so rule 2a can
    # reject a PR being used as a stand-in for a slice issue.
    try:
        result = subprocess.run(
            [
                'gh', 'api', f'repos/{repo}/issues/{number}',
                '--jq', '{title: .title, state: .state, '
                        'labels: [.labels[].name], body: .body, '
                        'is_pull_request: (.pull_request != null)}',
            ],
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        print(f'slice_gate: gh not found: {exc}', file=sys.stderr)
        raise SystemExit(2) from exc

    if result.returncode == 0:
        return json.loads(result.stdout) if result.stdout.strip() else None

    # A 404 is the legitimate "issue does not exist" case; every other
    # non-zero exit is a setup failure.
    if 'Not Found' in result.stderr or '404' in result.stderr:
        return None

    print(
        f'slice_gate: gh api repos/{repo}/issues/{number} failed '
        f'(exit {result.returncode}): {result.stderr.strip()}',
        file=sys.stderr,
    )
    raise SystemExit(2)


def gate(
    pr_title: str,
    pr_body: str,
    pr_files: list[str],
    template_path: Path,
    repo: str,
) -> list[str]:
    """Run all PR <-> issue checks. Return a list of failure messages
    (empty list means PASS)."""
    failures: list[str] = []

    # Rule 1: exactly one closing reference.
    refs = find_closing_references(pr_body)
    if not refs:
        return [
            'PR body has no closing reference. The PR must include '
            'exactly one line matching `Closes #N` (or Fixes/Resolves) '
            'where N is an OPEN slice-labelled issue.'
        ]
    if len(refs) > 1:
        return [
            f'PR body has {len(refs)} closing references '
            f'({", ".join(f"#{n}" for n in refs)}). The PR must close '
            f'exactly one slice issue.'
        ]
    issue_number = refs[0]

    # Rule 2: issue exists.
    issue = fetch_issue(repo, issue_number)
    if issue is None:
        return [
            f'issue #{issue_number} does not exist in {repo}. The '
            f'closing reference must point at a real OPEN slice issue.'
        ]

    # Rule 2a: the cited number must be an ISSUE, not another PR.
    # GitHub's /repos/:owner/:repo/issues/:num endpoint also returns
    # pull requests; a PR could in principle carry a 'slice' label
    # and a matching body. Reject that before the other rules run.
    if issue.get('is_pull_request'):
        return [
            f'#{issue_number} is a pull request, not an issue. The '
            f'closing reference must point at an OPEN slice issue '
            f'filed via the slice template at '
            f'`.github/ISSUE_TEMPLATE/slice.yml`, not another PR.'
        ]

    # Rule 3: state OPEN.
    state = str(issue.get('state', ''))
    if state.lower() != 'open':
        failures.append(
            f'issue #{issue_number} state is {state!r}; must be OPEN '
            f'for a PR to close it.'
        )

    # Rule 4: labels include 'slice'.
    raw_labels = issue.get('labels', [])
    labels = [str(x) for x in raw_labels] if isinstance(raw_labels, list) else []
    if 'slice' not in labels:
        failures.append(
            f'issue #{issue_number} is missing the "slice" label '
            f'(labels: {labels!r}). The issue must be filed using the '
            f'slice template at `.github/ISSUE_TEMPLATE/slice.yml`.'
        )

    # Rule 5: titles byte-equal.
    issue_title = str(issue.get('title', ''))
    if issue_title != pr_title:
        failures.append(
            f'title mismatch: PR title {pr_title!r} does not equal '
            f'issue #{issue_number} title {issue_title!r}. Titles must '
            f'be byte-identical.'
        )

    # Rule 6: issue body retains every full Significance blockquote from
    # the slice template. Blockquotes are extracted from the template at
    # runtime so the validator and the template cannot drift apart -- if
    # a Significance paragraph changes in the template, the new paragraph
    # must appear verbatim in every new slice issue's body.
    issue_body = str(issue.get('body') or '')
    required_blockquotes = extract_significance_blockquotes(template_path)
    missing_blocks = [b for b in required_blockquotes if b not in issue_body]
    if missing_blocks:
        # Produce a compact failure message: only the first line of each
        # missing blockquote (the heading line) + missing count.
        first_lines = [b.splitlines()[0] for b in missing_blocks]
        failures.append(
            f'issue #{issue_number} body is missing {len(missing_blocks)} '
            f'of {len(required_blockquotes)} full Significance blockquotes '
            f'from the slice template (byte-equal check). Each blockquote '
            f'must appear verbatim in the filed issue. Missing blockquote '
            f'headers: ' + '; '.join(f'{s!r}' for s in first_lines)
        )

    # Rule 7: PR diff scope is within the issue's Surfaces list.
    allowed_globs = extract_surfaces_globs(issue_body)
    if not allowed_globs:
        failures.append(
            f'issue #{issue_number} Surfaces section has no allowed '
            f'path globs. The scope check requires at least one entry '
            f'under Surfaces (in Modified, Added, or Removed).'
        )
    else:
        not_in_surfaces = [
            f for f in pr_files
            if not any(fnmatch.fnmatch(f, g) for g in allowed_globs)
        ]
        if not_in_surfaces:
            failures.append(
                f'PR touches {len(not_in_surfaces)} file(s) not listed in '
                f'issue #{issue_number} Surfaces: '
                + ', '.join(repr(f) for f in not_in_surfaces)
                + f'. Allowed globs: {allowed_globs!r}. '
                  f'Either add the file to the issue\'s Surfaces section '
                  f'(which requires reopening and amending the issue) or '
                  f'remove the change from this PR.'
            )

    # Rule 8: PR diff must not touch Out of Scope paths. This is the
    # deny-list complement to rule 7. A file that matches BOTH a
    # Surfaces glob and an Out of Scope glob fails this rule -- the
    # template's Out of Scope is the finer-grained block.
    denied_globs = extract_out_of_scope_globs(issue_body)
    if denied_globs:
        hits = [
            f for f in pr_files
            if any(fnmatch.fnmatch(f, g) for g in denied_globs)
        ]
        if hits:
            failures.append(
                f'PR touches {len(hits)} file(s) listed in issue '
                f'#{issue_number} Out of Scope: '
                + ', '.join(repr(f) for f in hits)
                + f'. Denied globs: {denied_globs!r}. '
                  f'Remove the change from this PR, or amend the issue '
                  f'to move the path out of the Out of Scope section.'
            )

    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description='Slice gate')
    parser.add_argument(
        '--pr-title',
        required=True,
        help='The PR title, byte-equal to what will match against the issue.',
    )
    parser.add_argument(
        '--pr-body-file',
        required=True,
        help='Path to a UTF-8 file containing the PR body verbatim.',
    )
    parser.add_argument(
        '--pr-files-file',
        required=True,
        help='Path to a UTF-8 file with one changed file path per line '
             '(as produced by `gh pr diff --name-only`).',
    )
    parser.add_argument(
        '--template',
        required=True,
        help='Path to the slice template '
             '(e.g. .github/ISSUE_TEMPLATE/slice.yml). The validator '
             'loads the template at runtime and requires every '
             'Significance blockquote to be present verbatim in the '
             'cited issue body.',
    )
    parser.add_argument(
        '--repo',
        required=True,
        help='GitHub repository in owner/name form (e.g. Vaquum/backtest_simulator).',
    )
    args = parser.parse_args()

    try:
        pr_body = Path(args.pr_body_file).read_text(encoding='utf-8')
    except OSError as exc:
        print(
            f'slice_gate: cannot read --pr-body-file {args.pr_body_file}: {exc}',
            file=sys.stderr,
        )
        return 2

    pr_files = read_lines(Path(args.pr_files_file))

    template_path = Path(args.template)
    if not template_path.is_file():
        print(
            f'slice_gate: --template {template_path} does not exist',
            file=sys.stderr,
        )
        return 2

    failures = gate(
        args.pr_title,
        pr_body,
        pr_files,
        template_path,
        args.repo,
    )

    if failures:
        print('SLICE GATE -- FAIL')
        print()
        for msg in failures:
            print(f'  - {msg}')
        print()
        print(f'{len(failures)} failure(s). Merge blocked.')
        return 1

    print('SLICE GATE -- PASS')
    return 0


if __name__ == '__main__':
    sys.exit(main())
