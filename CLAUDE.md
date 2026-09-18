# voice_core — Project-Specific Instructions

Shared rules are in the global `~/.claude/CLAUDE.md`. This file contains only
voice_core-specific overrides.

**Keep this file short.** No redundancy with the global CLAUDE.md. One bullet
per rule. If editing this file, remove or consolidate — never just append.

## Running tests

This repo has no venv of its own. Use a consumer's, and point `PYTHONPATH` at
the checkout under test so the suite reads your worktree and not an installed copy:

```bash
PYTHONPATH="<checkout>" "C:/path/to/fun_time/.venv/Scripts/python.exe" -m pytest "<checkout>/tests"
```

## What belongs here

- **Hearing, and nothing an app does with what was heard.** The microphone, the
  recognizers, the rules that accept or refuse a reading, the clips kept of what
  was missed. What a phrase means, what is shown or dispatched for it, and when
  an app listens are that app's.
- **No app knows another app exists**, and nothing here is shaped around one
  caller: a list of phrases and a few callables are the whole of what an app
  hands over.
- **The shells are not unit-tested, on purpose:** what opens the sound device
  and what loads a model need hardware and hundreds of megabytes. Logic lives
  outside them and is tested against fakes.

## Judging a recognizer change

Only on recordings of the owner's own voice, never on synthesized speech, which
reversed the ranking of the engines when it was tried: Fun Time keeps a clip of
every command it failed to understand under its `state/voice_misses`, and his
ordinary dictation, cut at its pauses, is the test for commands invented out of
talk. Both are private: run them locally, and never quote what was said in a
test, a commit or a reply.

## Changing this repo changes three apps

- A change here reaches an app when that app moves its pin, not before: each
  names a tag of this repo in its `[project.dependencies]`. Landing is two
  commits — this repo's, which tags a version, then the app's, which takes it.
- The apps' venvs are live: `~/.claude/engineering.md` (User Environment) says
  when and how a new tag may be installed into them.

## Test fixtures must be fabricated, never copied from the real library

Every fixture value that stands in for library data — a spoken phrase built from
the owner's content file, a title, a name — must be **invented**. The apps'
real command lists carry words from a private content file: never paste one
into a test, a commit message or a reply. Use `alpha`/`beta`/`gamma` and plain
made-up commands. `app_support.sanitize` fails the suite on a **known** blocked
term only; it cannot see the next one.

## Landing — GitHub merge queue, not local ff-merge

This repo is public at `github.com/haglio/voice_core` with a merge-queue ruleset
on `main`, so the global "ff-merge into the primary checkout under
`.git/agent-merge.lock`" flow does NOT apply here:

- **Land through a pull request.** From your worktree: commit, `git fetch origin
  && git rebase origin/main`, `git push -u origin <branch>`, then
  `gh pr create --fill`. Auto-merge arms itself; the queue rebases your PR onto
  `main`, runs the required check, and merges it when green. Don't ff-merge into
  the primary checkout, don't push `main` directly, and never force-push `main`.
- **Sync local checkouts by pulling.** `main` advances only on origin (via the
  queue), so the primary checkout and worktrees update with
  `git pull --ff-only origin main`. The primary is only ever fast-forwarded.
- **A red required check** (`.github/workflows/merge-gate.yml`) can't land.
