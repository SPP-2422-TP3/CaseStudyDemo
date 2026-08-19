---
name: run-spp2422-demo
description: Build, run, and drive the spp2422-demo Dash dashboard — start it in the background on localhost:8050 (per CLAUDE.md, keep it always running there), restart it after code changes, take screenshots, click through its pages, and run its test suite. Use when asked to start/run/build/test spp2422-demo, screenshot the dashboard, or verify a change works in the running app.
---

Single-app repo, driven headlessly with Playwright (Python, `uv`-managed — no
`chromium-cli`/Node/tmux in this environment). All paths below are relative to the
repo root.

## Prerequisites / Setup

```bash
uv sync   # installs the app + dev deps, including playwright, into .venv
```

Playwright's Chromium is a dev dependency (added via `uv add --dev playwright`) so
`uv sync` alone provisions it — no `apt-get`, no separate browser install needed in
this container (Chromium was already cached at `~/.cache/ms-playwright/`; `uv run
playwright install chromium` found it and skipped the download — if that cache is
cold on a different machine, run that command once after `uv sync`).

Trained models are committed in `data/models/` (`deep_drawing.pkl`, `ironing.pkl`),
so the first start is instant — no training step. `uv run spp2422-demo prepare
--force` only if you need to force-retrain the cache.

## Run in background (agent path — CLAUDE.md requires this always be up on :8050)

**Check before starting anything** — a server satisfying CLAUDE.md's "always runs on
localhost:8050" rule may already be up:

```bash
curl -fsS -o /dev/null -w "%{http_code}\n" http://127.0.0.1:8050/ --max-time 3
```

If that prints `200`, it's already serving — reuse it, don't touch it.

**If nothing answers**, start it detached and poll until ready (~5s cold start):

```bash
nohup uv run spp2422-demo serve --port 8050 >/tmp/spp2422-demo.log 2>&1 &
disown
for i in $(seq 30); do
  curl -fsS -o /dev/null http://127.0.0.1:8050/ && echo "ready after ${i}s" && break
  sleep 1
done
```

**After editing source**, the running server does *not* hot-reload (it's started
without `--debug`). To make `:8050` show the change (required by CLAUDE.md — "shows
the most recent state"):

```bash
lsof -ti:8050 -sTCP:LISTEN | xargs -r kill
sleep 1
nohup uv run spp2422-demo serve --port 8050 >/tmp/spp2422-demo.log 2>&1 &
disown
timeout 30 bash -c 'until curl -fsS -o /dev/null http://127.0.0.1:8050/; do sleep 1; done'
```

Only kill the port if you're the one who needs to restart it — don't tear it down
otherwise; CLAUDE.md wants it always up.

## Drive it — `driver.py`

`.claude/skills/run-spp2422-demo/driver.py` is a small `chromium-cli`-style batch
driver: it reads commands one per line from stdin, runs them in one headless
Playwright session, and exits. Invoke it through `uv run` so it uses the project's
own environment:

```bash
uv run python .claude/skills/run-spp2422-demo/driver.py <<'EOF'
nav http://127.0.0.1:8050/
wait-for css=.card-title:has-text("Tool Wear")
screenshot status
click css=.card-title:has-text("Tool Wear")
wait-for css=.modal-title:has-text("Tool Wear")
screenshot wear-detail
press Escape
wait-for css=a[title="About & Help"]
click css=a[title="About & Help"]
wait-for text=Tool Wear from Forming Force Signals
screenshot details
nav http://127.0.0.1:8050/wear-threshold
wait-for text=PLACED AT
screenshot wear-threshold
console --errors
EOF
```

This exact script was run this session: it produced 4 screenshots and reported "no
console errors." Screenshots land in `/tmp/spp2422-demo-shots/<name>.png`.

| command | what it does |
|---|---|
| `nav <url>` | navigate |
| `wait-for <selector>` | wait until visible (Playwright selector: `text=`, `css=`, `xpath=` all work) |
| `click <selector>` | click |
| `fill <selector> <text...>` | type into an input |
| `press <key>` | send a key to the focused element |
| `screenshot [name]` | full-page screenshot → `/tmp/spp2422-demo-shots/<name>.png` |
| `screenshot-element <selector> [name]` | crop to one element |
| `eval <js>` | run JS in the page, print the result |
| `sleep <ms>` | raw wait — last resort, prefer `wait-for` |
| `console [--errors]` | dump captured console messages (optionally errors only) |

Failures print to stderr with the failing line, and the driver saves a
`failure-N.png` before exiting non-zero — always worth checking when a script fails
partway through.

## The app's 3 pages

Dash multi-page app (`use_pages=True`); all reachable by direct URL, no query params
needed:

- `/` — **Status**. Top bar/nav are hidden by design (it's meant to fill a press-side
  screen) — the only way back to other pages from here is the ⓘ icon
  (`a[title="About & Help"]`), not a nav link.
- `/details` — About & Help. Full top nav present here (`Status`, `About & Help`,
  `Wear Threshold`), all direct links.
- `/wear-threshold` — a direct nav link like the other two pages. Its stat
  cards/chart are filled by a callback that fires *after* the heading renders —
  see Gotchas.

## Run (human path)

```bash
uv run spp2422-demo            # foreground, http://127.0.0.1:8050, Ctrl-C to stop
```

## Test

```bash
uv run ruff check .   # All checks passed!
uv run pytest -q      # 51 passed
```

Both ran clean this session. No browser-driven tests exist in `tests/` — those two
files are in-process unit tests; `driver.py` above is the only browser-driven check.

---

## Gotchas

- **Ambiguous `text=` selectors silently wait on the wrong (hidden) element.**
  `wait-for text=Tool Wear` matched the topbar's `.brand-sub` ("Tool Wear from
  Forming Force Signals") *and* the card title, in that DOM order — and since the
  topbar is hidden on `/`, Playwright polls forever waiting for the first match to
  become visible and times out at 15s, even though the card you actually want is
  right there. Scope selectors to something unique, e.g.
  `css=.card-title:has-text("Tool Wear")`.
- **Status-card click targets use Dash pattern-matching ids**
  (`{"type": "status-card", "card": "wear"}`), rendered as a literal-dict-string
  `id` attribute — don't try to match that by CSS; click by the visible
  `.card-title` text instead (works fine, the click bubbles to the wrapping div's
  handler).
- **`/wear-threshold`'s cards/chart render after the heading.** `wait-for text=
  Locating the Wear Threshold` resolves as soon as the `<h1>` paints, but the
  stat cards (`A2 PLACED AT`, etc.) and chart below are filled in by a follow-up
  callback and are still blank at that point — screenshot too early and you get a
  half-empty page. Wait for `text=PLACED AT` (or another element from the
  callback-filled region) instead of the heading.
- **`playwright install chromium` doesn't always mean "download."** In this
  container it found the existing `~/.cache/ms-playwright/chromium-1234` (same
  version as the system Python's separately-installed Playwright) and completed
  instantly with no output — the OS-level cache path isn't tied to which venv
  installed the `playwright` pip package.
