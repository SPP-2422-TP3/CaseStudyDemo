"""Headless-Chromium batch driver for spp2422-demo, in the spirit of chromium-cli.

Reads commands one per line from stdin until EOF and executes them in a single
Playwright session. Selectors are passed straight to Playwright, so its native
`text=`, `css=`, `xpath=` prefixes all work.

Usage:
    uv run python .claude/skills/run-spp2422-demo/driver.py <<'EOF'
    nav http://127.0.0.1:8050/
    wait-for text=Tool Wear
    screenshot status
    console --errors
    EOF

Commands:
    nav <url>
    wait-for <selector>              default state: visible
    click <selector>
    fill <selector> <text...>
    press <key>                      sent to the currently focused element
    screenshot [name]
    screenshot-element <selector> [name]
    eval <js-expression>
    sleep <ms>
    console [--errors]               dump captured console messages so far
"""

from __future__ import annotations

import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

SHOTS_DIR = Path("/tmp/spp2422-demo-shots")


def main() -> int:
    SHOTS_DIR.mkdir(parents=True, exist_ok=True)
    console_messages: list[tuple[str, str]] = []
    shot_count = 0

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1280, "height": 800})
        page.on("console", lambda msg: console_messages.append((msg.type, msg.text)))

        for lineno, raw in enumerate(sys.stdin, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(maxsplit=1)
            cmd = parts[0]
            rest = parts[1] if len(parts) > 1 else ""

            try:
                if cmd == "nav":
                    page.goto(rest, wait_until="load")
                elif cmd == "wait-for":
                    page.wait_for_selector(rest, state="visible", timeout=15000)
                elif cmd == "click":
                    page.click(rest, timeout=15000)
                elif cmd == "fill":
                    selector, _, text = rest.partition(" ")
                    page.fill(selector, text, timeout=15000)
                elif cmd == "press":
                    page.keyboard.press(rest)
                elif cmd == "screenshot":
                    shot_count += 1
                    name = rest or f"shot-{shot_count}"
                    out = SHOTS_DIR / f"{name}.png"
                    page.screenshot(path=str(out))
                    print(f"screenshot -> {out}")
                elif cmd == "screenshot-element":
                    selector, _, name = rest.partition(" ")
                    shot_count += 1
                    name = name or f"shot-{shot_count}"
                    out = SHOTS_DIR / f"{name}.png"
                    page.locator(selector).screenshot(path=str(out))
                    print(f"screenshot-element -> {out}")
                elif cmd == "eval":
                    print(page.evaluate(rest))
                elif cmd == "sleep":
                    page.wait_for_timeout(int(rest))
                elif cmd == "console":
                    msgs = console_messages
                    if rest.strip() == "--errors":
                        msgs = [m for m in msgs if m[0] == "error"]
                    for kind, text in msgs:
                        print(f"[{kind}] {text}")
                    if rest.strip() == "--errors" and not msgs:
                        print("no console errors")
                else:
                    print(f"line {lineno}: unknown command {cmd!r}", file=sys.stderr)
                    browser.close()
                    return 1
            except Exception as exc:  # noqa: BLE001
                print(f"line {lineno} ({line!r}) failed: {exc}", file=sys.stderr)
                shot_count += 1
                out = SHOTS_DIR / f"failure-{shot_count}.png"
                page.screenshot(path=str(out))
                print(f"failure screenshot -> {out}", file=sys.stderr)
                browser.close()
                return 1

        browser.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
