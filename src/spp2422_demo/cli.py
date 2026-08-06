"""Command line entry point: `spp2422-demo [prepare]`."""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="SPP 2422 TP3 tool wear demo")
    sub = parser.add_subparsers(dest="command")

    train = sub.add_parser("prepare", help="train and cache the models, then exit")
    train.add_argument("--force", action="store_true", help="retrain even if a cache exists")

    serve = sub.add_parser("serve", help="run the dashboard (default)")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8050)
    serve.add_argument("--debug", action="store_true")

    args = parser.parse_args()

    if args.command == "prepare":
        from .artifacts import prepare

        prepare(force=args.force)
        return

    from .app import app

    app.run(
        host=getattr(args, "host", "127.0.0.1"),
        port=getattr(args, "port", 8050),
        debug=getattr(args, "debug", False),
    )


if __name__ == "__main__":
    main()
