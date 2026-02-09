"""
LabOS CLI — command-line entry point.

Usage:
  labos run              Interactive CLI chat
  labos web              Launch Gradio web UI
  labos --help           Show usage
"""

import argparse
import sys


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="labos",
        description="LabOS — Self-evolving multi-agent framework for biomedical research",
    )
    parser.add_argument("--version", action="store_true", help="Show version and exit")

    sub = parser.add_subparsers(dest="command")

    # --- labos run ---
    run_p = sub.add_parser("run", help="Interactive CLI chat with the LabOS agent")
    run_p.add_argument("--use-template", action="store_true", default=True,
                       help="Enable knowledge-base templates (default: True)")
    run_p.add_argument("--no-template", action="store_true", help="Disable templates")
    run_p.add_argument("--use-mem0", action="store_true", help="Enable Mem0 enhanced memory")
    run_p.add_argument("--model", type=str, default="google/gemini-3",
                       help="OpenRouter model ID (default: google/gemini-3)")

    # --- labos web ---
    web_p = sub.add_parser("web", help="Launch Gradio web interface")
    web_p.add_argument("--port", type=int, default=7860, help="Server port (default: 7860)")
    web_p.add_argument("--share", action="store_true", help="Create public Gradio link")
    web_p.add_argument("--use-template", action="store_true", default=True,
                       help="Enable templates (default: True)")
    web_p.add_argument("--no-template", action="store_true", help="Disable templates")
    web_p.add_argument("--use-mem0", action="store_true", help="Enable Mem0 enhanced memory")
    web_p.add_argument("--model", type=str, default="google/gemini-3",
                       help="OpenRouter model ID (default: google/gemini-3)")

    return parser


def _cmd_run(args):
    """Interactive CLI chat loop."""
    from labos.core import initialize, run_task

    use_tpl = args.use_template and not args.no_template
    agent = initialize(
        use_template=use_tpl,
        use_mem0=args.use_mem0,
        model_id=args.model,
    )
    if agent is None:
        print("Failed to initialise LabOS. Check your OPENROUTER_API_KEY.")
        sys.exit(1)

    print("\nLabOS interactive mode. Type 'exit' or 'quit' to stop.\n")
    while True:
        try:
            query = input("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
        if not query:
            continue
        if query.lower() in ("exit", "quit", "q"):
            print("Goodbye!")
            break
        try:
            response = run_task(query)
            print(f"\nLabOS> {response}\n")
        except Exception as exc:
            print(f"\nError: {exc}\n")


def _cmd_web(args):
    """Launch Gradio web UI."""
    from labos.ui import main as ui_main

    use_tpl = args.use_template and not args.no_template
    ui_main(
        use_template=use_tpl,
        use_mem0=args.use_mem0,
        port=args.port,
        share=args.share,
    )


def main():
    """CLI entry point (called by ``labos`` command)."""
    parser = _build_parser()
    args = parser.parse_args()

    if args.version:
        from labos import __version__
        print(f"labos {__version__}")
        return

    if args.command == "run":
        _cmd_run(args)
    elif args.command == "web":
        _cmd_web(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
