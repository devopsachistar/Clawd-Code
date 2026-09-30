"""CLI entry point for Clawd Codex."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rich.console import Console
from rich.prompt import Prompt
from rich.table import Table


_COMMON_FLAGS = (
    (("--provider",), {"help": "LLM provider (overrides env/project/user config)"}),
    (("--model",), {"help": "Model name (overrides env/project/user config)"}),
    (("--base-url",), {"dest": "base_url", "help": "Provider API base URL override"}),
    (("--log-file",), {"dest": "log_file", "help": "Write structured JSON-lines logs to this file (or set CLAWD_LOG_FILE)"}),
    (("--verbose",), {"action": "store_true", "help": "Log runtime events to stderr"}),
)


def _add_common_flags(parser: argparse.ArgumentParser, *, suppress: bool) -> None:
    for names, kwargs in _COMMON_FLAGS:
        kw = dict(kwargs)
        if suppress:
            kw["default"] = argparse.SUPPRESS
        parser.add_argument(*names, **kw)


def _prog_name() -> str:
    from src.branding import CLI_NAMES

    stem = Path(sys.argv[0]).stem.lower() if sys.argv and sys.argv[0] else ""
    return stem if stem in CLI_NAMES else CLI_NAMES[0]


def _configure_streams() -> None:
    """Never crash on characters the console encoding cannot represent."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")  # type: ignore[union-attr]
        except Exception:
            pass


def _configure_logging_from_args(args: argparse.Namespace) -> None:
    from src.branding import env_lookup
    from src.observability import configure_logging

    log_file = getattr(args, "log_file", None) or env_lookup("LOG_FILE")[0]
    verbose = bool(getattr(args, "verbose", False))
    if log_file or verbose:
        configure_logging(log_file=log_file, verbose=verbose)


def build_parser() -> argparse.ArgumentParser:
    prog = _prog_name()
    parser = argparse.ArgumentParser(
        prog=prog,
        description="Clawd Codex - Claude Code Python Implementation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Examples:
  {prog}                          Start interactive REPL
  {prog} --stream                 Start REPL with live response rendering
  {prog} run "Fix the failing tests"      Run one task non-interactively (scripts/CI)
  {prog} run --output json --yes "..."    JSON result, auto-approve risky actions
  {prog} sessions                 List saved sessions for this project
  {prog} login                    Configure API keys
  {prog} config                   Show effective configuration (secrets masked)
  {prog} --version                Show version

Configuration precedence: CLI flag > environment variable > project
(.clawd/settings.json) > user (~/.clawd/config.json) > default.
API keys: ANTHROPIC_API_KEY, OPENAI_API_KEY, ZHIPUAI_API_KEY, MINIMAX_API_KEY
(or 'login'). Provider/model: INCRESCENT_PROVIDER / CLAWD_PROVIDER, *_MODEL.

Exit codes (run): 0 success, 1 task failure, 2 configuration error,
3 permission/security rejection.
"""
    )

    parser.add_argument(
        '--version',
        action='store_true',
        help='Show version information'
    )
    parser.add_argument(
        '--config',
        action='store_true',
        help='Show current configuration'
    )
    parser.add_argument(
        '--stream',
        action='store_true',
        help='Enable live response rendering in the REPL'
    )
    _add_common_flags(parser, suppress=False)

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # login subcommand
    subparsers.add_parser('login', help='Configure API keys')

    # config subcommand
    config_parser = subparsers.add_parser('config', help='Show current configuration')
    _add_common_flags(config_parser, suppress=True)

    # run subcommand (non-interactive)
    run_parser = subparsers.add_parser(
        'run',
        help='Run a task non-interactively (scripts/CI)',
        description='Run one agent task without the interactive REPL. '
                    'Exit codes: 0 success, 1 task failure, 2 configuration error, 3 permission/security rejection.',
    )
    run_parser.add_argument('prompt', nargs='*', help='Task to perform ("-" or omitted: read from stdin)')
    run_parser.add_argument('--max-turns', type=int, default=30, help='Maximum model turns (default: 30)')
    run_parser.add_argument('--output', choices=['text', 'json'], default='text', help='Output format (default: text)')
    run_parser.add_argument('--stream', action='store_true', help='Stream the response text as it is generated')
    run_parser.add_argument('-y', '--yes', action='store_true',
                            help='Auto-approve actions that need permission (blocked commands are still refused)')
    run_parser.add_argument('--resume', metavar='SESSION_ID', help='Continue a saved session ("latest" for the most recent in this project)')
    run_parser.add_argument('--cwd', help='Project directory to operate in (default: current directory)')
    run_parser.add_argument('--no-save', action='store_true', help='Do not save the session')
    run_parser.add_argument('--allow-unsafe-root', action='store_true',
                            help='Allow running with the home directory or a filesystem root as the project root')
    run_parser.add_argument('-q', '--quiet', action='store_true', help='Only print the final result')
    _add_common_flags(run_parser, suppress=True)

    # sessions subcommand
    sessions_parser = subparsers.add_parser('sessions', help='List saved sessions for this project')
    sessions_parser.add_argument('--all', action='store_true', help='List sessions from every project')

    return parser

def main(argv: list[str] | None = None):
    """CLI main entry point."""
    raw_args = sys.argv[1:] if argv is None else argv
    # Quick path for --version
    if len(raw_args) == 1 and raw_args[0] in ['--version', '-v', '-V']:
        from src import __version__
        print(f"clawd-codex version {__version__} (Python)")
        return 0

    _configure_streams()
    parser = build_parser()
    args = parser.parse_args(raw_args)

    # Handle --version
    if args.version:
        from src import __version__
        print(f"clawd-codex version {__version__} (Python)")
        return 0

    _configure_logging_from_args(args)

    # Handle --config
    if args.config:
        return show_config(args)

    # Handle commands
    if args.command == 'login':
        return handle_login()
    elif args.command == 'config':
        return show_config(args)
    elif args.command == 'run':
        return handle_run(args)
    elif args.command == 'sessions':
        return handle_sessions(args)

    # Default: start REPL
    return start_repl(
        stream=args.stream,
        provider=getattr(args, "provider", None),
        model=getattr(args, "model", None),
        base_url=getattr(args, "base_url", None),
    )

def handle_run(args: argparse.Namespace) -> int:
    """Run a single task non-interactively."""
    from src.headless import RunOptions, run_headless

    prompt = " ".join(args.prompt).strip()
    if prompt in ("", "-"):
        if sys.stdin is not None and not sys.stdin.isatty():
            prompt = sys.stdin.read().strip()
        else:
            prompt = ""

    options = RunOptions(
        prompt=prompt,
        provider=getattr(args, "provider", None),
        model=getattr(args, "model", None),
        base_url=getattr(args, "base_url", None),
        cwd=args.cwd,
        max_turns=args.max_turns,
        output=args.output,
        stream=args.stream,
        auto_approve=args.yes,
        resume=args.resume,
        save_session=not args.no_save,
        allow_unsafe_root=args.allow_unsafe_root,
        quiet=args.quiet,
    )
    return int(run_headless(options).exit_code)

def handle_sessions(args: argparse.Namespace) -> int:
    """List saved sessions."""
    from src.agent.session import list_sessions
    from src.project import detect_project

    console = Console()
    project_root = None if args.all else detect_project().project_root
    sessions = list_sessions(project_root=project_root)
    if not sessions:
        scope = "any project" if args.all else str(project_root)
        console.print(f"No saved sessions for {scope}.")
        return 0
    table = Table(title="Saved sessions", show_header=True, header_style="bold")
    table.add_column("Session", style="cyan")
    table.add_column("Updated")
    table.add_column("Provider / model", style="magenta")
    table.add_column("Messages", justify="right")
    if args.all:
        table.add_column("Project", style="green")
    for item in sessions:
        row = [
            str(item["session_id"]),
            str(item.get("updated_at", ""))[:19],
            f"{item.get('provider')} / {item.get('model')}",
            str(item.get("messages", 0)),
        ]
        if args.all:
            row.append(str(item.get("project_root") or ""))
        table.add_row(*row)
    console.print(table)
    console.print("[dim]Resume with: run --resume <session> \"...\"  or /load <session> in the REPL[/dim]")
    return 0

def _show_provider_defaults_table() -> None:
    """Print a table showing available providers and their defaults."""
    from src.providers import PROVIDER_INFO

    console = Console()
    table = Table(title="Available Providers & Defaults", show_header=True, header_style="bold")
    table.add_column("Provider", style="cyan")
    table.add_column("Default Model", style="magenta")
    table.add_column("Base URL", style="green")

    for name, info in PROVIDER_INFO.items():
        table.add_row(
            f"{name} ({info['label']})",
            info["default_model"],
            info["default_base_url"],
        )

    console.print(table)
    console.print()

def handle_login():
    """Interactive API configuration."""
    console = Console()
    console.print("\n[bold blue]Clawd Codex - API Configuration[/bold blue]\n")

    # Show available providers and their defaults
    _show_provider_defaults_table()

    # Select provider
    from src.providers import PROVIDER_INFO
    provider_names = list(PROVIDER_INFO.keys())

    provider = Prompt.ask(
        "Select LLM provider",
        choices=provider_names,
        default="anthropic"
    )

    info = PROVIDER_INFO[provider]

    # Input API Key
    api_key = Prompt.ask(
        f"Enter {provider.upper()} API Key",
        password=True
    )

    if not api_key:
        console.print("\n[red]Error: API Key cannot be empty[/red]")
        return 1

    # Optional: Base URL (show default)
    console.print(f"\n[dim]Default:[/dim] {info['default_base_url']}")
    base_url = Prompt.ask(
        f"{provider.upper()} Base URL",
        default=info["default_base_url"]
    )

    # Optional: Default Model (show available options)
    console.print(f"\n[dim]Available models:[/dim] {', '.join(info['available_models'])}")
    console.print(f"[dim]Default:[/dim] [bold]{info['default_model']}[/bold]")
    default_model = Prompt.ask(
        f"{provider.upper()} Default Model",
        default=info["default_model"]
    )

    # Save configuration
    from src.config import set_api_key, set_default_provider

    set_api_key(provider, api_key=api_key, base_url=base_url, default_model=default_model)
    set_default_provider(provider)

    console.print(f"\n[green]✓ {provider.upper()} API Key saved successfully![/green]")
    console.print(f"[green]✓ Default provider set to: {provider}[/green]\n")
    return 0

def show_config(args: argparse.Namespace | None = None):
    """Show current configuration (API keys are never printed)."""
    console = Console()

    try:
        from src.config import ConfigError, get_config_path, load_config, resolve_provider_config
        from src.project import detect_project

        config_path = get_config_path()
        config = load_config(create_if_missing=False) if config_path.exists() else {}

        console.print(f"\n[bold]Configuration File:[/bold] {config_path}"
                      + ("" if config_path.exists() else " [dim](not created yet)[/dim]") + "\n")

        project = detect_project()
        console.print("[bold]Project:[/bold]")
        console.print(f"  Project root: {project.project_root}")
        console.print(f"  Git root: {project.git_root or '(not a git repository)'}")
        if project.is_unsafe_root:
            console.print("  [yellow]Warning: project root is your home directory or a filesystem root[/yellow]")

        console.print("\n[bold]Effective provider configuration:[/bold]")
        try:
            resolved = resolve_provider_config(
                cli_provider=getattr(args, "provider", None),
                cli_model=getattr(args, "model", None),
                cli_base_url=getattr(args, "base_url", None),
                project_root=project.project_root,
            )
            info = resolved.describe()
            for key in ("provider", "model", "base_url", "api_key"):
                console.print(f"  {key}: {info[key]} [dim]({info['sources'].get(key, '-')})[/dim]")
            for warning in resolved.warnings:
                console.print(f"  [yellow]warning: {warning}[/yellow]")
        except ConfigError as e:
            console.print(f"  [red]{e}[/red]")
            return 2

        console.print("\n[bold]Current Configuration:[/bold]\n")

        # Show default provider
        console.print(f"[cyan]Default Provider:[/cyan] {config.get('default_provider', 'Not set')}")

        # Show providers (without showing full API keys)
        console.print("\n[cyan]Configured Providers:[/cyan]")
        for provider_name, provider_config in config.get("providers", {}).items():
            api_key = provider_config.get("api_key", "")
            masked_key = f"{api_key[:4]}...{api_key[-4:]}" if len(api_key) > 12 else ("set" if api_key else "Not set")

            console.print(f"\n  [yellow]{provider_name.upper()}:[/yellow]")
            console.print(f"    API Key: {masked_key}")
            console.print(f"    Base URL: {provider_config.get('base_url', 'Not set')}")
            console.print(f"    Default Model: {provider_config.get('default_model', 'Not set')}")

        console.print()

    except Exception as e:
        console.print(f"\n[red]Error loading configuration: {e}[/red]\n")
        return 1

    return 0

def start_repl(stream: bool = False, provider: str | None = None, model: str | None = None,
               base_url: str | None = None):
    """Start interactive REPL."""
    from src.config import ConfigError, missing_api_key_message, resolve_provider_config
    from src.project import detect_project
    from src.repl import ClawdREPL

    console = Console()
    project = detect_project()
    try:
        resolved = resolve_provider_config(
            cli_provider=provider, cli_model=model, cli_base_url=base_url, project_root=project.project_root,
        )
    except ConfigError as e:
        console.print(f"[red]Configuration error: {e}[/red]")
        return 2
    for warning in resolved.warnings:
        console.print(f"[yellow]warning: {warning}[/yellow]")
    if not resolved.has_api_key:
        console.print(f"[red]Error: {missing_api_key_message(resolved)}[/red]")
        return 2
    if project.is_unsafe_root:
        console.print(
            f"[yellow]Warning: the project root is {project.project_root} (your home directory or a "
            "filesystem root). File tools can reach everything below it. Consider starting inside a project.[/yellow]"
        )

    repl = ClawdREPL(provider_name=resolved.provider, stream=stream, resolved=resolved, project=project)
    repl.run()
    return 0

if __name__ == '__main__':
    sys.exit(main())
