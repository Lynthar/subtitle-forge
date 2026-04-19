"""HTTP server command — exposes subtitle-forge as a REST API."""

import typer

app = typer.Typer(no_args_is_help=False)


@app.command("run")
@app.command(hidden=True)
def serve(
    host: str = typer.Option("0.0.0.0", "--host", help="Bind address"),
    port: int = typer.Option(8765, "--port", "-p", help="Bind port"),
    workers: int = typer.Option(
        1,
        "--workers",
        "-w",
        min=1,
        max=4,
        help="Concurrent job workers. GPU-bound — leave at 1 unless you know what you're doing.",
    ),
    no_auth: bool = typer.Option(
        False,
        "--no-auth",
        help="Disable bearer-token auth. Only safe on a fully trusted host (loopback only).",
    ),
    log_level: str = typer.Option("info", "--log-level", help="uvicorn log level"),
):
    """
    Start the subtitle-forge HTTP server.

    Auth: by default the server requires a bearer token, read from the
    SUBTITLE_FORGE_TOKEN environment variable. Pick a long random string and
    set it before launching:

        export SUBTITLE_FORGE_TOKEN="$(openssl rand -hex 32)"
        subtitle-forge serve

    Use --no-auth ONLY when binding to 127.0.0.1 on a single-user machine.
    """
    try:
        import uvicorn
    except ImportError:
        typer.echo(
            "ERROR: serve requires the [serve] extra. Install with:\n"
            "  pip install -e '.[serve]'",
            err=True,
        )
        raise typer.Exit(1)

    from ...server.auth import TOKEN_ENV_VAR, get_configured_token
    from ...server.app import create_app

    if not no_auth:
        if get_configured_token() is None:
            typer.echo(
                f"ERROR: ${TOKEN_ENV_VAR} is not set. Either set it (recommended)\n"
                f"or pass --no-auth (only safe on a trusted single-user host).",
                err=True,
            )
            raise typer.Exit(2)
    else:
        if host not in ("127.0.0.1", "localhost"):
            typer.echo(
                f"WARNING: --no-auth on a non-loopback host ({host}). Anyone on "
                f"the network can submit jobs.",
                err=True,
            )

    app_instance = create_app(max_workers=workers, require_auth=not no_auth)

    uvicorn.run(
        app_instance,
        host=host,
        port=port,
        log_level=log_level,
    )
