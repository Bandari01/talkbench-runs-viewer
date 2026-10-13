"""One path-isolated MCP host for the seven synthetic TalkBench domains.

    cd <ai-ds-research>/talk-bench-talkdesk
    uv run python <viewer>/talkdesk_best_seven/seven_domain_server.py [--port 3030] [--ngrok]

Upstream's ``talkdesk-mcp-server-seven`` (talkdesk_mcp_server.seven_domain_server),
unchanged. Each domain is served at ``/<domain>/`` with its own registry.
"""

from __future__ import annotations

import logging
from functools import partial
from typing import Iterable

import click
from talk_bench_core.cli.logging import setup_logging

from talkdesk_mcp_server.backends import NativeBackend
from talkdesk_mcp_server.registry import ConversationRegistry
from talkdesk_mcp_server.server import build_app

logger = logging.getLogger(__name__)

SEVEN_SYNTHETIC_DOMAINS = (
    "cedarway_specialty_pharmacy",
    "everfield_utilities",
    "griddlepoint",
    "harborstone_credit_union",
    "meridian",
    "northline_insurance",
    "pinehaven_childrens_health",
)


def build_seven_domain_app(
    domains: Iterable[str] = SEVEN_SYNTHETIC_DOMAINS,
):
    """Mount an independent native-domain app and registry at each domain path."""
    from starlette.applications import Starlette
    from starlette.requests import Request
    from starlette.responses import JSONResponse
    from starlette.routing import Mount, Route

    domain_names = tuple(domains)
    if not domain_names:
        raise ValueError("At least one domain is required")
    if len(domain_names) != len(set(domain_names)):
        raise ValueError("Domain paths must be unique")

    registries: dict[str, ConversationRegistry] = {}
    mounts = []
    for domain in domain_names:
        registry = ConversationRegistry(partial(NativeBackend, domain=domain))
        registries[domain] = registry
        mounts.append(
            Mount(
                f"/{domain}",
                app=build_app(registry, source="talk_bench", domain=domain),
                name=domain,
            )
        )

    async def root(_request: Request) -> JSONResponse:
        return JSONResponse(
            {
                "name": "talkdesk-mcp-talk_bench-seven-domain",
                "status": "ok",
                "domains": {
                    domain: {
                        "server_url_path": f"/{domain}/",
                        "health_path": f"/{domain}/health",
                    }
                    for domain in domain_names
                },
            }
        )

    async def health(_request: Request) -> JSONResponse:
        return JSONResponse(
            {
                "status": "ok",
                "domains": {
                    domain: {
                        "activeContexts": await registry.active_count(),
                        "server_url_path": f"/{domain}/",
                    }
                    for domain, registry in registries.items()
                },
            }
        )

    return Starlette(
        debug=False,
        routes=[
            Route("/", endpoint=root, methods=["GET"]),
            Route("/health", endpoint=health, methods=["GET"]),
            *mounts,
        ],
    )


def run_seven_domain_http(
    host: str,
    port: int,
    *,
    use_ngrok: bool = False,
    log_level: str = "INFO",
) -> None:
    """Run all seven path-isolated domain apps in one process and tunnel."""
    import uvicorn

    app = build_seven_domain_app()
    logger.info("Starting seven-domain MCP host on http://%s:%d", host, port)
    for domain in SEVEN_SYNTHETIC_DOMAINS:
        logger.info("  %s -> /%s/", domain, domain)

    if use_ngrok:
        import ngrok

        listener = ngrok.forward(port, authtoken_from_env=True)
        logger.info("  ngrok base URL: %s", listener.url())

    uvicorn.run(app, host=host, port=port, log_level=log_level.lower())


@click.command()
@click.option("--host", default="0.0.0.0", show_default=True)
@click.option("--port", type=int, default=3030, show_default=True)
@click.option("--ngrok", is_flag=True, help="Expose the shared host through one ngrok tunnel.")
@click.option(
    "--log-level",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"], case_sensitive=False),
    default="INFO",
    show_default=True,
)
def main(host: str, port: int, ngrok: bool, log_level: str) -> None:
    """Serve all seven synthetic domains under isolated URL paths."""
    setup_logging(log_level)
    run_seven_domain_http(
        host=host,
        port=port,
        use_ngrok=ngrok,
        log_level=log_level,
    )


if __name__ == "__main__":
    main()
