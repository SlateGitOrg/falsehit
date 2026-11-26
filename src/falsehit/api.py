"""The FastAPI application: OpenAI-compatible chat completions with a semantic cache in front."""

from __future__ import annotations

import os

from fastapi import Depends, FastAPI, Header, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response

from .clock import Clock
from .config import Settings, load_settings
from .gateway import Gateway, GatewayError
from .resilience import CircuitBreaker, FailoverUpstream
from .schemas import (
    ChatCompletionResponse,
    ChatRequest,
    ErrorBody,
    ErrorResponse,
    InvalidateRequest,
)
from .upstream import ChatModel, FakeUpstream, OpenAICompatUpstream


def _error(status: int, code: str, message: str, headers: dict[str, str] | None = None):
    body = ErrorResponse(error=ErrorBody(message=message, type=code, code=code))
    return JSONResponse(body.model_dump(), status_code=status, headers=headers)


def create_app(
    settings: Settings,
    upstream: ChatModel | None = None,
    clock: Clock | None = None,
) -> FastAPI:
    """Build the service. `upstream` defaults to the offline fake; the app keeps its gateway on
    `app.state.gateway` so tests and the replay can inspect the cache and the ledger."""
    if upstream is None:
        upstream = FakeUpstream("fake")
    gateway = Gateway(settings, upstream, clock)
    app = FastAPI(title="falsehit", version="0.1.0")
    app.state.gateway = gateway

    @app.middleware("http")
    async def default_cache_header(request: Request, call_next):
        response: Response = await call_next(request)
        response.headers.setdefault("x-cache", "bypass")
        return response

    @app.exception_handler(GatewayError)
    async def _gateway_error(_: Request, exc: GatewayError):
        return _error(exc.status, exc.code, exc.message, exc.headers)

    @app.exception_handler(RequestValidationError)
    async def _invalid(_: Request, exc: RequestValidationError):
        first = exc.errors()[0] if exc.errors() else {}
        where = ".".join(str(p) for p in first.get("loc", ()))
        return _error(
            400, "invalid_request_error", f"{where}: {first.get('msg', 'invalid request')}"
        )

    def tenant_of(authorization: str | None = Header(default=None)) -> str:
        scheme, _, token = (authorization or "").partition(" ")
        tenant = settings.tenant_for_key(token.strip()) if scheme.lower() == "bearer" else None
        if tenant is None:
            raise GatewayError(401, "invalid_api_key", "missing or unknown bearer key")
        return tenant

    @app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
    def chat_completions(body: ChatRequest, response: Response, tenant: str = Depends(tenant_of)):
        result = gateway.handle(tenant, body)
        for name, value in result.headers().items():
            response.headers[name] = value
        return result.response

    @app.get("/v1/cache/stats")
    def cache_stats(tenant: str = Depends(tenant_of)):
        return gateway.stats(tenant)

    @app.post("/v1/cache/invalidate")
    def cache_invalidate(body: InvalidateRequest, tenant: str = Depends(tenant_of)):
        removed = gateway.invalidate(tenant, body.tag)
        return {"tenant": tenant, "tag": body.tag, "removed": removed}

    return app


def create_app_from_env() -> FastAPI:
    """Factory for `uvicorn --factory`: settings from $FALSEHIT_CONFIG, upstreams from env.

    With FALSEHIT_UPSTREAM_BASE_URL unset the service runs on the offline fake upstream.
    """
    settings = load_settings()
    if "FALSEHIT_UPSTREAM_BASE_URL" in os.environ:
        primary: ChatModel = OpenAICompatUpstream.from_env("FALSEHIT_UPSTREAM", "primary")
        secondary: ChatModel | None = (
            OpenAICompatUpstream.from_env("FALSEHIT_SECONDARY", "secondary")
            if "FALSEHIT_SECONDARY_BASE_URL" in os.environ
            else None
        )
    else:
        primary, secondary = FakeUpstream("primary"), FakeUpstream("secondary")
    from .clock import SystemClock

    clock = SystemClock()
    breaker = CircuitBreaker(clock, settings.breaker_failures, settings.breaker_reset_seconds)
    return create_app(settings, FailoverUpstream(primary, secondary, breaker), clock)
