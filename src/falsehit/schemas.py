"""OpenAI-style request and response models for the chat completions endpoint."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="ignore")

    role: Literal["system", "developer", "user", "assistant", "tool"]
    content: str = ""

    @field_validator("content", mode="before")
    @classmethod
    def _flatten_parts(cls, value: Any) -> Any:
        """Accept the list-of-text-parts form and flatten it to a string."""
        if value is None:
            return ""
        if isinstance(value, list):
            parts = []
            for part in value:
                if isinstance(part, dict) and part.get("type", "text") == "text":
                    parts.append(str(part.get("text", "")))
                else:
                    raise ValueError("only text content parts are supported")
            return "\n".join(parts)
        return value


class ChatRequest(BaseModel):
    """The subset of the OpenAI request that matters for caching, plus three cache controls."""

    model_config = ConfigDict(extra="ignore")

    model: str
    messages: list[ChatMessage] = Field(min_length=1)
    temperature: float | None = Field(default=None, ge=0, le=2)
    top_p: float | None = Field(default=None, gt=0, le=1)
    max_tokens: int | None = Field(default=None, gt=0)
    tools: list[dict[str, Any]] | None = None
    response_format: dict[str, Any] | None = None
    stream: bool = False
    cache: bool = True
    cache_tag: str | None = Field(default=None, max_length=64)
    cache_ttl: int | None = Field(default=None, gt=0)


class ResponseMessage(BaseModel):
    role: Literal["assistant"] = "assistant"
    content: str


class Choice(BaseModel):
    index: int = 0
    message: ResponseMessage
    finish_reason: Literal["stop", "length", "tool_calls", "content_filter"] = "stop"


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: list[Choice]
    usage: Usage


class InvalidateRequest(BaseModel):
    tag: str | None = Field(default=None, max_length=64)


class ErrorBody(BaseModel):
    message: str
    type: str
    code: str


class ErrorResponse(BaseModel):
    error: ErrorBody
