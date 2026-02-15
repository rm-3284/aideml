"""Backend for vLLM OpenAI-compatible API."""

import json
import logging
import os
import time

from funcy import notnone, once, select_values
import openai

from .utils import FunctionSpec, OutputType, opt_messages_to_list, backoff_create

logger = logging.getLogger("aide")

_client: openai.OpenAI = None  # type: ignore

OPENAI_TIMEOUT_EXCEPTIONS = (
    openai.RateLimitError,
    openai.APIConnectionError,
    openai.APITimeoutError,
    openai.InternalServerError,
)


def _get_vllm_base_url() -> str | None:
    return os.getenv("OPENAI_API_BASE") or os.getenv("VLLM_API_BASE")


@once
def _setup_vllm_client():
    global _client
    base_url = _get_vllm_base_url()
    if not base_url:
        raise RuntimeError(
            "vLLM backend requires OPENAI_API_BASE or VLLM_API_BASE to be set."
        )
    api_key = os.getenv("OPENAI_API_KEY") or "EMPTY"
    _client = openai.OpenAI(api_key=api_key, base_url=base_url, max_retries=0)


def query(
    system_message: str | None,
    user_message: str | None,
    func_spec: FunctionSpec | None = None,
    **model_kwargs,
) -> tuple[OutputType, float, int, int, dict]:
    _setup_vllm_client()

    filtered_kwargs: dict = select_values(notnone, model_kwargs)

    messages = opt_messages_to_list(system_message, user_message)
    if func_spec is not None:
        filtered_kwargs["tools"] = [func_spec.as_openai_tool_dict]
        filtered_kwargs["tool_choice"] = func_spec.openai_tool_choice_dict

    logger.info(f"vLLM API request: system={system_message}, user={user_message}")

    t0 = time.time()
    try:
        response = backoff_create(
            _client.chat.completions.create,
            OPENAI_TIMEOUT_EXCEPTIONS,
            messages=messages,
            **filtered_kwargs,
        )
    except openai.BadRequestError as e:
        # Handle models that do not support tool calling
        if "function calling" in str(e).lower() or "tools" in str(e).lower():
            logger.warning(
                "Tool calling not supported by this model. Falling back to text."
            )
            filtered_kwargs.pop("tools", None)
            filtered_kwargs.pop("tool_choice", None)
            response = backoff_create(
                _client.chat.completions.create,
                OPENAI_TIMEOUT_EXCEPTIONS,
                messages=messages,
                **filtered_kwargs,
            )
        else:
            raise

    req_time = time.time() - t0

    message = response.choices[0].message
    if (
        hasattr(message, "tool_calls")
        and message.tool_calls
        and func_spec is not None
    ):
        tool_call = message.tool_calls[0]
        if tool_call.function.name == func_spec.name:
            try:
                output = json.loads(tool_call.function.arguments)
            except json.JSONDecodeError as ex:
                logger.error(
                    "Error decoding function arguments:\n"
                    f"{tool_call.function.arguments}"
                )
                raise ex
        else:
            logger.warning(
                f"Function name mismatch: expected {func_spec.name}, "
                f"got {tool_call.function.name}. Fallback to text."
            )
            output = message.content
    else:
        output = message.content

    usage = getattr(response, "usage", None)
    in_tokens = getattr(usage, "prompt_tokens", 0) if usage else 0
    out_tokens = getattr(usage, "completion_tokens", 0) if usage else 0

    info = {
        "system_fingerprint": getattr(response, "system_fingerprint", None),
        "model": response.model,
        "created": getattr(response, "created", None),
    }

    logger.info(
        f"vLLM API call completed - {response.model} - {req_time:.2f}s - "
        f"{in_tokens + out_tokens} tokens (in: {in_tokens}, out: {out_tokens})"
    )
    logger.info(f"vLLM API response: {output}")

    return output, req_time, in_tokens, out_tokens, info
