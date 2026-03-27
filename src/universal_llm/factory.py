from __future__ import annotations

from typing import Any

from .config import ResolvedLLMConfig, load_langchain_runtime


def _compact(**kwargs: Any) -> dict[str, Any]:
    return {key: value for key, value in kwargs.items() if value not in (None, {}, [], ())}


def create_chat_model(config: ResolvedLLMConfig) -> Any:
    """根据统一配置创建 LangChain chat model。"""
    runtime = load_langchain_runtime()

    if config.transport == "native":
        kwargs = _compact(
            model=config.model,
            model_provider=config.provider,
            api_key=config.api_key,
            base_url=config.base_url,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            timeout=config.timeout,
            max_retries=config.max_retries,
            tags=config.tags or None,
            metadata=config.metadata or None,
        )
        kwargs.update(config.model_kwargs)
        return runtime["init_chat_model"](**kwargs)

    kwargs = _compact(
        model=config.model,
        api_key=config.api_key,
        base_url=config.base_url,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        timeout=config.timeout,
        max_retries=config.max_retries,
        top_p=config.top_p,
        presence_penalty=config.presence_penalty,
        frequency_penalty=config.frequency_penalty,
        default_headers=config.headers or None,
        tags=config.tags or None,
        metadata=config.metadata or None,
        model_kwargs=config.model_kwargs or None,
        extra_body=config.extra_body or None,
    )
    return runtime["ChatOpenAI"](**kwargs)
