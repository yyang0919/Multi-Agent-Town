from __future__ import annotations

from typing import Any, Iterable, Iterator

from .config import LLMConfig, LLMInvocationError, LLMResponse, NATIVE_PROVIDERS, OPENAI_COMPATIBLE_PROVIDERS
from .factory import create_chat_model
from .messages import MessageContent, MessageLike, build_messages, coerce_message, content_to_text, safe_dict


class UniversalLLM:
    """简洁但通用的 LangChain LLM 封装。"""

    def __init__(self, config: LLMConfig):
        self.config = config
        self.resolved_config = config.resolve()
        self.llm = create_chat_model(self.resolved_config)

    @staticmethod
    def supported_native_providers() -> tuple[str, ...]:
        return tuple(sorted(NATIVE_PROVIDERS))

    @staticmethod
    def supported_openai_compatible_examples() -> tuple[str, ...]:
        return tuple(sorted(OPENAI_COMPATIBLE_PROVIDERS))

    def build_messages(
        self,
        prompt: MessageContent | None = None,
        *,
        system_prompt: str | None = None,
        history: Iterable[MessageLike] | None = None,
    ) -> list[Any]:
        return build_messages(
            prompt=prompt,
            system_prompt=system_prompt,
            default_system_prompt=self.resolved_config.system_prompt,
            history=history,
        )

    def generate(
        self,
        prompt: MessageContent,
        *,
        system_prompt: str | None = None,
        history: Iterable[MessageLike] | None = None,
    ) -> LLMResponse:
        return self.invoke_messages(self.build_messages(prompt, system_prompt=system_prompt, history=history))

    def chat(
        self,
        prompt: MessageContent,
        *,
        system_prompt: str | None = None,
        history: Iterable[MessageLike] | None = None,
    ) -> str:
        return self.generate(prompt, system_prompt=system_prompt, history=history).text

    def invoke_messages(self, messages: Iterable[MessageLike]) -> LLMResponse:
        normalized_messages = [coerce_message(item) for item in messages]
        try:
            message = self.llm.invoke(normalized_messages)
        except Exception as exc:
            raise LLMInvocationError(
                f"模型调用失败(provider={self.resolved_config.provider}, model={self.resolved_config.model})."
            ) from exc
        return self._to_response(message)

    def stream(
        self,
        prompt: MessageContent,
        *,
        system_prompt: str | None = None,
        history: Iterable[MessageLike] | None = None,
    ) -> Iterator[str]:
        yield from self.stream_messages(self.build_messages(prompt, system_prompt=system_prompt, history=history))

    def stream_messages(self, messages: Iterable[MessageLike]) -> Iterator[str]:
        normalized_messages = [coerce_message(item) for item in messages]
        try:
            for chunk in self.llm.stream(normalized_messages):
                text = content_to_text(getattr(chunk, "content", ""))
                if text:
                    yield text
        except Exception as exc:
            raise LLMInvocationError(
                f"模型流式调用失败(provider={self.resolved_config.provider}, model={self.resolved_config.model})."
            ) from exc

    async def agenerate(
        self,
        prompt: MessageContent,
        *,
        system_prompt: str | None = None,
        history: Iterable[MessageLike] | None = None,
    ) -> LLMResponse:
        return await self.ainvoke_messages(self.build_messages(prompt, system_prompt=system_prompt, history=history))

    async def achat(
        self,
        prompt: MessageContent,
        *,
        system_prompt: str | None = None,
        history: Iterable[MessageLike] | None = None,
    ) -> str:
        return (await self.agenerate(prompt, system_prompt=system_prompt, history=history)).text

    async def ainvoke_messages(self, messages: Iterable[MessageLike]) -> LLMResponse:
        normalized_messages = [coerce_message(item) for item in messages]
        try:
            message = await self.llm.ainvoke(normalized_messages)
        except Exception as exc:
            raise LLMInvocationError(
                f"模型异步调用失败(provider={self.resolved_config.provider}, model={self.resolved_config.model})."
            ) from exc
        return self._to_response(message)

    async def astream(
        self,
        prompt: MessageContent,
        *,
        system_prompt: str | None = None,
        history: Iterable[MessageLike] | None = None,
    ):
        async for chunk in self.astream_messages(
            self.build_messages(prompt, system_prompt=system_prompt, history=history)
        ):
            yield chunk

    async def astream_messages(self, messages: Iterable[MessageLike]):
        normalized_messages = [coerce_message(item) for item in messages]
        try:
            async for chunk in self.llm.astream(normalized_messages):
                text = content_to_text(getattr(chunk, "content", ""))
                if text:
                    yield text
        except Exception as exc:
            raise LLMInvocationError(
                f"模型异步流式调用失败(provider={self.resolved_config.provider}, model={self.resolved_config.model})."
            ) from exc

    def get_llm(self) -> Any:
        return self.llm

    def _to_response(self, message: Any) -> LLMResponse:
        return LLMResponse(
            text=content_to_text(getattr(message, "content", "")),
            provider=self.resolved_config.provider,
            model=self.resolved_config.model,
            usage=safe_dict(getattr(message, "usage_metadata", None)),
            response_metadata=safe_dict(getattr(message, "response_metadata", None)),
            raw=message,
        )
