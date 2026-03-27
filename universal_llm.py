from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Iterator, Optional

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_openai import ChatOpenAI

load_dotenv()


MessageLike = tuple[str, str]


@dataclass
class LLMConfig:
    """统一管理大模型调用参数。"""

    provider: str
    model: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 1024
    timeout: int = 60
    model_kwargs: dict[str, Any] = field(default_factory=dict)
    extra_body: dict[str, Any] = field(default_factory=dict)


class UniversalLLM:
    """基于 LangChain 的通用 LLM 调用封装。"""

    def __init__(self, config: LLMConfig):
        self.config = config
        self.llm = self._build_llm()

    def _build_llm(self):
        # 如果提供了 base_url，就按 OpenAI 兼容接口处理。
        # 例如：vLLM、DeepSeek、Moonshot、LM Studio 等。
        if self.config.base_url:
            kwargs = dict(
                model=self.config.model,
                api_key=self.config.api_key or "EMPTY",
                base_url=self.config.base_url,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                timeout=self.config.timeout,
                extra_body=self.config.extra_body or None,
            )
            kwargs.update(self.config.model_kwargs)
            kwargs = {k: v for k, v in kwargs.items() if v is not None}
            return ChatOpenAI(**kwargs)

        # 否则走 LangChain 官方 provider。
        # 例如：OpenAI、Anthropic、Google、Groq、Ollama 等。
        kwargs = dict(
            model=self.config.model,
            model_provider=self.config.provider,
            api_key=self.config.api_key,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            timeout=self.config.timeout,
        )
        kwargs.update(self.config.model_kwargs)
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        return init_chat_model(**kwargs)

    @staticmethod
    def build_messages(
        prompt: str,
        system_prompt: str = "你是一个有帮助的 AI 助手。",
        history: Optional[Iterable[MessageLike]] = None,
    ):
        messages: list[MessageLike] = [("system", system_prompt)]
        if history:
            messages.extend(history)
        messages.append(("human", prompt))
        return messages

    @staticmethod
    def _text(content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict) and "text" in item:
                    parts.append(str(item["text"]))
                else:
                    parts.append(str(item))
            return "".join(parts)
        return str(content)

    def chat(
        self,
        prompt: str,
        system_prompt: str = "你是一个有帮助的 AI 助手。",
        history: Optional[Iterable[MessageLike]] = None,
    ) -> str:
        messages = self.build_messages(prompt, system_prompt, history)
        response = self.llm.invoke(messages)
        return self._text(response.content)

    def stream(
        self,
        prompt: str,
        system_prompt: str = "你是一个有帮助的 AI 助手。",
        history: Optional[Iterable[MessageLike]] = None,
    ) -> Iterator[str]:
        messages = self.build_messages(prompt, system_prompt, history)
        for chunk in self.llm.stream(messages):
            text = self._text(chunk.content)
            if text:
                yield text

    def get_llm(self):
        return self.llm


if __name__ == "__main__":
    config = LLMConfig(
        provider="vllm",
        model="Qwen/Qwen2.5-1.5B-Instruct",
        base_url="http://127.0.0.1:8000/v1",
        api_key="EMPTY",
        temperature=0.2,
        extra_body={"top_k": 20},
    )

    client = UniversalLLM(config)
    history = [
        ("human", "你是谁？"),
        ("ai", "我是一个本地部署的 AI 助手。"),
    ]
    print(client.chat("请继续用中文介绍一下 LangChain。", history=history))
