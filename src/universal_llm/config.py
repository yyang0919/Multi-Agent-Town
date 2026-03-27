from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Literal, TypeAlias
from urllib.parse import urlparse

Transport: TypeAlias = Literal["auto", "native", "openai_compatible"]

DEFAULT_SYSTEM_PROMPT = "你是一个专业、可靠、简洁的 AI 助手。"


class LLMError(RuntimeError):
    """统一的 LLM 异常基类。"""


class LLMDependencyError(LLMError):
    """LangChain 依赖缺失。"""


class LLMConfigError(LLMError):
    """配置错误。"""


class LLMInvocationError(LLMError):
    """模型调用失败。"""

PROVIDER_ALIASES: dict[str, str] = {
    "custom": "openai_compatible",
    "google": "google_genai",
    "google-genai": "google_genai",
    "lm-studio": "lm_studio",
    "local": "local_openai",
    "local-openai": "local_openai",
    "openai-compatible": "openai_compatible",
}

PROVIDER_SPECS: dict[str, dict[str, Any]] = {
    # LangChain native providers：保留最常见、最容易讲解的代表
    "anthropic": {"transport": "native", "api_key_env": "ANTHROPIC_API_KEY"},
    "google_genai": {"transport": "native", "api_key_env": "GOOGLE_API_KEY"},
    "ollama": {"transport": "native", "api_key_env": None},
    "openai": {"transport": "native", "api_key_env": "OPENAI_API_KEY"},
    # OpenAI-compatible providers：保留云端兼容接口 + 本地接口代表
    "deepseek": {"transport": "openai_compatible", "api_key_env": "DEEPSEEK_API_KEY"},
    "lm_studio": {"transport": "openai_compatible", "api_key_env": None, "local": True},
    "local_openai": {"transport": "openai_compatible", "api_key_env": None, "local": True},
    "openai_compatible": {"transport": "openai_compatible", "api_key_env": "OPENAI_API_KEY"},
    "vllm": {"transport": "openai_compatible", "api_key_env": None, "local": True},
}

NATIVE_PROVIDERS = frozenset(
    provider for provider, spec in PROVIDER_SPECS.items() if spec["transport"] == "native"
)

OPENAI_COMPATIBLE_PROVIDERS = frozenset(
    provider for provider, spec in PROVIDER_SPECS.items() if spec["transport"] == "openai_compatible"
)

LOCAL_OPENAI_COMPATIBLE_PROVIDERS = frozenset(
    provider for provider, spec in PROVIDER_SPECS.items() if spec.get("local")
)

DEFAULT_API_KEY_ENVS = {
    provider: spec.get("api_key_env") for provider, spec in PROVIDER_SPECS.items()
}


try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv() -> bool:
        return False


load_dotenv()


@lru_cache(maxsize=1)
def load_langchain_runtime() -> dict[str, Any]:
    try:
        from langchain.chat_models import init_chat_model
        from langchain_core.messages import AIMessage, BaseMessage, ChatMessage, HumanMessage, SystemMessage
        from langchain_openai import ChatOpenAI
    except ImportError as exc:
        raise LLMDependencyError(
            "缺少 LangChain 依赖，请先执行: pip install -r requirements.txt"
        ) from exc

    return {
        "AIMessage": AIMessage,
        "BaseMessage": BaseMessage,
        "ChatMessage": ChatMessage,
        "ChatOpenAI": ChatOpenAI,
        "HumanMessage": HumanMessage,
        "SystemMessage": SystemMessage,
        "init_chat_model": init_chat_model,
    }


@dataclass(frozen=True, slots=True)
class ResolvedLLMConfig:
    provider: str
    transport: Literal["native", "openai_compatible"]
    model: str
    api_key: str | None
    api_key_env: str | None
    base_url: str | None
    temperature: float | None
    max_tokens: int | None
    timeout: float | None
    max_retries: int
    top_p: float | None
    presence_penalty: float | None
    frequency_penalty: float | None
    system_prompt: str
    headers: dict[str, str]
    tags: list[str]
    metadata: dict[str, Any]
    model_kwargs: dict[str, Any]
    extra_body: dict[str, Any]


@dataclass(slots=True)
class LLMResponse:
    text: str
    provider: str
    model: str
    usage: dict[str, Any] = field(default_factory=dict)
    response_metadata: dict[str, Any] = field(default_factory=dict)
    raw: Any = None


@dataclass(slots=True)
class LLMConfig:
    """统一的 LLM 配置。"""

    provider: str
    model: str
    api_key: str | None = None
    api_key_env: str | None = None
    base_url: str | None = None
    temperature: float | None = 0.7
    max_tokens: int | None = 1024
    timeout: float | None = 60.0
    max_retries: int = 2
    top_p: float | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    transport: Transport = "auto"
    headers: dict[str, str] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    model_kwargs: dict[str, Any] = field(default_factory=dict)
    extra_body: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.provider = self.provider.strip()
        self.model = self.model.strip()
        self.base_url = self.base_url.rstrip("/") if self.base_url else None
        self.system_prompt = self.system_prompt.strip()

        if not self.provider:
            raise LLMConfigError("provider 不能为空。")
        if not self.model:
            raise LLMConfigError("model 不能为空。")
        if self.transport not in {"auto", "native", "openai_compatible"}:
            raise LLMConfigError("transport 仅支持 auto / native / openai_compatible。")

        self._validate_range("temperature", self.temperature, min_value=0.0)
        self._validate_range("top_p", self.top_p, min_value=0.0, max_value=1.0)
        self._validate_range("presence_penalty", self.presence_penalty, min_value=-2.0, max_value=2.0)
        self._validate_range("frequency_penalty", self.frequency_penalty, min_value=-2.0, max_value=2.0)
        self._validate_int("max_tokens", self.max_tokens, minimum=1)
        self._validate_int("max_retries", self.max_retries, minimum=0)

    @classmethod
    def for_vllm(
        cls,
        model: str,
        base_url: str = "http://127.0.0.1:8000/v1",
        **kwargs: Any,
    ) -> "LLMConfig":
        kwargs.setdefault("provider", "vllm")
        kwargs.setdefault("transport", "openai_compatible")
        kwargs.setdefault("api_key", "EMPTY")
        return cls(model=model, base_url=base_url, **kwargs)

    @classmethod
    def for_openai_compatible(
        cls,
        provider: str,
        model: str,
        base_url: str,
        **kwargs: Any,
    ) -> "LLMConfig":
        kwargs.setdefault("transport", "openai_compatible")
        return cls(provider=provider, model=model, base_url=base_url, **kwargs)

    @property
    def normalized_provider(self) -> str:
        normalized = self.provider.strip().lower().replace(" ", "_")
        return PROVIDER_ALIASES.get(normalized, normalized)

    @property
    def provider_spec(self) -> dict[str, Any]:
        return PROVIDER_SPECS.get(self.normalized_provider, {})

    def resolve(self) -> ResolvedLLMConfig:
        provider = self.normalized_provider
        provider_spec = self.provider_spec
        transport = self._resolve_transport(provider, provider_spec)

        if transport == "native" and provider not in NATIVE_PROVIDERS:
            supported = ", ".join(sorted(NATIVE_PROVIDERS))
            raise LLMConfigError(
                f"provider={provider!r} 不在当前内置的 LangChain native provider 列表中。"
                f"可用 provider: {supported}。"
            )

        if transport == "openai_compatible" and not self.base_url:
            raise LLMConfigError("OpenAI-compatible / vLLM 模式必须提供 base_url。")

        api_key_env = self.api_key_env or provider_spec.get("api_key_env")
        api_key = self.api_key or (os.getenv(api_key_env) if api_key_env else None)

        if transport == "openai_compatible" and not api_key:
            if provider_spec.get("local") or self._looks_like_local_endpoint(self.base_url):
                api_key = "EMPTY"
            else:
                raise LLMConfigError(
                    f"provider={provider!r} 需要 api_key。"
                    f"请传入 api_key，或设置环境变量 {api_key_env or '<YOUR_API_KEY_ENV>'}。"
                )

        return ResolvedLLMConfig(
            provider=provider,
            transport=transport,
            model=self.model,
            api_key=api_key,
            api_key_env=api_key_env,
            base_url=self.base_url,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            timeout=self.timeout,
            max_retries=self.max_retries,
            top_p=self.top_p,
            presence_penalty=self.presence_penalty,
            frequency_penalty=self.frequency_penalty,
            system_prompt=self.system_prompt or DEFAULT_SYSTEM_PROMPT,
            headers=dict(self.headers),
            tags=list(self.tags),
            metadata=dict(self.metadata),
            model_kwargs=dict(self.model_kwargs),
            extra_body=dict(self.extra_body),
        )

    def _resolve_transport(
        self,
        provider: str,
        provider_spec: dict[str, Any],
    ) -> Literal["native", "openai_compatible"]:
        if self.transport == "native":
            return "native"
        if self.transport == "openai_compatible":
            return "openai_compatible"
        if provider_spec:
            return provider_spec["transport"]
        if self.base_url:
            return "openai_compatible"
        raise LLMConfigError(
            f"无法自动判断 provider={provider!r} 的路由方式。"
            "请显式设置 transport='native' 或 transport='openai_compatible'。"
        )

    @staticmethod
    def _looks_like_local_endpoint(url: str | None) -> bool:
        if not url:
            return False
        hostname = (urlparse(url).hostname or "").lower()
        return hostname in {"127.0.0.1", "0.0.0.0", "localhost"}

    @staticmethod
    def _validate_range(
        name: str,
        value: float | None,
        *,
        min_value: float | None = None,
        max_value: float | None = None,
    ) -> None:
        if value is None:
            return
        if min_value is not None and value < min_value:
            raise LLMConfigError(f"{name} 不能小于 {min_value}。")
        if max_value is not None and value > max_value:
            raise LLMConfigError(f"{name} 不能大于 {max_value}。")

    @staticmethod
    def _validate_int(name: str, value: int | None, *, minimum: int = 0) -> None:
        if value is None:
            return
        if value < minimum:
            raise LLMConfigError(f"{name} 不能小于 {minimum}。")
