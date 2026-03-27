from __future__ import annotations

from typing import Any, Iterable, Mapping, TypeAlias

from .config import LLMConfigError, load_langchain_runtime

MessageContent: TypeAlias = str | list[Any]
MessageLike: TypeAlias = Any

ROLE_ALIASES = {
    "assistant": "ai",
    "bot": "ai",
    "human": "human",
    "model": "ai",
    "system": "system",
    "user": "human",
}


def normalize_role(role: str) -> str:
    normalized = role.strip().lower()
    return ROLE_ALIASES.get(normalized, normalized)


def safe_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, Mapping) and "text" in item:
                parts.append(str(item["text"]))
            else:
                parts.append(str(item))
        return "".join(parts)
    return str(content)


def make_message(role: str, content: MessageContent) -> Any:
    runtime = load_langchain_runtime()
    normalized_role = normalize_role(role)

    if normalized_role == "system":
        return runtime["SystemMessage"](content=content)
    if normalized_role == "human":
        return runtime["HumanMessage"](content=content)
    if normalized_role == "ai":
        return runtime["AIMessage"](content=content)
    return runtime["ChatMessage"](role=role, content=content)


def coerce_message(message: MessageLike) -> Any:
    runtime = load_langchain_runtime()
    base_message_cls = runtime["BaseMessage"]
    chat_message_cls = runtime["ChatMessage"]

    if isinstance(message, base_message_cls):
        return message

    if isinstance(message, tuple) and len(message) == 2:
        role, content = message
        return make_message(role, content)

    if isinstance(message, Mapping):
        role = message.get("role") or message.get("type")
        if not role:
            raise LLMConfigError("字典消息必须包含 role 或 type 字段。")

        content = message.get("content", "")
        name = message.get("name")
        additional_kwargs = safe_dict(message.get("additional_kwargs"))
        normalized_role = normalize_role(str(role))

        if normalized_role in {"system", "human", "ai"}:
            return make_message(normalized_role, content)

        return chat_message_cls(
            role=str(role),
            content=content,
            name=name,
            additional_kwargs=additional_kwargs,
        )

    raise LLMConfigError(f"不支持的消息类型: {type(message)!r}")


def build_messages(
    *,
    prompt: MessageContent | None,
    system_prompt: str | None,
    default_system_prompt: str,
    history: Iterable[MessageLike] | None,
) -> list[Any]:
    messages: list[Any] = []
    effective_system_prompt = default_system_prompt if system_prompt is None else system_prompt.strip()

    if effective_system_prompt:
        messages.append(make_message("system", effective_system_prompt))

    if history:
        messages.extend(coerce_message(item) for item in history)

    if prompt is not None:
        messages.append(make_message("human", prompt))

    if not messages:
        raise LLMConfigError("消息列表不能为空。")

    return messages
