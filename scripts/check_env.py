import os


def show_env(name: str) -> None:
    print(f"{name}: {'已设置' if os.getenv(name) else '未设置'}")


print("环境变量检查：")
show_env("OPENAI_API_KEY")
show_env("DEEPSEEK_API_KEY")
show_env("ANTHROPIC_API_KEY")

print("\n导入检查：")
try:
    import universal_llm  # noqa: F401

    print("universal_llm: OK")
except Exception as exc:
    print(f"universal_llm: FAIL -> {exc}")
