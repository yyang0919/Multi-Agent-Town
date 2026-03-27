# Universal LLM

一个基于 **LangChain** 的通用 LLM 调用项目，支持：

- LangChain 原生 provider
- OpenAI-compatible provider
- 本地 **vLLM** 模型服务
- 普通对话、流式输出、异步调用

本项目已经采用标准 **`src-layout`** 工程结构，适合作为：

- 课程作业
- LLM 接口统一封装练习
- 本地 vLLM + LangChain 联调示例
- 后续扩展 Agent / RAG / Tool 的基础工程

---

## 项目结构

当前核心目录如下：

```text
.
├─ src/
│  └─ universal_llm/
│     ├─ __init__.py
│     ├─ client.py
│     ├─ config.py
│     ├─ factory.py
│     └─ messages.py
├─ scripts/
│  ├─ check_env.py
│  ├─ run_openai_example.py
│  └─ run_vllm_example.py
├─ examples/
│  ├─ basic_chat.py
│  └─ stream_chat.py
├─ tests/
│  └─ test_config.py
├─ pyproject.toml
├─ requirements.txt
├─ README.md
└─ README_langchain_vllm.md
```

> 说明：执行 `pip install -e .` 后，`src/` 下通常还会生成 `universal_llm.egg-info/`，这是标准安装元数据目录，不属于手写源码。

---

## 快速开始

### 1. 安装依赖

```powershell
pip install -r requirements.txt
pip install -e .
```

### 2. 启动本地 vLLM

详细步骤请查看：

**[README_langchain_vllm.md](./README_langchain_vllm.md)**

### 3. 运行示例

```powershell
python scripts/run_vllm_example.py
python scripts/run_openai_example.py
python examples/basic_chat.py
python examples/stream_chat.py
```

### 4. 运行测试

```powershell
python tests/test_config.py
```

---

## 支持的 provider

### LangChain 原生 provider

- `openai`
- `anthropic`
- `google_genai`
- `ollama`

### OpenAI-compatible provider

- `deepseek`
- `openai_compatible`
- `vllm`
- `lm_studio`
- `local_openai`

---

## 核心设计

### `src/universal_llm/config.py`
- 配置类
- provider 注册信息
- 响应结构
- 异常类
- LangChain 运行时加载

### `src/universal_llm/factory.py`
- 根据配置创建底层 chat model

### `src/universal_llm/messages.py`
- 统一消息结构
- role 标准化
- 输出文本提取

### `src/universal_llm/client.py`
- `UniversalLLM` 主调用接口

---

## 文档说明

- `README.md`：项目主页与快速开始
- `README_langchain_vllm.md`：详细部署文档、工作流程、目录说明、运行方式

---

## 适合汇报时强调的点

- 统一接口设计
- 本地模型与云端模型统一调用
- vLLM 本地部署流程
- 标准 `src-layout` 工程结构
- 便于扩展到 Agent / RAG / Tool
