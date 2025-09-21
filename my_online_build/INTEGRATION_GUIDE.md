# LazyLLM 自定义在线模型服务集成指南（对话 & 嵌入）

本文档面向需要将第三方/自建在线推理服务接入 LazyLLM 的开发者，提供基于 Online 模块的二次开发指引与示例，涵盖在线对话与在线嵌入两类接口的实现与测试方法。

---

## 1. 整体思路

- 对话类：继承 OnlineChatModuleBase，实现服务鉴权、请求 URL、消息到 prompt 的转换、请求体映射、（可选）流式增量解析与 usage 记录。
- 嵌入类：继承 OnlineEmbeddingModuleBase，实现服务鉴权与响应解析（_parse_response），批量/并发逻辑由基类统一调度。
- 通过少量覆写与映射，即可将任意 HTTP 接口（如与 OpenAI/Ollama 兼容的服务）接入 LazyLLM，复用其提示模板、格式化、流式输出、批处理等能力。

参考实现文件：my_online_build/online_web_deploy.py（见下方“符号索引”）。

---

## 2. 相关基类与职责

- OnlineChatModuleBase：定义在线对话模块的通用流程（构造、鉴权、URL 设定、消息处理、请求发送、流式/非流式解析、usage 统计等）。
- OnlineEmbeddingModuleBase：定义在线嵌入模块的通用流程（构造、鉴权、_encapsulated_data 打包、_parse_response 解析、run_embed/ run_embed_batch 批处理等）。

建议阅读：
- 基类：lazyllm/module/llms/onlinemodule/base/onlineChatModuleBase.py（行 41 起）
- 基类：lazyllm/module/llms/onlinemodule/base/onlineEmbeddingModuleBase.py（行 8 起）

---

## 3. 自定义在线对话模块（CustomChatModule）

文件位置：my_online_build/online_web_deploy.py（行 50 起）

关键点：
- 构造参数：base_url（服务地址）、model（服务端模型名/别名）、system_prompt、stream、return_trace、static_params 等；若本地/内网服务无需鉴权，可设 skip_auth=True。
- 必覆写方法：
  1) _set_headers：设置请求头（无鉴权时至少包含 Content-Type: application/json，若需要鉴权则带上 Authorization）。
  2) _set_chat_url：拼接对话接口 URL（例如 Ollama 为 /api/generate）。
  3) forward：
     - 复用 LazyLLM 的提示模板生成逻辑（self._prompt.generate_prompt）。
     - 将 messages 转换为服务侧需要的 prompt（示例使用 _messages_to_prompt）。
     - 组装请求体（模型名、prompt、是否流式、推理参数 options 等），并通过 requests.post 发送。
     - 流式模式：逐行读取、拼接增量文本、可同时解析 usage（如 prompt_eval_count / eval_count）。
     - 非流式模式：一次性解析完整 JSON，并记录 usage 信息。
     - 可选：get_last_raw_response() 暴露最近一次的原始响应（非流式为完整 JSON，流式为合并文本）。
- 推理参数映射：通过 _map_options 将 LazyLLM 的 static_params（temperature/top_p/max_tokens等）映射为服务端字段（示例中映射为 Ollama options）。

---

## 4. 自定义在线嵌入模块（CustomEmbeddingModule）

文件位置：my_online_build/online_web_deploy.py（行 223 起）

关键点：
- 构造参数：embed_url（嵌入接口地址）、embed_model_name（服务端模型名）、api_key（可选）、return_trace、batch_size、num_worker、timeout、skip_auth 等。
- 覆写方法：
  1) _set_headers：设置请求头；当需要鉴权时在此添加 Authorization: Bearer <api_key>。
  2) _parse_response：解析服务返回为向量或向量列表。
     - 示例兼容 Ollama：
       - 单条：{"embedding": [...]}；
       - 批量：{"embeddings": [[...], ...]}；
     - 亦回退兼容 OpenAI 风格（{"data":[{"embedding":[...]}]}）。
- 批量能力：基类已提供 run_embed / run_embed_batch，利用 batch_size、num_worker、timeout 等参数自动进行分片和并发；自定义类只需专注于解析与鉴权。

---

## 5. 与服务端接口的映射示例（以 Ollama 为例）

- 对话接口：POST /api/generate
  - 请求体要点：{"model": "<name>", "prompt": "<拼接后的文本>", "stream": true/false, "options": {...}}
  - 流式响应：按行返回 JSON，字段如 response（增量文本）、prompt_eval_count、eval_count、done。
- 嵌入接口：POST /api/embed
  - 请求体要点：{"model": "<name>", "input": "<文本或数组>"}
  - 响应：{"embedding": [...]}（单条）或 {"embeddings": [[...], ...]}（批量）。

---

## 6. 测试与使用

- 嵌入测试脚本：项目根目录 custom_embed_test.py（已提供）。
  - 前置：确保服务已启动（如 Ollama 本地或代理服务），embed_url / model 名称正确。
  - 运行：`python custom_embed_test.py`
  - 预期：打印单条与批量嵌入的维度/数量，并将原始响应保存为 embed_check.json，便于对比排查。
- 对话调试：
  - 直接在代码中实例化 CustomChatModule，传入 base_url 与 model，调用实例（或 forward）发起请求；
  - 如接口与 OpenAI 兼容（如 vLLM），也可直接使用 lazyllm.OnlineChatModule 并设置 base_url 指向服务端。

---

## 7. 常见问题与排查

- 401/403：检查是否设置了 Authorization（_set_headers）与 api_key 的传递；skip_auth=False 才会生效。
- 404/路径错误：检查 _set_chat_url 与 embed_url 是否与服务端实际路径一致（是否包含 /api/，是否多/少了斜杠）。
- 流式乱码或不输出：确认服务是否按行返回 JSON；逐行解析失败时，可打印原始行做兼容。
- 维度不一致：不同嵌入模型维度不同，确认服务端模型名称是否正确；可在测试中打印 len(embedding)。
- 批量限流或超时：调节 batch_size、num_worker、timeout；必要时在服务端侧做并发/速率限制配置。

---

## 8. 复用与扩展清单（Checklist）

- 鉴权：是否需要 Authorization 头？是否使用 Bearer 令牌？
- URL：是否已正确设置基础地址与具体路径（generate/embed 等）？
- Prompt：是否正确将消息转为服务端期望的输入（文本或结构体）？
- 流式：服务是否支持流式？是否按行返回 JSON？是否正确拼接与输出？
- 用量：是否从响应中提取 tokens 使用并记录（可选）？
- 嵌入：_parse_response 是否覆盖了单条与批量两种情况，并对 OpenAI 风格做了回退？
- 可靠性：是否设置请求超时？是否处理了非 200 的错误信息以便排查？

---

## 9. 符号索引（便于快速定位源码）

- my_online_build/online_web_deploy.py
  - CustomChatModule（行 50 起）
  - CustomEmbeddingModule（行 223 起）
- lazyllm/module/llms/onlinemodule/base
  - onlineChatModuleBase.py（行 41 起）
  - onlineEmbeddingModuleBase.py（行 8 起）
- 测试脚本
  - 项目根目录 custom_embed_test.py（嵌入调用与对比校验）

---

## 10. 小结

- 对话与嵌入两类在线服务的接入，核心只在于“鉴权 + URL + 请求体/响应体映射 +（可选）流式解析”。
- 通过继承 OnlineChatModuleBase / OnlineEmbeddingModuleBase 可高效复用 LazyLLM 的模板、格式化、流控与批处理能力，最小代价完成对接。
- 如需进一步接入到 RAG 全流程或工具调用，可在此基础上继续扩展。