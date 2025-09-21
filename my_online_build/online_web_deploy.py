


import lazyllm
import requests
from urllib.parse import urljoin
from typing import Optional, Dict, Any, List, Union
from lazyllm.module import OnlineChatModuleBase, OnlineEmbeddingModuleBase
# from lazyllm.module.llms.onlinemodule.fileHandler import FileHandlerBase  # 如需微调能力再引入
class CustomChatModule(OnlineChatModuleBase):
    def __init__(self,
        base_url: str = "<new platform base url>",
        model: str = "<new platform model name>",
        system_prompt: str = "<new platform system prompt>",
        stream: bool = True,
        return_trace: bool = False,
        static_params: Optional[Dict[str, Any]] = None,
        type: Optional[str] = None,
    ):
        # 保存系统提示词
        self._custom_system_prompt = system_prompt or ""
        # 调用基类构造：
        # - model_series: 供应商标识（自定义即可）
        # - api_key: 无鉴权可留空，并设置 skip_auth=True
        # - model_name: 模型名称/别名
        # - base_url: 服务基础地址（Ollama 推荐以 /api/ 结尾）
        # - stream/return_trace: 与 LazyLLM 一致
        # - static_params: 固定推理参数（temperature/top_p/max_tokens 等）
        super().__init__(
            model_series="custom_chat",
            api_key="",
            base_url=base_url,
            model_name=model,
            stream=stream,
            return_trace=return_trace,
            skip_auth=True,
            static_params=static_params,
            type=type,
        )

    def _get_system_prompt(self):
        return self._custom_system_prompt

    # ---- 以下为 Ollama /api/generate 兼容实现 ----
    def _set_headers(self):
        # 本地服务通常无需鉴权
        self._headers = {
            'Content-Type': 'application/json'
        }

    def _set_chat_url(self):
        # base_url 如 http://127.0.0.1:11434/api/ -> 拼接 generate 为 /api/generate
        self._url = urljoin(self._base_url, 'generate')

    @staticmethod
    def _messages_to_prompt(messages: List[Dict[str, Any]]) -> str:
        if not messages:
            return ''
        parts: List[str] = []
        role_map = {
            'system': 'System',
            'user': 'User',
            'assistant': 'Assistant'
        }
        for m in messages:
            role = role_map.get(m.get('role', 'user'), 'User')
            content = m.get('content', '')
            if isinstance(content, list):
                content = '\n'.join([c.get('text', '') if isinstance(c, dict) else str(c) for c in content])
            parts.append(f"{role}: {content}")
        parts.append('Assistant:')
        return '\n'.join(parts)

    @staticmethod
    def _map_options(static_params: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
        # 将 LazyLLM 的静态参数映射为 Ollama 的 options
        mapping = {
            'temperature': 'temperature',
            'top_p': 'top_p',
            'top_k': 'top_k',
            'max_tokens': 'num_predict',
            'frequency_penalty': 'repeat_penalty',
            'presence_penalty': 'presence_penalty',
        }
        opts: Dict[str, Any] = {}
        for k, v in (static_params or {}).items():
            if k in mapping and v is not None:
                opts[mapping[k]] = v
        direct_keys = [
            'stop', 'penalize_newline', 'min_p', 'tfs_z', 'repeat_penalty', 'presence_penalty',
            'num_ctx', 'mirostat', 'mirostat_eta', 'mirostat_tau', 'repeat_last_n', 'seed',
        ]
        for k in direct_keys:
            if static_params and k in static_params and static_params[k] is not None:
                opts[k] = static_params[k]
        for k in direct_keys:
            if overrides and k in overrides and overrides[k] is not None:
                opts[k] = overrides[k]
        return opts

    def forward(self, __input: Union[Dict, str] = None, *, llm_chat_history: List[List[str]] = None,
                tools: List[Dict[str, Any]] = None, stream_output: bool = False, lazyllm_files=None, **kw):
        # 复用 LazyLLM 的提示模板生成逻辑
        stream_output = stream_output or self._stream
        __input, files = self._get_files(__input, lazyllm_files)
        params = {'input': __input, 'history': llm_chat_history, 'return_dict': True}
        if tools: params['tools'] = tools
        data = self._prompt.generate_prompt(**params)

        # 将 messages 拼接为纯文本 prompt
        messages = data.get('messages', [])
        if files and isinstance(messages, list) and len(messages) > 0:
            last = messages[-1]
            if isinstance(last.get('content'), str):
                last['content'] = self._format_input_with_files(last['content'], files)
        prompt_str = self._messages_to_prompt(messages) if messages else str(data.get('input', ''))

        # 组装 Ollama 请求体
        payload: Dict[str, Any] = {
            'model': self._model_name,
            'prompt': prompt_str,
            'stream': bool(stream_output),
            'options': self._map_options(self._static_params, kw)
        }
        for top in ['keep_alive', 'system', 'template', 'context']:
            if top in kw and kw[top] is not None:
                payload[top] = kw[top]

        self._last_raw_response: Dict[str, Any] = {}
        with requests.post(self._url, json=payload, headers=self._headers, stream=stream_output) as r:
            if r.status_code != 200:
                msg = '\n'.join([c.decode('utf-8') for c in r.iter_content(None)]) if stream_output else r.text
                raise requests.RequestException(f'{r.status_code}: {msg}')

            total_text_parts: List[str] = []
            usage = {'prompt_tokens': -1, 'completion_tokens': -1}

            if stream_output:
                with self.stream_output(stream_output):
                    color = stream_output.get('color') if isinstance(stream_output, dict) else None
                    for line in r.iter_lines():
                        if not line:
                            continue
                        try:
                            obj = lazyllm.loads(line.decode('utf-8')) if hasattr(lazyllm, 'loads') else None
                        except Exception:
                            obj = None
                        if obj is None:
                            try:
                                import json
                                obj = json.loads(line.decode('utf-8'))
                            except Exception:
                                continue
                        chunk = obj.get('response', '')
                        if chunk:
                            total_text_parts.append(chunk)
                            self._stream_output(chunk, color)
                        if 'prompt_eval_count' in obj or 'eval_count' in obj:
                            usage['prompt_tokens'] = obj.get('prompt_eval_count', usage['prompt_tokens'])
                            usage['completion_tokens'] = obj.get('eval_count', usage['completion_tokens'])
                        if obj.get('done'):
                            break
                text = ''.join(total_text_parts)
                # 流式无法拿到完整原始对象，仅保留合并后的文本
                self._last_raw_response = {'response': text}
            else:
                obj = r.json()
                self._last_raw_response = obj
                text = obj.get('response', '')
                if 'prompt_eval_count' in obj or 'eval_count' in obj:
                    usage['prompt_tokens'] = obj.get('prompt_eval_count', usage['prompt_tokens'])
                    usage['completion_tokens'] = obj.get('eval_count', usage['completion_tokens'])

            # 记录 tokens 使用信息
            self._record_usage(usage)
            return self._formatter(text) if text else ''

    # 便捷方法：获取最近一次请求的原始响应（非流式模式下完整 JSON，流式模式下仅含合并文本）
    def get_last_raw_response(self) -> Optional[Dict[str, Any]]:
        return getattr(self, '_last_raw_response', None)

# ------------------ 自定义在线嵌入模块 ------------------
class CustomEmbeddingModule(OnlineEmbeddingModuleBase):
    """
    基于 OnlineEmbeddingModuleBase 的自定义在线嵌入实现。
    默认兼容本地 Ollama /api/embed 接口；若服务需要鉴权，可传入 api_key 并将 skip_auth 置为 False。
    """
    NO_PROXY = True

    def __init__(self,
                 embed_url: str = 'http://127.0.0.1:11434/api/embed',
                 embed_model_name: str = 'nomic-embed-text:v1.5',
                 api_key: Optional[str] = None,
                 return_trace: bool = False,
                 batch_size: int = 10,
                 num_worker: int = 1,
                 timeout: int = 10,
                 skip_auth: bool = True,
                 ):
        self._skip_auth = bool(skip_auth or not api_key)
        super().__init__(
            model_series='custom_embed',
            embed_url=embed_url.strip() if embed_url else '',
            api_key=(api_key or ''),
            embed_model_name=embed_model_name,
            return_trace=return_trace,
            batch_size=batch_size,
            num_worker=num_worker,
            timeout=timeout,
        )

    def _set_headers(self) -> Dict[str, str]:
        headers = {'Content-Type': 'application/json'}
        if not getattr(self, '_skip_auth', False) and getattr(self, '_api_key', ''):
            headers['Authorization'] = f'Bearer {self._api_key}'
        self._headers = headers
        return headers

    def _parse_response(self, response: Dict, input: Union[List, str]) -> Union[List[List[float]], List[float]]:
        # 兼容 Ollama 风格：{"embeddings": [[...], ...]} 或单条 {"embedding": [...]}
        if 'embeddings' in response:
            embs = response.get('embeddings') or []
            if isinstance(input, str):
                return embs[0] if embs else []
            return embs
        if 'embedding' in response:
            vec = response.get('embedding') or []
            if isinstance(input, str):
                return vec
            # 传入为列表时，某些实现也可能返回单向量；做最小兼容
            return [vec]
        # 回退到基类的 OpenAI 风格解析（包含 data: [{embedding: [...]}]）
        return super()._parse_response(response, input)