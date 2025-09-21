import sys
import os

# 确保可以导入 lazyllm 包与当前目录模块
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import lazyllm
from my_online_build.online_web_deploy import CustomChatModule


def main():
    # 与 gemma_use.py 保持一致：调用本地 Ollama /api/generate
    base_url = 'http://127.0.0.1:11434/api/'  # 请确保本地已启动 ollama 服务，并已拉取 gemma3:1b 模型
    model = 'gemma3:1b'

    llm = CustomChatModule(
        base_url=base_url,
        model=model,
        system_prompt='你是一个乐于助人的中文助手。',
        stream=False,  # 先用非流式，方便对比 gemma_use.py 的完整 JSON
        static_params={
            'temperature': 0.6,
            'num_ctx': 4096,
            'max_tokens': 500,
        },
    )

    # 与 gemma_use 同步：直接给出 instruction 文本
    prompt = '请介绍一下机器学习的基本概念'
    llm.prompt(lazyllm.ChatPrompter(instruction='{query}', extra_keys=['query']))

    res = llm({
        'query': prompt
    })

    print('AI回复:', res)
    raw = llm.get_last_raw_response()
    if raw is not None:
        # 尽量贴近 gemma_use.py 的打印结构
        print(raw)


if __name__ == '__main__':
    main()