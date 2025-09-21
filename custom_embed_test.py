import os
import sys
sys.path.append(os.path.abspath('.'))

from my_online_build.online_web_deploy import CustomEmbeddingModule


def main():
    # 按照 embed_use.py 的风格直接访问本地 Ollama /api/embed
    embedder = CustomEmbeddingModule(
        embed_url='http://127.0.0.1:11434/api/embed',
        embed_model_name='nomic-embed-text:v1.5',
        api_key=None,  # 本地默认无鉴权
        batch_size=5,
        num_worker=1,
        timeout=15,
    )

    # 单条输入
    text = '你好世界'
    vec = embedder(text)
    print('Single embedding length:', len(vec))

    # 批量输入
    texts = ['今天心情不错', 'LazyLLM 很强大', 'Ollama embedding 测试']
    vecs = embedder(texts)
    print('Batch size:', len(vecs), 'Each dim:', len(vecs[0]) if vecs else 0)

    # 与 embed_use.py 行为一致，尝试请求原始接口并打印保存
    import requests, json
    url = 'http://127.0.0.1:11434/api/embed'
    payload = {"model": "nomic-embed-text:v1.5", "input": text}
    r = requests.post(url, json=payload)
    with open('embed_check.json', 'w', encoding='utf-8') as f:
        json.dump(r.json(), f, ensure_ascii=False, indent=2)
    print('Saved raw response to embed_check.json')


if __name__ == '__main__':
    main()