"""下载 bge-m3 与 bge-reranker-v2-m3 到本地 models/ 目录。

国内网络建议走 ModelScope；若在海外可改用 huggingface_hub。
用法：python scripts/download_models.py [--root ./models]
"""
from __future__ import annotations

import argparse
from pathlib import Path


def download_modelscope(root: Path) -> None:
    from modelscope import snapshot_download

    bert_patterns = [
        "config.json", "modules.json", "sentence_bert_config.json",
        "special_tokens_map.json", "tokenizer.json", "tokenizer_config.json",
        "vocab.txt", "sentencepiece.bpe.model", "pytorch_model.bin", "1_Pooling/*",
    ]
    rerank_patterns = [
        "config.json", "model.safetensors", "tokenizer.json",
        "tokenizer_config.json", "special_tokens_map.json", "sentencepiece.bpe.model",
    ]
    targets = [
        ("BAAI/bge-m3", bert_patterns, "bge-m3"),
        ("BAAI/bge-reranker-v2-m3", rerank_patterns, "bge-reranker-v2-m3"),
    ]
    for repo, patterns, name in targets:
        print(f"[modelscope] 下载 {repo} -> {root / name}")
        path = snapshot_download(repo, cache_dir=str(root), allow_patterns=patterns, max_workers=8)
        print(f"  完成：{path}")


def download_hf(root: Path) -> None:
    from huggingface_hub import snapshot_download

    for repo, name in [("BAAI/bge-m3", "bge-m3"), ("BAAI/bge-reranker-v2-m3", "bge-reranker-v2-m3")]:
        print(f"[hf] 下载 {repo} -> {root / name}")
        snapshot_download(repo_id=repo, local_dir=str(root / name), local_dir_use_symlinks=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="models", help="模型存放目录")
    parser.add_argument("--source", choices=["modelscope", "hf"], default="modelscope")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    if args.source == "modelscope":
        download_modelscope(root)
    else:
        download_hf(root)
    print("全部完成。请把 .env 中 EMBEDDING_MODEL_PATH / RERANK_MODEL_PATH 指向对应子目录。")


if __name__ == "__main__":
    main()
