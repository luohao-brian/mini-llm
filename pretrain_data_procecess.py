import pathlib
import json
import numpy as np
from loguru import logger
from transformers import Qwen2Tokenizer
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import random

tokenizer = Qwen2Tokenizer.from_pretrained("Qwen/Qwen-tokenizer")
max_seq_length = 32768

BASE_PATH = pathlib.Path("./")
BASE_PRETRAIN_DATA_DIR = BASE_PATH / "pretrain_data"

if not BASE_PRETRAIN_DATA_DIR.exists():
    BASE_PRETRAIN_DATA_DIR.mkdir(parents=True)


def process_item(item):
    content = item.get("completion", "") + "<|endoftext|>"
    # 在调用 encode 之前截断 content
    content = content[:max_seq_length]
    text_ids = tokenizer.encode(content)
    return text_ids if len(text_ids) > 5 else []


def process_wiki():
    wiki_path = (
        BASE_PATH / "input" / "wikipedia-cn-20230720-filtered/wikipedia-cn-20230720-filtered.json"
    )
    wiki_pretrain_path = BASE_PRETRAIN_DATA_DIR / "wiki_pretrain.npy"

    with wiki_path.open("r") as f:
        wiki_data = json.load(f)
    logger.info("成功加载Wiki数据")

    doc_ids = []
    with ProcessPoolExecutor() as executor:
        results = list(tqdm(executor.map(process_item, wiki_data), total=len(wiki_data), desc="处理Wiki数据"))
        for text_ids in results:
            doc_ids += text_ids

    doc_ids_arr = np.array(doc_ids, np.uint32)
    logger.info(f"Wiki token数量为: {len(doc_ids_arr)}")

    with (BASE_PRETRAIN_DATA_DIR / "wiki_token_count.txt").open("w") as f:
        f.write(str(len(doc_ids_arr)))

    np.save(wiki_pretrain_path, doc_ids_arr)
    logger.info("Wiki处理完成")

    # 随机选择3个item进行测试
    test_data = np.load(wiki_pretrain_path)
    random_indices = random.sample(range(len(test_data)), 3)
    for index in random_indices:
        decoded_text = tokenizer.decode([test_data[index]])
        logger.info(f"随机测试解码结果: {decoded_text}")


def main():
    process_wiki()


if __name__ == "__main__":
    main()
