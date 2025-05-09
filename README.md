创建环境

```
pip install -r requirements.txt
cd mini-llm
export HF_ENDPOINT=http://hf-mirror.com
```

下载数据集

```
#导入镜像
export HF_ENDPOINT=http://hf-mirror.com
#tokenizer
huggingface-cli download Qwen/Qwen-tokenizer --local-dir ./input/Qwen-tokenizer
#SFT
huggingface-cli download --repo-type dataset --resume-download stanfordnlp/imdb --local-dir ./input/sft-data 
#DPO
huggingface-cli download --repo-type dataset --resume-download trl-lib/ultrafeedback_binarized --local-dir ./input/dpo-data 
```

Pretrain

```
python ./pretrain.py --model_path ./models/qwen_0.12B.config  --tokenizer_path ./input/Qwen-tokenizer --dataset_path ./pretrain_data/wiki_pretrain.npy --output_dir ./output/pretain/qwen_0.12B --max_length 1024
```

SFT

```
python ./sft.py --model_path ./output/pretain/qwen_0.12B --tokenizer_path ./output/pretain/qwen_0.12B --dataset_path ./input/sft-data --output_dir ./output/sft/qwen_0.12B --max_length 1024
```

```
 bash ./scripts/sft_run.sh 
```

DPO

```
python ./dpo.py --model_path ./output/sft/qwen_0.12B --tokenizer_path ./output/sft/qwen_0.12B --dataset_path ./input/dpo-data --output_dir ./output/dpo/qwen_0.12B --max_length 1024
```

```
 bash ./scripts/dpo_run.sh 
```

可能会出现如下虚假警告(https://github.com/huggingface/transformers/pull/36316)

```
Sliding Window Attention is enabled but not implemented for `sdpa`; unexpected results may be encountered.
```

