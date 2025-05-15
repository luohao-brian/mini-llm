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

下载Qwen2.5-3B作为checkpoint base model

```
huggingface-cli download Qwen/Qwen2.5-3B --local-dir ./input/Qwen2.5-3B/
```

Pretrain

```
python ./pretrain.py --from_ckpt True --model_path ./models/qwen_0.12B.config  --tokenizer_path ./input/Qwen-tokenizer --dataset_path ./input/wikipedia-cn-20230720-filtered --output_dir ./output/pretain/qwen_0.12B --max_length 1024
```

```
bash ./scripts/pretrain_run.sh 
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

example

```
python ./chat_example.py --model_path ./output/pretain/qwen_0.12B --tokenizer_path ./input/Qwen-tokenizer 
```

Project Wanda API Key

```
e7d828cb3382249340d3d1d480acdac7444c0721
```

