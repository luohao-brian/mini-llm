export HF_ENDPOINT=http://hf-mirror.com;
huggingface-cli download Qwen/Qwen-tokenizer --local-dir ./input/Qwen-tokenizer;
huggingface-cli download --repo-type dataset --resume-download Skywork/SkyPile-150B --local-dir ./input/SkyPile-150B;
huggingface-cli download --repo-type dataset --resume-download stanfordnlp/imdb --local-dir ./input/;
huggingface-cli download --repo-type dataset --resume-download trl-lib/ultrafeedback_binarized --local-dir ./input/dpo-data 