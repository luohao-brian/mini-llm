export HF_ENDPOINT=http://hf-mirror.com;
huggingface-cli download Qwen/Qwen-tokenizer --local-dir ./input/Qwen-tokenizer;
huggingface-cli download --repo-type dataset --resume-download stanfordnlp/imdb --local-dir ./input/sft-data ;
huggingface-cli download --repo-type dataset --resume-download trl-lib/ultrafeedback_binarized --local-dir ./input/dpo-data 