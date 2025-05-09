# chat_example.py
import argparse
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

parser = argparse.ArgumentParser(description='chat example Script')
parser.add_argument('--model_path', type=str, required=True, help='Path to sfted model')
parser.add_argument('--tokenizer_path', type=str, required=True, help='Path to tokenizer dataset')
args = parser.parse_args()

model = AutoModelForCausalLM.from_pretrained(args.model_path)
tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)

questions = ["你是谁？", "你能做什么？"]
for text in questions:
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=50)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))

