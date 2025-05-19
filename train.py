import transformers
from huggingface_hub import login

# insert hugging face token here

tokenizer = transformers.AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
model = transformers.AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")


msg = [{'role': 'user', 'content': 'Hello, how are you?'}]
input_ids = tokenizer.apply_chat_template(msg, tokenize=True)
print('='*100)
print(input_ids)

raw = tokenizer.decode(input_ids, skip_special_tokens=True)
print("="*100)
print(raw)
